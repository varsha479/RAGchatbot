
import chromadb

from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


class RAGChatbot:

    def __init__(self, api_key):

        # 1. Load embedding model
        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # 2. Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection_name = "pdf_documents"

        # 3. Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )

        self.document_name = None

    # =====================================
    # PART 1: SPLIT TEXT INTO CHUNKS
    # =====================================

    def split_text(self, text, chunk_size=800, overlap=120):

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    # =====================================
    # PART 2: PROCESS PDF
    # =====================================

    def process_pdf(self, pdf_bytes, filename):

        reader = PdfReader(BytesIO(pdf_bytes))

        all_chunks = []
        all_metadata = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):

            text = page.extract_text() or ""

            if not text.strip():
                continue

            page_chunks = self.split_text(text)

            for chunk in page_chunks:

                all_chunks.append(chunk)

                all_metadata.append({
                    "source": filename,
                    "page": page_number
                })

        if not all_chunks:
            raise ValueError(
                "No readable text found in this PDF. "
                "Scanned PDFs require OCR."
            )

        # Generate embeddings
        embeddings = self.embedder.encode(
            all_chunks,
            show_progress_bar=False
        ).tolist()

        # Remove previous collection
        try:
            self.client.delete_collection(
                self.collection_name
            )
        except Exception:
            pass

        # Create new collection
        collection = self.client.create_collection(
            self.collection_name
        )

        # Store chunks, embeddings and metadata
        collection.add(
            ids=[
                f"chunk-{i}"
                for i in range(len(all_chunks))
            ],
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadata
        )

        self.document_name = filename

        return len(all_chunks)

    # =====================================
    # PART 3: EXTRACT GEMINI TEXT
    # =====================================

    def extract_response_text(self, response):

        content = response.content

        # Case 1: Normal string response
        if isinstance(content, str):
            return content.strip()

        # Case 2: Structured content blocks
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, str):

                    text_parts.append(item)

                elif isinstance(item, dict):

                    if item.get("type") == "text":

                        text = item.get("text", "")

                        if isinstance(text, str):
                            text_parts.append(text)

            return "\n".join(text_parts).strip()

        return ""

    # =====================================
    # PART 4: ANSWER QUESTIONS
    # =====================================

    def answer_question(self, question, k=4):

        # Get collection
        try:

            collection = self.client.get_collection(
                self.collection_name
            )

        except Exception:

            return (
                "Please upload and process a PDF first.",
                []
            )

        total_chunks = collection.count()

        if total_chunks == 0:

            return (
                "Please upload and process a PDF first.",
                []
            )

        # Convert question into embedding
        question_embedding = self.embedder.encode(
            [question]
        ).tolist()

        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=question_embedding,
            n_results=min(k, total_chunks),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results["documents"][0]
        metadata = results["metadatas"][0]

        if not documents:

            return (
                "I cannot find that information in the document.",
                []
            )

        # Prepare context
        context_parts = []

        evidence = []

        for document, meta in zip(
            documents, metadata
        ):

            context_parts.append(
                f"Page {meta['page']}:\n{document}"
            )

            evidence.append({
                "content": document,
                "page": meta["page"],
                "source": meta["source"]
            })

        context = "\n\n".join(context_parts)

        # Create prompt
        prompt = f"""
You are a PDF document question-answering assistant.

STRICT RULES:

1. Answer using ONLY the provided document context.
2. Do not use outside knowledge.
3. Do not invent facts or information.
4. If the answer is not available, respond exactly:
"I cannot find that information in the document."
5. Give a clear and simple answer.
6. Mention page numbers when useful.
7. Do not include technical metadata in your answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        # Generate Gemini response
        response = self.llm.invoke(prompt)

        # Extract only actual text
        answer = self.extract_response_text(
            response
        )

        if not answer:
            answer = (
                "I cannot generate an answer from the document."
            )

        return answer, evidence
