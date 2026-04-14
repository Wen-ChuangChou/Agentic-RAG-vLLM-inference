"""
Utilities for managing and creating FAISS vector databases from Hugging Face datasets.
Incluldes tools for document splitting, deduplication, and batch embedding.
"""
import datasets
import os
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List


def sanitize_filename(dataset_name: str) -> str:
    """Convert dataset name to a valid filename by replacing '/' with '_'"""
    return dataset_name.replace("/", "_")


class DocumentProcessor:
    """
    A thread-safe class for splitting documents into chunks.
    This class ensures that the text splitter is initialized only once per thread
    to avoid pickling issues during parallel processing.
    """

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        """
        Initialize the processor with chunk size and overlap.
        
        Args:
            chunk_size: The number of characters per chunk.
            chunk_overlap: The number of characters to overlap between chunks.
        """
        self.text_splitter = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._lock = threading.Lock()

    def _get_text_splitter(self):
        """
        Initialize and return the text splitter in a thread-safe manner.
        Uses RecursiveCharacterTextSplitter with a HuggingFace tokenizer.
        """
        if self.text_splitter is None:
            with self._lock:
                if self.text_splitter is None:
                    # Use a pre-trained tokenizer for accurate chunking
                    self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                        AutoTokenizer.from_pretrained("thenlper/gte-small"),
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        add_start_index=True,
                        strip_whitespace=True,
                        separators=["\n\n", "\n", ".", " ", ""],
                    )
        return self.text_splitter

    def split_documents_chunk(self,
                              docs_chunk: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        Args:
            docs_chunk: A list of Document objects to be split.

        Returns:
            A list of split Document objects.
        """
        text_splitter = self._get_text_splitter()
        processed_docs = []
        for doc in docs_chunk:
            new_docs = text_splitter.split_documents([doc])
            processed_docs.extend(new_docs)
        return processed_docs


def parallel_document_splitting(
        source_docs: List[Document],
        max_workers: int = None,
        chunk_size: int = 100,
        text_chunk_size: int = 200,
        text_chunk_overlap: int = 20) -> List[Document]:
    """
    Split a list of documents into chunks using multiple threads.

    Args:
        source_docs: The list of documents to split.
        max_workers: Maximum number of worker threads (defaults based on doc count).
        chunk_size: Number of documents to process in each thread.
        text_chunk_size: Characters per text chunk.
        text_chunk_overlap: Overlap between text chunks.

    Returns:
        List of all processed document chunks.
    """
    if max_workers is None:
        max_workers = min(8, len(source_docs) // chunk_size + 1)

    print(f"Splitting documents in parallel using {max_workers} threads...")

    # Split source_docs into chunks for parallel processing
    doc_chunks = [
        source_docs[i:i + chunk_size]
        for i in range(0, len(source_docs), chunk_size)
    ]

    # Create a shared processor instance with configurable chunk size
    processor = DocumentProcessor(text_chunk_size, text_chunk_overlap)

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process chunks in parallel with progress bar
        results = list(
            tqdm(executor.map(processor.split_documents_chunk, doc_chunks),
                 total=len(doc_chunks),
                 desc="Processing document chunks"))

    # Flatten results
    all_processed_docs = []
    for chunk_result in results:
        all_processed_docs.extend(chunk_result)

    return all_processed_docs


def remove_duplicates(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents from a list based on their text content.

    Args:
        docs: List of Document objects to check.

    Returns:
        List of unique Document objects.
    """
    print("Removing duplicate documents...")
    unique_texts = set()
    unique_docs = []

    for doc in tqdm(docs, desc="Checking for duplicates"):
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            unique_docs.append(doc)

    print(f"Removed {len(docs) - len(unique_docs)} duplicate documents")
    return unique_docs


def batch_embed_documents(docs: List[Document],
                          embedding_model,
                          batch_size: int = 100) -> FAISS:
    """
    Create a FAISS vectorstore by embedding documents in batches.
    
    Args:
        docs: List of documents to embed.
        embedding_model: The model to use for generating embeddings.
        batch_size: Number of documents to process in each batch.

    Returns:
        An initialized FAISS vector database.
    """
    print(f"Embedding documents in batches of {batch_size}...")

    if not docs:
        raise ValueError("No documents to embed")

    # Create initial vectorstore with first batch
    first_batch = docs[:batch_size]
    print(f"Creating initial vectorstore with {len(first_batch)} documents...")

    vectordb = FAISS.from_documents(
        documents=first_batch,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Process remaining documents in batches
    remaining_docs = docs[batch_size:]
    if remaining_docs:
        total_batches = (len(remaining_docs) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(remaining_docs), batch_size),
                      desc="Adding document batches",
                      total=total_batches):
            batch = remaining_docs[i:i + batch_size]
            vectordb.add_documents(batch)

    return vectordb


def sequential_document_splitting(
        source_docs: List[Document],
        text_chunk_size: int = 200,  # Add this parameter
        text_chunk_overlap: int = 20) -> List[Document]:
    """
    Split documents one by one. Used as a fallback if parallel processing fails.

    Args:
        source_docs: The list of documents to split.
        text_chunk_size: Characters per text chunk.
        text_chunk_overlap: Overlap between text chunks.

    Returns:
        List of all processed document chunks.
    """
    print("Using sequential document splitting...")

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
        chunk_size=text_chunk_size,
        chunk_overlap=text_chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in tqdm(source_docs, desc="Splitting documents"):
        new_docs = text_splitter.split_documents([doc])
        docs_processed.extend(new_docs)

    return docs_processed


def load_or_create_vectordb(dataset_name: str,
                            batch_size: int = 100,
                            max_workers: int = None,
                            doc_chunk_size: int = 100,
                            text_chunk_size: int = 200,
                            text_chunk_overlap: int = 20,
                            force_rebuild: bool = False,
                            use_parallel: bool = True,
                            embedding_device: str = "cpu") -> FAISS:
    """
    Main entry point to get a FAISS vector database.
    Tries to load an existing database from disk, otherwise creates and saves a new one.

    Args:
        dataset_name: Hugging Face dataset identifier.
        batch_size: Documents per embedding batch.
        max_workers: Threads for document splitting.
        doc_chunk_size: Documents per split task.
        text_chunk_size: Chunks character size.
        text_chunk_overlap: Chunks character overlap.
        force_rebuild: If True, always recreate the database.
        use_parallel: If True, use multi-threading for splitting.

    Returns:
        FAISS: Loaded or created vector database.
    """

    # Create sanitized filename
    safe_filename = sanitize_filename(dataset_name)
    vectordb_path = os.path.join(
        "vectordb",
        f"{safe_filename}_{text_chunk_size}")  # Include chunk size in filename

    # Check if vectordb already exists and not forcing rebuild
    if os.path.exists(vectordb_path) and not force_rebuild:
        print(f"Found existing vector database at {vectordb_path}")
        print("Loading existing vector database...")

        try:
            # Load the embedding model (needed for loading)
            # Use specified device (defaults to cpu to avoid vLLM conflicts)
            embedding_model = HuggingFaceEmbeddings(
                model_name="thenlper/gte-small",
                model_kwargs={"device": embedding_device})

            # Load the existing vectordb
            vectordb = FAISS.load_local(vectordb_path,
                                        embedding_model,
                                        allow_dangerous_deserialization=True)
            print("Vector database loaded successfully!")
            return vectordb

        except Exception as e:
            print(f"Error loading existing vectordb: {e}")
            print("Creating new vector database...")

    else:
        if force_rebuild:
            print("Force rebuild requested. Creating new vector database...")
        else:
            print(f"No existing vector database found at {vectordb_path}")
            print("Creating new vector database...")

    # Load and process the dataset
    print("Loading dataset...")
    knowledge_base = datasets.load_dataset(dataset_name, split="train")
    source_docs = [
        Document(page_content=doc["text"],
                 metadata={"source": doc["source"].split("/")[1]})
        for doc in tqdm(knowledge_base, desc="Processing dataset")
    ]

    print(f"Loaded {len(source_docs)} documents from dataset")

    # Split documents (with fallback to sequential processing)
    try:
        if use_parallel and len(source_docs) > 50:
            docs_processed = parallel_document_splitting(
                source_docs, max_workers, doc_chunk_size, text_chunk_size,
                text_chunk_overlap)
        else:
            docs_processed = sequential_document_splitting(
                source_docs, text_chunk_size, text_chunk_overlap)

    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        docs_processed = sequential_document_splitting(
            source_docs, text_chunk_size,
            text_chunk_overlap)  # Pass new parameters

    print(f"Split into {len(docs_processed)} document chunks")

    # Remove duplicates
    docs_processed = remove_duplicates(docs_processed)

    print(f"Final document count after deduplication: {len(docs_processed)}")

    # Create embedding model
    print(f"Using device: {embedding_device} to load embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small",
                                            model_kwargs={"device": embedding_device})

    # Create vectordb with batch processing
    vectordb = batch_embed_documents(docs_processed, embedding_model,
                                     batch_size)

    # Save the vectordb
    print(f"Saving vector database to {vectordb_path}...")
    try:
        vectordb.save_local(vectordb_path)
        print("Vector database saved successfully!")
    except Exception as e:
        print(f"Error saving vectordb: {e}")
        print("Continuing without saving...")

    return vectordb
