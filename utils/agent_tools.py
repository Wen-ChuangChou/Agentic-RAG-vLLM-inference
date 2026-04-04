import logging
from langchain_core.vectorstores import VectorStore
from smolagents import Tool
from typing import Optional


class RetrieverTool(Tool):
    name = "retriever"
    # description = """
    # Retrieves relevant documents from the knowledge base using semantic similarity search.
    # Choose k based on query complexity:
    # - k=5-7: For standard questions needing some context
    # - k=8-10: For complex topics requiring multiple perspectives
    # - k=10-13: For comprehensive research or broad exploration
    # """

    # inputs = {
    #     "query": {
    #         "type":
    #         "string",
    #         "description":
    #         "The search query. Use descriptive keywords and phrases rather than questions. Example: 'machine learning algorithms' instead of 'what are ML algorithms?'",
    #     },
    #     "k": {
    #         "type": "integer",
    #         "description":
    #         "Number of documents to retrieve (7-20). Start with 7, increase if no relevant information is found. Default: 7 if not specified.",
    #         "nullable": True
    #     }
    # }

    description = """
    Retrieves relevant documents from the knowledge base using semantic similarity search.
    Choose k based on query complexity.
    """

    inputs = {
        "query": {
            "type":
            "string",
            "description":
            "The search query. Use descriptive keywords and phrases rather than questions. Example: 'machine learning algorithms' instead of 'what are ML algorithms?'",
        },
        "k": {
            "type": "integer",
            "description":
            "Number of documents to retrieve (3-8). Start with 3, increase if no relevant information is found. Default: 3 if not specified.",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self,
                 vectordb: VectorStore,
                 k: int = 7,
                 score_threshold: Optional[float] = None,
                 max_content_length: int = 10000,
                 **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.k = k
        self.score_threshold = score_threshold
        self.max_content_length = max_content_length
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, query: str, k: Optional[int] = None) -> str:
        # Input validation
        if not query or not isinstance(query, str) or not query.strip():
            return "Error: Query must be a non-empty string"

        # Parameter validation
        retrieval_k = k if k is not None else self.k
        if retrieval_k is not None and (not isinstance(retrieval_k, int) or
                                        retrieval_k < 3 or retrieval_k > 20):
            retrieval_k = self.k

        query = query.strip()

        try:
            # Try to get documents with scores if available
            if hasattr(self.vectordb, 'similarity_search_with_score'
                       ) and self.score_threshold is not None:
                docs_with_scores = self.vectordb.similarity_search_with_score(
                    query, k=retrieval_k)
                # Filter by score threshold
                filtered_docs = [
                    doc for doc, score in docs_with_scores
                    if score >= self.score_threshold
                ]
                docs = filtered_docs[:retrieval_k] if filtered_docs else []

                if not docs and docs_with_scores:
                    # If no docs meet threshold, take the best one anyway
                    docs = [docs_with_scores[0][0]]

            else:
                docs = self.vectordb.similarity_search(query, k=retrieval_k)

        except Exception as e:
            self.logger.error(f"Error during similarity search: {str(e)}")
            return f"Error retrieving documents: {str(e)}"

        # Handle no results
        if not docs:
            return f"No relevant documents found for query: '{query}'"

        # Format results
        return self._format_results(query, docs)

    def _format_results(self, query: str, docs) -> str:
        """Format the retrieved documents into a readable string."""
        result_parts = [
            f"Search Query: {query}",
            f"Retrieved {len(docs)} relevant document(s):", ""
        ]

        for i, doc in enumerate(docs, 1):
            # Add document header
            result_parts.append(f"=== Document {i} ===")

            # Add metadata if available
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata_info = []
                if 'source' in doc.metadata:
                    metadata_info.append(f"Source: {doc.metadata['source']}")
                if 'title' in doc.metadata:
                    metadata_info.append(f"Title: {doc.metadata['title']}")
                if 'page' in doc.metadata:
                    metadata_info.append(f"Page: {doc.metadata['page']}")

                if metadata_info:
                    result_parts.append(" | ".join(metadata_info))

            # Add content (with truncation if needed)
            content = doc.page_content
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length].rsplit(
                    ' ', 1)[0] + "... [truncated]"

            result_parts.append(content)
            result_parts.append("")  # Empty line between documents

        return "\n".join(result_parts).strip()

    def get_retrieval_stats(self) -> dict:
        """Get basic stats about the vector store if available."""
        try:
            # This depends on the vector store implementation
            if hasattr(self.vectordb, '__len__'):
                return {"total_documents": len(self.vectordb)}
            return {"status": "Vector store connected"}
        except Exception:
            return {"status": "Unable to retrieve stats"}
