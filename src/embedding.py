from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import Config
from langchain_openai import OpenAIEmbeddings

class EmbeddingManager:
    """
    Manages embeddings and retrieval using LangChain components.
    Uses OpenAIEmbeddings for embeddings and FAISS for vector storage.
    """
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings( 
            model=Config.EMBEDDING_MODEL 
        )
        self.vectorstore = None
        self.retriever = None

    def create_embeddings(self, documents: List[Document]):
            """
            Creates embeddings. If vectorstore exists, it adds to it. If not, creates new.
            """
            if not documents:
                return False
                
            try:
                if self.vectorstore:
                    print("EmbeddingManager: Adding documents to existing vectorstore...")
                    self.vectorstore.add_documents(documents)
                else:
                    print("EmbeddingManager: Creating new vectorstore...")
                    self.vectorstore = FAISS.from_documents(
                        documents, 
                        self.embedding_model
                    )
                
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": Config.TOP_K}
                )
                
                return True
            except Exception as e:
                print(f"Error creating/updating embeddings: {str(e)}")
                return False
            
            
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Searches for relevant documents based on the query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve (defaults to Config.TOP_K)
            
        Returns:
            List[Document]: A list of relevant Document objects
        """
        if not k:
            k = Config.TOP_K
            
        if not self.vectorstore:
            return []
            
        try:
            relevant_docs = self.vectorstore.similarity_search(query, k=k)
            return relevant_docs
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
            
    def clear_embeddings(self):
        """
        Clears the current vector store and retriever.
        """
        self.vectorstore = None
        self.retriever = None