from typing import List
import os
from langchain_community.document_loaders import UnstructuredFileLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.config import Config
import warnings
warnings.filterwarnings("ignore")

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
        
    def split_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)
    
    def process_url(self, url: str) -> List[Document]:
        """
        Loads content from a URL, processes it, and splits it into chunks.
        """
        try:
            print(f"Processor: Loading content from URL: {url}")
            loader = WebBaseLoader(url,
                                   header_template={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Referer': 'https://www.google.com/'
                })
            initial_documents = loader.load()
            if initial_documents:
                print("\n--- DEBUG: Content Start ---")
                print(initial_documents[0].page_content[:500]) 
                print("--- DEBUG: Content End ---\n")
            if not initial_documents:
                print("Warning: No content found in URL.")
                return []
                
            documents = self.text_splitter.split_documents(initial_documents)
            
            final_documents = []
            for i, doc in enumerate(documents):
                doc.metadata["source"] = url
                doc.metadata["chunk_id"] = i
                final_documents.append(doc)
                
            print(f"Processor: Successfully processed {len(final_documents)} chunks from URL.")
            return final_documents
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return []
    
    def process_document(self, uploaded_file) -> List[Document]:
        
        # streamlit
        # file_name = uploaded_file.name
        # temp_file_path = f"./temp_{file_name}"
        # try:
        #     with open(temp_file_path, "wb") as f:
        #         f.write(uploaded_file.getbuffer())
        # except Exception as e:
        #     print(f"Error saving temporary file: {e}")
        #     return []        
                
        file_name = uploaded_file.filename
        temp_file_path = f"./temp_{file_name}"
        
        try:
            file_content = uploaded_file.file.read()
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
        except Exception as e:
            print(f"Error saving temporary file: {e}")
            return []

        try:
            loader = UnstructuredFileLoader(
                temp_file_path, 
                mode="elements", 
                strategy="hi_res", 
                languages=["ara", "eng"],
                extract_tables=True
            )
            
            initial_documents = loader.load()
            
            # ADDED DEBUG PRINT: Check content of initial documents/elements
            print(f"Processor Debug: Total initial documents/elements extracted from {file_name}: {len(initial_documents)}")
            if initial_documents:
                print("\n--- Initial Document Content Preview (DEBUG) ---")
                print(f"Content Preview: {initial_documents[0].page_content[:200]}...")
                print("------------------------------------------------\n")
            
            if not initial_documents:
                print(f"Warning: No documents/elements extracted from {file_name}.")
                return []
                
            documents = self.text_splitter.split_documents(initial_documents)
            
            # ADDED DEBUG PRINT: Check number of final chunks
            print(f"Processor Debug: Total chunks created for {file_name}: {len(documents)}")

            if not documents:
                print(f"Warning: Document splitting resulted in zero chunks.")
                return []
                
            final_documents = [
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": file_name,
                        "chunk_id": i
                    }
                ) for i, doc in enumerate(documents)
            ]
            
            return final_documents
            
        except Exception as e:
            print(f"Error processing document {file_name}: {str(e)}")
            return []
        finally:
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")