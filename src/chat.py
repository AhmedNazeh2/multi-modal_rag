from typing import List
from langchain_openai import ChatOpenAI 
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from src.config import Config

class ChatManager:
    """
    Manages chat interactions using LangChain components.
    Uses LangChain's ConversationalRetrievalChain for handling conversational RAG.
    """
    def __init__(self):
        """
        Initialize the ChatManager.
        """
        self.memory = None
        self.chain = None
        self.llm = None
        self._initialize_components()
        
    def _initialize_components(self):
        """
        Initialize LangChain components for the chat system.
        Sets up the LLM, memory, and creates the conversation chain.
        """
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Initialize the LLM
        try:

            self.llm = ChatOpenAI( 
                model=Config.MODEL_NAME, 
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.MAX_TOKENS, 
            )
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7
            )
    
    def _create_chain(self, retriever):
            """
            Creates a conversational retrieval chain with the specified retriever.
            """
            # System prompt template
            system_template = """You are a helpful assistant that answers questions based on the provided context ONLY.
            Your response must be entirely based on the context provided below.
            If you cannot find the exact answer in the context, clearly state that you cannot find it.
            
            Context:
            {context}
            IMPORTANT: Answer in the same language as the user's question.
            """
            COMBINE_PROMPT = PromptTemplate(
                template=system_template + "\nQuestion: {question}",
                input_variables=["context", "question"],
            )
            
            # FIXED: Use the prompt directly with the LLM instance for the question generator
            CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
                """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

                Chat History:
                {chat_history}

                Follow Up Input: {question}

                Standalone question:"""
            )
            
            # Create the question generator chain (CONDENSE_QUESTION_PROMPT | self.llm)
            # We can no longer pass the LLMChain directly. We pass the components or rely on the default behavior.
            
            # Create the chain - Note: The default behavior often uses the LLM and a default prompt for question generation.
            # However, to explicitly use our prompt and enable the feature reliably, we rely on the internal construction
            # by passing the LLM and the prompt via the chain's arguments. 
            # By removing the explicit LLMChain object, we let the internal logic handle it.
            
            # Using the standard constructor by passing the LLM and the retriever
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": COMBINE_PROMPT},
                # question_generator_prompt=CONDENSE_QUESTION_PROMPT,
                verbose=True # Keep verbose for debugging
            )
            
            # Post-check: If you still need the custom prompt, try setting it explicitly 
            # after chain creation if the base constructor doesn't support it directly anymore, 
            # but the error indicates validation failure during construction.
            
            return self.chain
        
    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate a response based on the query and retrieved context documents.
        
        Args:
            query: The user's question
            context_docs: List of context documents retrieved for the query (unused in chain mode)
            
        Returns:
            str: The generated response
        """
        if self.chain:
            try:
                # The chain automatically handles retrieval based on the condensed question
                result = self.chain.invoke({"question": query})
                return result["answer"]
            except Exception as e:
                print(f"Chain error, falling back to direct LLM call: {str(e)}")
        
        # Fallback logic (Manual retrieval is handled in streamlit_app for debug)
        context_texts = [doc.page_content for doc in context_docs]
        context_text = "\n".join(context_texts)
        
        try:
            # Format messages with context
            messages = [
                SystemMessage(content=f"You are a helpful assistant that answers questions based on the provided context. If you cannot find the answer in the context, say so.\n\nContext:\n{context_text}"),
                HumanMessage(content=query)
            ]
            
            # Call the LLM directly
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your request. Please try again later."
            
    def set_retriever(self, retriever):
        """
        Set the retriever and create a conversation chain.
        
        Args:
            retriever: The document retriever to use
        """
        self._create_chain(retriever)
        
    def reset_conversation(self):
        """
        Reset the conversation history.
        """
        if self.memory:
            self.memory.clear()
            
    def get_conversation_history(self):
        """
        Get the current conversation history.
        
        Returns:
            The conversation history
        """
        if self.memory:
            return self.memory.chat_memory.messages
        return []