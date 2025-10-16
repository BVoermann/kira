from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGEngine:
    def __init__(self, vectorstore, model_name="mistral"):
        self.vectorstore = vectorstore

        # local LLM
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.7
        )

        # Custom prompt template
        template = """Use the following context to answer the question
        If you don't know the answer based on the context, say so. 
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Take top 4 relevant chunks
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question):
        """Query the RAG system"""
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"],
        }
