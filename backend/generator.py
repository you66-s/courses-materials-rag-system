import dotenv
from backend.retriever import Retriever
from backend.embeddings_model import EmbeddingsModel
from langchain_google_genai import ChatGoogleGenerativeAI

class Generator:
    def __init__(self, query: str):
        self.__query = query
        dotenv.load_dotenv()
        self.__llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    
    def generate(self, retreiver: Retriever, top_k: int = 5):
        results = retreiver.retrieve(self.__query, top_k=top_k)
        pages_metadata = [doc['metadata']['page'] for doc in results]
        pages_metadata = ", ".join([str(page) for page in pages_metadata])
        module_metadata = [doc['metadata']['module'] for doc in results]
        module_metadata = "\n\n".join(set(module_metadata))
        context = "\n\n".join([doc['document'] for doc in results])
        if context is None or context.strip() == "":
            return "No relevant information found to answer the query."
        prompt = f"""
        You are an AI Professor Assistant specializing in simplifying complex academic concepts,
        explaining course materials, and providing accurate, structured, and pedagogical responses.
        
        ROLE: 
        - Act as a knowledgeable, patient, and clear university-level teaching assistant.
        - Provide explanations adapted to the student's level.
        - Use examples, analogies, and step-by-step reasoning.
        - Never fabricate information; if something is unclear, ask for clarification.
        
        CONTEXT:
        {context}
        The context contains retrieved documents, course materials, or any information 
        extracted. Use ONLY this context when answering.
        
        Each part of the context contains metadata in the form:
        module: {module_metadata}
        pages: {pages_metadata}
        Use this metadata to cite your sources in the final answer.
        
        USER QUERY:
        {self.__query}
        This is the exact question the student asked.
        
        OUTPUT FORMAT:
        Your response must follow this structure:
            1. **Short Answer (2–3 sentences)**
                - Provide a direct, clear, concise answer.
            2. **Detailed Explanation**
                - Break down the concept step by step.
                - Explain relevant definitions.
                - Add clarifications to avoid common misunderstandings.
            3. **If applicable: Mini Summary**
                - 3–5 bullet points summarizing the key concepts.
            4. **Sources**
                - List the metadata (module names and page numbers) of the documents you used to construct your answer.
                
        RULES & CONSTRAINTS:
        - Base your answer STRICTLY on the provided context.
        - Do NOT use or invent external facts.
        - If the context is insufficient, explicitly say:
        "The context does not provide enough information to answer this fully."
        - Keep formatting clean and structured (titles, bullet points, numbering).
        - Do NOT hallucinate sources; only cite metadata from the context provided.
        - Only include sources that directly contributed to your reasoning.
        """
        
        response = self.__llm.invoke(
            [prompt.format(context=context, query=self.__query)]
        )
        return response.content