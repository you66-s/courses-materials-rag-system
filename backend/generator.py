import pprint, sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'backend')))
import dotenv
from retriever import Retriever
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.messages import ( HumanMessage, SystemMessage)

class Generator:
    def __init__(self, query: str):
        print("start initialization from init...")
        self.__query = query
        dotenv.load_dotenv()
        self.__llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2-1.5B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            provider="auto",  # let Hugging Face choose the best provider for you
        )
    
    def generate(self, retreiver: Retriever, top_k: int = 5):
        print("start generating from generate...")
        results = retreiver.retrieve(self.__query, top_k=top_k)
        pages_metadata = [doc['metadata']['page'] for doc in results]
        pages_metadata = ", ".join([str(page) for page in pages_metadata])
        context = "\n\n".join([doc['document'] for doc in results])
        if context is None or context.strip() == "":
            return "No relevant information found to answer the query."
        prompt = f"""
        INSTRUCTIONS (hard constraints — MUST be obeyed):
1. Use ONLY the text found in the CONTEXT block below to produce an answer. Do NOT use any external knowledge, memory, or assumptions.
2. Every factual statement in the answer MUST come directly from the CONTEXT. If a detail does not exist in the CONTEXT, you must not infer or invent it.
3. If the CONTEXT does not contain enough information to fully answer the QUERY, reply exactly with:
   INSUFFICIENT_CONTEXT
   (no additional words, no punctuation).
4. When giving an answer, you MUST use metadata. Every factual sentence must include an inline citation in this format:
   [module: <module_name>, pages: <page_numbers>]
5. Metadata MUST come only from the metadata provided at the end of the CONTEXT block. Never create metadata.
6. Keep tone clear, concise, and pedagogical.
7. Never ask the user for clarification. If information is missing, return INSUFFICIENT_CONTEXT.

OUTPUT REQUIREMENTS (if context suffices):
    "short_answer": "2–3 sentence direct answer.",
    "detailed_explanation": "step-by-step breakdown, definitions, examples, and reasoning referencing only CONTEXT.",
    "mini_summary": ["3-5 bullet points"],
    "sources": ["pages: <n>"]
- Inline citations inside explanation must match the "sources" entries.
- All inline citations MUST correspond to entries in "sources".
- Do NOT include any extra fields.

CONTEXT (retrieved documents):
{context}

AVAILABLE METADATA:
pages: {pages_metadata}

QUERY:
{self.__query}

Begin.

        """
        chat_model = ChatHuggingFace(llm=self.__llm)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=self.__query),
        ]
        ai_msg = chat_model.invoke(messages)
        return ai_msg.content