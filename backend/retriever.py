from backend.embeddings_model import EmbeddingsModel
from backend.vectorDB import VectorDataBase

class Retriever:
    def __init__(self, collection_name: str = "course_materials"):
        self.__vector_db = VectorDataBase(collection_name=collection_name)
        self.__embedding_model = EmbeddingsModel()
        
    def retrieve(self, query: str, top_k: int, distance_threshold: float = 0.65) -> list[dict]:
        retrieved_docs = []
        try:
            query_embedding = self.__embedding_model.embed_texts([query])[0]
            results = self.__vector_db.query_db(query_embedding=query_embedding, top_k=top_k)
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadata = results['metadatas'][0]
            for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadata)):
                if dist > distance_threshold:
                    retrieved_docs.append({
                        "document": doc,
                        "metadata": meta
                    })
            return retrieved_docs
        except Exception as err:
            print("Error retrieving documents:", err)
            return None

        