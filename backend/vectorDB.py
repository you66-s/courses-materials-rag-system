import chromadb, os, dotenv
import numpy as np

class VectorDataBase:
    def __init__(self, collection_name: str):
        try:
            dotenv.load_dotenv()
            self.__client = chromadb.CloudClient(
                api_key=os.getenv("CHROMA_DB_API_KEY"),
                tenant=os.getenv("CHROMA_DB_TENANT"),
                database=os.getenv("CHROMA_DB_NAME")
            )
            self.__collection = self.__client.get_or_create_collection(name=collection_name)
            print(f"ChromaDB client initialized and collection '{collection_name}' ready.")
            print(f"collection documents count: {self.__collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB client: {e}")
            raise

    def add_document(self, id: str, document: str, metadata: dict, embedding: np.ndarray):
        try:
            self.__collection.add(
                ids = id,
                documents = document,
                metadatas = metadata,
                embeddings = embedding
            )
            print(f"Document with ID '{id}' added successfully.")
        except Exception as e:
            print(f"Error adding document with ID '{id}': {e}")
            raise
        
    def query_db(self, query_embedding: list[float], top_k: int) -> dict:
        return self.__collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )