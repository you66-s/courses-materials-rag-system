import os, uuid
import chromadb

import numpy as np
from chromadb.config import Settings


class VectorDataBase:
    def __init__(self, collection_name: str, persist_directory: str = "chroma_db"):
        try:
            # Ensure persistence directory exists
            os.makedirs(persist_directory, exist_ok=True)
            # Initialize Persistent Client
            self.__client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            # Create or load collection
            self.__collection = self.__client.get_or_create_collection(
                name=collection_name
            )

            print(f"ChromaDB PersistentClient initialized")
            print(f"Persist directory: {persist_directory}")
            print(f"Collection: {collection_name}")
            print(f"Documents count: {self.__collection.count()}")

        except Exception as e:
            print(f"Error initializing PersistentClient: {e}")
            raise

    # ---------------------------------------------------------
    # ADD DOCUMENT
    def add_document( self, id: str, document: str, metadata: dict, embedding: np.ndarray ):
        try:
            self.__collection.add(
                ids=id,
                documents=document,
                metadatas=metadata,
                embeddings=embedding.tolist()
            )
            print(f"Document '{id}' added & persisted.")

        except Exception as e:
            print(f"Error adding document '{id}': {e}")
            raise

    # ---------------------------------------------------------
    # QUERY DATABASE
    def query_db( self, query_embedding: list[float], top_k: int ) -> dict:
        try:
            return self.__collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        except Exception as e:
            print(f"Error querying database: {e}")
            raise
