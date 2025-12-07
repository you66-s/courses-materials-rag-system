from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingsModel:
    def __init__(self):
        self.__model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.__model.encode(texts)
        return embeddings

    def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        similarity = self.__model.similarity(vec1, vec2)
        return similarity