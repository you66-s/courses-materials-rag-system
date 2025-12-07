import chromadb, os, dotenv

class VectorDataBase:
    def __init__(self):
        dotenv.load_dotenv()
        self.__client = chromadb.Client(
            api_key=os.getenv("CHROMA_DB_API_KEY"),
            tenant=os.getenv("CHROMA_DB_TENANT"),
            database=os.getenv("CHROMA_DB_NAME")
        )