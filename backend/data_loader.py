from langchain_community.document_loaders import PyMuPDFLoader

class DataLoader:
    def __init__(self, file_path: str, reading_mode: str = "page"):
        self.__file_path = file_path
        self.__reading_mode = reading_mode
        self.__loader = PyMuPDFLoader(file_path=self.__file_path, mode=self.__reading_mode)

    def load_data(self):
        documents = self.__loader.load()
        return documents
        