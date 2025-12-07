import os, uuid, logging
from tqdm import tqdm
from data_loader import DataLoader
from chunker import Chunker
from embeddings_model import EmbeddingsModel
from vectorDB import VectorDataBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

chunker = Chunker()
embedding_model = EmbeddingsModel()
vector_db = VectorDataBase(collection_name="course_materials")

BASE = "data"
raw_data = os.listdir(BASE)

logger.info("Starting indexing process...")
logger.info(f"Detected {len(raw_data)} semesters.")

for semester in tqdm(raw_data, desc="Semesters", colour="green"):
    modules = os.listdir(os.path.join(BASE, semester))

    for module in tqdm(modules, desc=f"Modules ({semester})", leave=False, colour="yellow"):
        materials = os.listdir(os.path.join(BASE, semester, module))

        for material in tqdm(materials, desc=f"Materials ({module})", leave=False, colour="cyan"):
            path = os.path.join(BASE, semester, module, material)
            try:
                documents = DataLoader(file_path=path).load_data()
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue
            logger.info(f"Loaded {len(documents)} document(s) from {path}")
            additional_metadata = {
                "semester": semester,
                "module": module,
            }
            chunks = chunker.chunk_documents(document=documents)
            for chunk in chunks:
              metadata = {**additional_metadata, **chunk.metadata}
              try:
                     embedding = embedding_model.embed_texts([chunk.page_content])[0]
                     vector_db.add_document(
                            id=str(uuid.uuid4()),
                            document=chunk.page_content,
                            metadata=metadata,
                            embedding=embedding
                        )
              except Exception as e:
                     logger.error(f"Error embedding or inserting chunk: {e}")
logger.info("Indexing completed successfully.")
