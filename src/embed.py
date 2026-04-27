import json
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from pymilvus.model.sparse import BM25EmbeddingFunction
import underthesea

def my_analyzer(text: str):
    # Dùng underthesea để tách từ tiếng Việt chuẩn xác
    return underthesea.word_tokenize(text.lower())

class Embed:
    def __init__(
        self, 
        model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "keepitreal/vietnamese-sbert"), 
        db_uri: str = None, 
        collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "legal_docs_collection"),
        bm25_path: str = None
    ):
        base_dir_env = os.getenv("BASE_DIR", "/Users/nghia/Documents/khoa_luan_tot_nghiep")
        
        if db_uri is None:
            db_uri_env = os.getenv("MILVUS_DB_URI", "./data/vector_database/milvus.db")
            if db_uri_env.startswith("./"):
                db_uri = str(Path(base_dir_env) / db_uri_env[2:])
            else:
                db_uri = db_uri_env
                
        if bm25_path is None:
            bm25_path_env = os.getenv("BM25_MODEL_PATH", "data/vector_database/bm25_model.json")
            if bm25_path_env.startswith("./"):
                bm25_path = str(Path(base_dir_env) / bm25_path_env[2:])
            elif not bm25_path_env.startswith("/"):
                bm25_path = str(Path(base_dir_env) / bm25_path_env)
            else:
                bm25_path = bm25_path_env
                
        print(f"Loading HuggingFace Embedding model ({model_name})...")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.db_uri = db_uri
        self.bm25_path = bm25_path
        self.collection_name = collection_name

    def load_chunks_from_json(self, file_path: Path) -> list[dict]:
        if not file_path.exists():
            print(f"File {file_path} does not exist.")
            return []
            
        print(f"Loading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            
        print(f"Loaded {len(chunks_data)} chunks from {file_path.name}")
        return chunks_data

    def ingest_documents(self, chunks_data: list[dict]):
        if not chunks_data:
            print("No documents to ingest.")
            return
            
        print(f"Connecting to Milvus at {self.db_uri} and indexing {len(chunks_data)} documents...")
        
        Path(self.db_uri).parent.mkdir(parents=True, exist_ok=True)
        client = MilvusClient(uri=self.db_uri)
        
        if client.has_collection(collection_name=self.collection_name):
            print(f"Dropping existing collection '{self.collection_name}'...")
            client.drop_collection(collection_name=self.collection_name)
            
        # 1. Trích xuất văn bản
        texts = [chunk['content'] for chunk in chunks_data]
        
        # 2. Huấn luyện (Fit) BM25 Model
        print("Training BM25 model for Sparse Vectors...")
        bm25_ef = BM25EmbeddingFunction(analyzer=my_analyzer)
        bm25_ef.fit(texts)
        
        print(f"Saving BM25 model to {self.bm25_path}...")
        bm25_ef.save(self.bm25_path)
        
        # 3. Tạo Dense Vector và Sparse Vector
        print(f"Encoding {len(texts)} texts for Dense Vectors...")
        dense_vectors = self.embeddings.embed_documents(texts)
        
        print(f"Encoding {len(texts)} texts for Sparse Vectors...")
        sparse_vectors = bm25_ef.encode_documents(texts)
        
        # 4. Chuẩn bị Schema cho Milvus
        print(f"Creating collection '{self.collection_name}' with Native Sparse Vector support...")
        schema = MilvusClient.create_schema(
            auto_id=False, 
            enable_dynamic_field=True # Cho phép lưu Metadata tùy ý dưới dạng scalar
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=len(dense_vectors[0]))
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        
        client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )
        
        # 5. Tạo Index
        print("Creating Indexes for Dense and Sparse fields...")
        index_params = client.prepare_index_params()
        # Index cho Dense
        index_params.add_index(
            field_name="vector", 
            index_name="dense_index", 
            index_type="FLAT", 
            metric_type="IP"
        )
        # Index cho Sparse
        index_params.add_index(
            field_name="sparse_vector", 
            index_name="sparse_index", 
            index_type="SPARSE_INVERTED_INDEX", 
            metric_type="IP"
        )
        client.create_index(collection_name=self.collection_name, index_params=index_params)
        
        # 6. Load Collection
        client.load_collection(self.collection_name)
        
        # 7. Ingest dữ liệu
        print("Preparing data for Milvus...")
        data = []
        sparse_csr = sparse_vectors.tocsr()
        
        for i, (chunk, d_vec) in enumerate(zip(chunks_data, dense_vectors)):
            start = sparse_csr.indptr[i]
            end = sparse_csr.indptr[i+1]
            s_vec_dict = {int(idx): float(val) for idx, val in zip(sparse_csr.indices[start:end], sparse_csr.data[start:end])}
            
            row = {
                "id": str(uuid.uuid4()),
                "text": chunk['content'],
                "vector": d_vec,
                "sparse_vector": s_vec_dict
            }
            for k, v in chunk['metadata'].items():
                row[k] = v
            data.append(row)
            
        print(f"Inserting {len(data)} rows into collection '{self.collection_name}'...")
        res = client.insert(collection_name=self.collection_name, data=data)
        print(f"Successfully inserted {res['insert_count']} documents.")

if __name__ == "__main__":
    base_dir_env = os.getenv("BASE_DIR", "/Users/nghia/Documents/khoa_luan_tot_nghiep")
    base_dir = Path(base_dir_env)
    
    # Initialize the class
    embedder = Embed()
    
    # 1. Load data
    json_files = [
        base_dir / os.getenv("NEU_JSON_OUT", "data/NEU.json"),
        base_dir / os.getenv("GDDT_JSON_OUT", "data/GDDT.json"),
        base_dir / os.getenv("LEGAL_DOCS_JSON_OUT", "data/legal_docs.json")
    ]
    
    all_chunks = []
    for f in json_files:
        all_chunks.extend(embedder.load_chunks_from_json(f))
        
    # 2. Ingest
    if all_chunks:
        embedder.ingest_documents(all_chunks)
