import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus.model.sparse import BM25EmbeddingFunction
import underthesea

def my_analyzer(text: str):
    return underthesea.word_tokenize(text.lower())

class Retrieval:
    _instance = None
    _embeddings_model = None
    _bm25_ef = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Retrieval, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        base_dir_env = os.getenv("BASE_DIR", "/Users/nghia/Documents/khoa_luan_tot_nghiep")
        base_dir = Path(base_dir_env)
        
        # Load db uri
        db_uri_env = os.getenv("MILVUS_DB_URI", "./data/vector_database/milvus.db")
        if db_uri_env.startswith("./"):
            self.db_uri = str(base_dir / db_uri_env[2:])
        else:
            self.db_uri = db_uri_env
            
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME", "legal_docs_collection")
        self.client = MilvusClient(uri=self.db_uri)
        
        # Load Dense Model
        if Retrieval._embeddings_model is None:
            model_name = os.getenv("EMBEDDING_MODEL_NAME", "keepitreal/vietnamese-sbert")
            print(f"Loading Dense Embedding model ({model_name})...")
            Retrieval._embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        self.embeddings = Retrieval._embeddings_model
        
        # Load Sparse Model (BM25)
        if Retrieval._bm25_ef is None:
            bm25_path_env = os.getenv("BM25_MODEL_PATH", "data/vector_database/bm25_model.json")
            if bm25_path_env.startswith("./"):
                bm25_path = str(base_dir / bm25_path_env[2:])
            elif not bm25_path_env.startswith("/"):
                bm25_path = str(base_dir / bm25_path_env)
            else:
                bm25_path = bm25_path_env
                
            print(f"Loading BM25 model from {bm25_path}...")
            bm25_ef = BM25EmbeddingFunction(analyzer=my_analyzer)
            bm25_ef.load(bm25_path)
            Retrieval._bm25_ef = bm25_ef
        self.bm25_ef = Retrieval._bm25_ef
        
        self._initialized = True

    def _build_milvus_filter(self, filter_dict: Dict[str, str]) -> str:
        if not filter_dict:
            return ""
        conditions = []
        for k, v in filter_dict.items():
            if isinstance(v, str):
                conditions.append(f'{k} == "{v}"')
            else:
                conditions.append(f"{k} == {v}")
        return " and ".join(conditions)

    def search(self, query: str, filter_dict: Dict[str, str] = None, top_k: int = 5, alpha: float = 0.4) -> List[Dict[str, Any]]:
        milvus_filter_expr = self._build_milvus_filter(filter_dict)
        
        # 1. Dense query
        query_vector = self.embeddings.embed_query(query)
        req_dense = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP"},
            limit=top_k * 2,
            expr=milvus_filter_expr
        )
        
        # 2. Sparse query
        sparse_csr = self.bm25_ef.encode_queries([query]).tocsr()
        start, end = sparse_csr.indptr[0], sparse_csr.indptr[1]
        sparse_dict = {int(idx): float(val) for idx, val in zip(sparse_csr.indices[start:end], sparse_csr.data[start:end])}
        
        # Milvus requires sparse_dict to not be entirely empty if it is passed in the request
        if not sparse_dict:
            sparse_dict = {0: 0.0001} # Dummy value for zero overlap
            
        req_sparse = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k * 2,
            expr=milvus_filter_expr
        )
        
        # 3. Native Hybrid Search using WeightedRanker
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],
            ranker=WeightedRanker(alpha, 1.0 - alpha),
            limit=top_k,
            output_fields=["*"] # Load toàn bộ field kể cả dynamic metadata
        )
        
        # 4. Format results
        final_results = []
        if results and results[0]:
            for hit in results[0]:
                entity = hit['entity']
                metadata = {k: v for k, v in entity.items() if k not in ['text', 'id', 'vector', 'sparse_vector']}
                
                final_results.append({
                    "content": entity.get('text', ''),
                    "metadata": metadata,
                    "distance": hit['distance']
                })
                
        return final_results

if __name__ == "__main__":
    retriever1 = Retrieval()
    
    print("\n--- Testing Native Hybrid Retrieval ---")
    query = "Quy định về đạo văn và liêm chính học thuật"
    filters = {"pham_vi": "NEU/Kinh tế quốc dân"}
    
    print(f"Query: {query}")
    print(f"Filters: {filters}")
    print(f"Alpha: 0.5 (Cân bằng)")
    
    results = retriever1.search(query, filter_dict=filters, top_k=3, alpha=0.5)
    
    for i, res in enumerate(results, start=1):
        print(f"\n[{i}] Hybrid Score (Distance): {res['distance']:.4f}")
        print(f"Source: {res['metadata'].get('source', 'Unknown')}")
        print(f"Title: {res['metadata'].get('title', 'Unknown')}")
        print(f"Snippet: {res['content'][:200]}...")
