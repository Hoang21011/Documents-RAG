import numpy as np
from typing import List, Dict, Any
from src.retrieval import Retrieval

class RerankerAndFormatter:
    def __init__(self):
        # Re-use the Singleton instance of Retrieval to get the cached embeddings model
        retrieval = Retrieval()
        self.embeddings = retrieval.embeddings

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def mmr_rerank(self, query: str, chunks: List[Dict[str, Any]], lambda_mult: float = 0.5) -> List[Dict[str, Any]]:
        if not chunks: return []
        query_vector = np.array(self.embeddings.embed_query(query))
        texts = [chunk['content'] for chunk in chunks]
        chunk_vectors = np.array(self.embeddings.embed_documents(texts))
        
        query_sims = [self._cosine_similarity(query_vector, v) for v in chunk_vectors]
        num_chunks = len(chunks)
        chunk_sim_matrix = np.zeros((num_chunks, num_chunks))
        for i in range(num_chunks):
            for j in range(num_chunks):
                if i != j:
                    chunk_sim_matrix[i][j] = self._cosine_similarity(chunk_vectors[i], chunk_vectors[j])
                else:
                    chunk_sim_matrix[i][j] = 1.0
                    
        unselected = list(range(num_chunks))
        selected = []
        first_idx = int(np.argmax(query_sims))
        selected.append(first_idx)
        unselected.remove(first_idx)
        
        while unselected:
            mmr_scores = []
            for i in unselected:
                max_sim_to_selected = max([chunk_sim_matrix[i][j] for j in selected])
                score = lambda_mult * query_sims[i] - (1 - lambda_mult) * max_sim_to_selected
                mmr_scores.append((score, i))
            best_chunk_idx = max(mmr_scores, key=lambda x: x[0])[1]
            selected.append(best_chunk_idx)
            unselected.remove(best_chunk_idx)
            
        return [chunks[i] for i in selected]

    def lost_in_the_middle_reorder(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reordered = []
        for i in range(0, len(chunks), 2):
            reordered.append(chunks[i])
        for i in range(len(chunks) - 1, 0, -1):
            if i % 2 != 0:
                reordered.append(chunks[i])
        return reordered

    def format_to_markdown(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Rút gọn triệt để Context gửi cho LLM. 
        Đổi tên header thành [CHUNK-i] để tránh AI tự ý liệt kê lại 'Tài liệu X'.
        """
        formatted = ""
        for i, chunk in enumerate(chunks, start=1):
            formatted += f"### [ID: {i}]\n"
            formatted += f"{chunk['content']}\n"
            formatted += "\n---\n\n"
        return formatted

    def process(self, query: str, chunks: List[Dict[str, Any]], lambda_mult: float = 0.5) -> str:
        if not chunks: return ""
        reranked = self.mmr_rerank(query, chunks, lambda_mult)
        reordered = self.lost_in_the_middle_reorder(reranked)
        return self.format_to_markdown(reordered)
