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
        """
        Reranks chunks using Maximal Marginal Relevance (MMR).
        lambda_mult = 1.0: Focus completely on relevance (similarity to query)
        lambda_mult = 0.0: Focus completely on diversity (difference from selected chunks)
        """
        if not chunks:
            return []
            
        # 1. Embed query and chunks
        query_vector = np.array(self.embeddings.embed_query(query))
        texts = [chunk['content'] for chunk in chunks]
        chunk_vectors = np.array(self.embeddings.embed_documents(texts))
        
        # 2. Compute similarity between query and chunks
        query_sims = [self._cosine_similarity(query_vector, v) for v in chunk_vectors]
        
        # 3. Compute similarity matrix between chunks
        num_chunks = len(chunks)
        chunk_sim_matrix = np.zeros((num_chunks, num_chunks))
        for i in range(num_chunks):
            for j in range(num_chunks):
                if i != j:
                    chunk_sim_matrix[i][j] = self._cosine_similarity(chunk_vectors[i], chunk_vectors[j])
                else:
                    chunk_sim_matrix[i][j] = 1.0
                    
        # 4. Perform MMR
        unselected = list(range(num_chunks))
        selected = []
        
        # Select the first chunk (highest query similarity)
        first_idx = int(np.argmax(query_sims))
        selected.append(first_idx)
        unselected.remove(first_idx)
        
        while unselected:
            mmr_scores = []
            for i in unselected:
                # Max similarity to already selected chunks
                max_sim_to_selected = max([chunk_sim_matrix[i][j] for j in selected])
                
                # MMR formula
                score = lambda_mult * query_sims[i] - (1 - lambda_mult) * max_sim_to_selected
                mmr_scores.append((score, i))
                
            # Find the unselected chunk with highest MMR score
            best_chunk_idx = max(mmr_scores, key=lambda x: x[0])[1]
            selected.append(best_chunk_idx)
            unselected.remove(best_chunk_idx)
            
        reranked_chunks = [chunks[i] for i in selected]
        return reranked_chunks

    def lost_in_the_middle_reorder(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reorders chunks so that the most relevant ones are at the edges:
        [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
        """
        reordered = []
        
        # First pass: append elements at even indices (0, 2, 4...) -> 1st, 3rd, 5th...
        for i in range(0, len(chunks), 2):
            reordered.append(chunks[i])
            
        # Second pass: append elements at odd indices (1, 3, 5...) in reverse order
        for i in range(len(chunks) - 1, 0, -1):
            if i % 2 != 0:
                reordered.append(chunks[i])
                
        return reordered

    def format_to_markdown(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Formats the chunks into a Markdown string suitable for LLM Context.
        """
        formatted = ""
        for i, chunk in enumerate(chunks, start=1):
            metadata = chunk.get('metadata', {})
            formatted += f"### Tài liệu {i}\n"
            
            # Print available metadata nicely
            if 'title' in metadata and metadata['title']:
                formatted += f"- **Tiêu đề:** {metadata['title']}\n"
            if 'source' in metadata and metadata['source']:
                formatted += f"- **Nguồn:** {metadata['source']}\n"
            if 'co_quan_ban_hanh' in metadata and metadata['co_quan_ban_hanh']:
                formatted += f"- **Cơ quan ban hành:** {metadata['co_quan_ban_hanh']}\n"
            if 'ngay_ban_hanh' in metadata and metadata['ngay_ban_hanh']:
                formatted += f"- **Ngày ban hành:** {metadata['ngay_ban_hanh']}\n"
            if 'ngay_co_hieu_luc' in metadata and metadata['ngay_co_hieu_luc']:
                formatted += f"- **Ngày hiệu lực:** {metadata['ngay_co_hieu_luc']}\n"
            if 'pham_vi' in metadata and metadata['pham_vi']:
                formatted += f"- **Phạm vi:** {metadata['pham_vi']}\n"
            
            formatted += f"\n**Nội dung:**\n{chunk['content']}\n"
            formatted += "\n---\n\n"
            
        return formatted

    def process(self, query: str, chunks: List[Dict[str, Any]], lambda_mult: float = 0.5) -> str:
        if not chunks:
            return ""
            
        print(f"Reranking {len(chunks)} chunks with MMR (lambda={lambda_mult})...")
        reranked = self.mmr_rerank(query, chunks, lambda_mult)
        
        print("Applying 'Lost in the middle' reordering...")
        reordered = self.lost_in_the_middle_reorder(reranked)
        
        print("Formatting to Markdown...")
        final_markdown = self.format_to_markdown(reordered)
        
        return final_markdown

if __name__ == "__main__":
    print("Initializing Retrieval module...")
    retriever = Retrieval()
    
    print("\n--- Retrieving Chunks ---")
    query = "Quy định về đạo văn và liêm chính học thuật"
    filters = {"pham_vi": "NEU/Kinh tế quốc dân"}
    
    # Retrieve top 5 to apply MMR
    results = retriever.search(query, filter_dict=filters, top_k=5, alpha=0.5)
    
    print(f"Found {len(results)} chunks. Before MMR order:")
    for i, res in enumerate(results):
        print(f"[{i+1}] Distance: {res['distance']:.4f} - {res['content'][:60].replace(chr(10), ' ')}...")
        
    print("\n--- Running Rerank & Format ---")
    formatter = RerankerAndFormatter()
    markdown_context = formatter.process(query, results, lambda_mult=0.5)
    
    print("\n--- FINAL CONTEXT FOR LLM ---\n")
    print(markdown_context)
