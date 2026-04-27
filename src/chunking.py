import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunking:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_metadata(self, text: str, filename: str) -> dict:
        metadata = {
            "title": "",
            "so_ky_hieu": "",
            "ngay_ban_hanh": "",
            "ngay_co_hieu_luc": "",
            "loai_van_ban": "",
            "linh_vuc": "Giáo dục và Đào tạo",
            "co_quan_ban_hanh": "",
            "chuc_danh": "",
            "nguoi_ky": "",
            "year_of_issue": "",
            "source": filename
        }
        
        lines = text.split('\n')
        
        # Extract co_quan_ban_hanh
        for line in lines[:10]:
            line_clean = line.strip()
            if line_clean.isupper() and len(line_clean) > 10 and not line_clean.startswith("CỘNG HÒA") and "Độc lập" not in line_clean:
                metadata["co_quan_ban_hanh"] = line_clean
                break
                
        # Extract so_ky_hieu
        so_ky_hieu_match = re.search(r"Số:\s*([^\s\n,]+(?:/[^\s\n,]+)+)", text, re.IGNORECASE)
        if so_ky_hieu_match:
            metadata["so_ky_hieu"] = so_ky_hieu_match.group(1).strip()
        else:
            so_ky_hieu_match2 = re.search(r"Số\s+([^\s\n,]+(?:/[^\s\n,]+)+|[\d]+\s*/[^\s\n,]+)", text, re.IGNORECASE)
            if so_ky_hieu_match2:
                metadata["so_ky_hieu"] = so_ky_hieu_match2.group(1).strip()
                
        # Extract ngay_ban_hanh
        ngay_ban_hanh_match = re.search(r"(?:Hà Nội|.+?),\s*ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", text, re.IGNORECASE)
        if ngay_ban_hanh_match:
            metadata["ngay_ban_hanh"] = f"{ngay_ban_hanh_match.group(1).zfill(2)}/{ngay_ban_hanh_match.group(2).zfill(2)}/{ngay_ban_hanh_match.group(3)}"
            metadata["year_of_issue"] = ngay_ban_hanh_match.group(3)
            
        # Extract loai_van_ban
        loai_match = re.search(r"^(QUYẾT ĐỊNH|THÔNG TƯ|QUY ĐỊNH|CÔNG VĂN|CHỈ THỊ|HƯỚNG DẪN)[ \t]*$", text, re.MULTILINE)
        if loai_match:
            metadata["loai_van_ban"] = loai_match.group(1).strip()
            
        # Extract title
        title_match = re.search(r"(?:Về việc ban hành|Về việc)\s+([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[A-Z]|HIỆU TRƯỞNG|BỘ TRƯỞNG|THỦ TƯỚNG)", text)
        if title_match:
            metadata["title"] = title_match.group(1).replace('\n', ' ').strip()
        else:
            title_match2 = re.search(r"(?:QUYẾT ĐỊNH|THÔNG TƯ)\s*\n+(Quy định[^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[A-Z])", text)
            if title_match2:
                metadata["title"] = title_match2.group(1).replace('\n', ' ').strip()
                
        # Extract ngay_co_hieu_luc
        hieu_luc_match = re.search(r"có hiệu lực(?: thi hành)?(?: kể)? từ ngày\s*(\d{1,2}(?:/|-)\d{1,2}(?:/|-)\d{4}|\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})", text, re.IGNORECASE)
        if hieu_luc_match:
            metadata["ngay_co_hieu_luc"] = hieu_luc_match.group(1).strip()
        elif re.search(r"có hiệu lực(?: thi hành)?(?: kể)? từ ngày ký", text, re.IGNORECASE):
            metadata["ngay_co_hieu_luc"] = "ngày ký"

        # Extract chuc_danh and nguoi_ky
        last_lines = [l.strip() for l in lines[-40:] if l.strip()]
        if last_lines:
            metadata["nguoi_ky"] = last_lines[-1]
            if len(last_lines) > 1:
                for i in range(len(last_lines)-2, -1, -1):
                    # Check for keywords indicating title
                    if last_lines[i].isupper() or any(kw in last_lines[i].upper() for kw in ["KT.", "HIỆU TRƯỞNG", "BỘ TRƯỞNG", "THỦ TƯỚNG", "PHÓ"]):
                        chuc_danh_candidate = " ".join(last_lines[i:len(last_lines)-1])
                        if len(chuc_danh_candidate) < 100:
                            metadata["chuc_danh"] = chuc_danh_candidate
                        break
                        
        for k, v in metadata.items():
            if isinstance(v, str):
                metadata[k] = re.sub(r'\s+', ' ', v).strip()
                
        return metadata

    def process_directory(self, input_dir: str, output_file: str, pham_vi: str = "Toàn quốc"):
        all_chunks = []
        dir_path = Path(input_dir)
        
        if not dir_path.exists():
            print(f"Directory {input_dir} does not exist. Skipping.")
            return
            
        for file_path in dir_path.glob("*.txt"):
            print(f"Processing {file_path.name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            metadata = self.extract_metadata(content, file_path.name)
            metadata["pham_vi"] = pham_vi
            
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "metadata": metadata.copy(),
                    "chunk_id": f"{file_path.name}_chunk_{i}",
                    "content": chunk
                }
                all_chunks.append(chunk_data)
                
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)
            
        print(f"\nSuccessfully processed and generated {len(all_chunks)} chunks to {output_file}")

    def process_csv(self, input_csv: str, output_file: str, pham_vi: str = "Toàn quốc"):
        import pandas as pd
        all_chunks = []
        csv_path = Path(input_csv)
        
        if not csv_path.exists():
            print(f"CSV file {input_csv} does not exist. Skipping.")
            return
            
        print(f"Processing CSV {csv_path.name}...")
        df = pd.read_csv(csv_path)
        
        for index, row in df.iterrows():
            content = row.get('clean_content', '')
            if pd.isna(content) or not str(content).strip():
                continue
                
            content = str(content)
            
            metadata = {
                "title": str(row.get('title', '')) if not pd.isna(row.get('title')) else "",
                "so_ky_hieu": str(row.get('so_ky_hieu', '')) if not pd.isna(row.get('so_ky_hieu')) else "",
                "ngay_ban_hanh": str(row.get('ngay_ban_hanh', '')) if not pd.isna(row.get('ngay_ban_hanh')) else "",
                "ngay_co_hieu_luc": str(row.get('ngay_co_hieu_luc', '')) if not pd.isna(row.get('ngay_co_hieu_luc')) else "",
                "loai_van_ban": str(row.get('loai_van_ban', '')) if not pd.isna(row.get('loai_van_ban')) else "",
                "linh_vuc": str(row.get('linh_vuc', 'Giáo dục và Đào tạo')) if not pd.isna(row.get('linh_vuc')) else "Giáo dục và Đào tạo",
                "co_quan_ban_hanh": str(row.get('co_quan_ban_hanh', '')) if not pd.isna(row.get('co_quan_ban_hanh')) else "",
                "chuc_danh": str(row.get('chuc_danh', '')) if not pd.isna(row.get('chuc_danh')) else "",
                "nguoi_ky": str(row.get('nguoi_ky', '')) if not pd.isna(row.get('nguoi_ky')) else "",
                "year_of_issue": str(row.get('year_of_issue', '')) if not pd.isna(row.get('year_of_issue')) else "",
                "pham_vi": pham_vi,
                "source": f"{csv_path.name}_row_{index}"
            }
            
            for k, v in metadata.items():
                if isinstance(v, str):
                    metadata[k] = v.strip()
                    if metadata[k] == 'nan':
                        metadata[k] = ""
            
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "metadata": metadata.copy(),
                    "chunk_id": f"{csv_path.name}_row_{index}_chunk_{i}",
                    "content": chunk
                }
                all_chunks.append(chunk_data)
                
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)
            
        print(f"\nSuccessfully processed and generated {len(all_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    chunker = Chunking()
    
    base_dir_env = os.getenv("BASE_DIR", "/Users/nghia/Documents/khoa_luan_tot_nghiep")
    base_dir = Path(base_dir_env)
    
    neu_dir = base_dir / os.getenv("NEU_DOCS_DIR", "data/extracted/NEU_docs")
    neu_out = base_dir / os.getenv("NEU_JSON_OUT", "data/NEU.json")
    chunker.process_directory(neu_dir, neu_out, pham_vi="NEU/Kinh tế quốc dân")
    
    gddt_dir = base_dir / os.getenv("GDDT_DOCS_DIR", "data/extracted/GDDT_docs")
    gddt_out = base_dir / os.getenv("GDDT_JSON_OUT", "data/GDDT.json")
    chunker.process_directory(gddt_dir, gddt_out, pham_vi="Toàn quốc")
    
    legal_docs_csv = base_dir / os.getenv("LEGAL_DOCS_CSV", "data/raw/legal_docs.csv")
    legal_docs_out = base_dir / os.getenv("LEGAL_DOCS_JSON_OUT", "data/legal_docs.json")
    chunker.process_csv(legal_docs_csv, legal_docs_out, pham_vi="Toàn quốc")
