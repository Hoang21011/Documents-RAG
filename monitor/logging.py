import logging
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    def __init__(self, service_name: str, env: str):
        super().__init__()
        self.service_name = service_name
        self.env = env

    def format(self, record: logging.LogRecord) -> str:
        # Chuẩn bị dữ liệu theo logging_schema.json
        log_record: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "event": record.getMessage(),
            "env": self.env
        }
        
        # Thêm các trường optional nếu chúng được truyền qua "extra"
        if hasattr(record, 'model'):
            log_record['model'] = getattr(record, 'model')
            
        if hasattr(record, 'latency_ms'):
            log_record['latency_ms'] = getattr(record, 'latency_ms')
            
        if hasattr(record, 'token_in_out'):
            log_record['token_in_out'] = getattr(record, 'token_in_out')
            
        # Ghi log lỗi nếu có exception
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)

def get_logger(name: str, service_name: str = "backend", env: str = "dev") -> logging.Logger:
    """
    Khởi tạo và trả về một logger in log dưới định dạng JSON theo schema đã quy định.
    """
    logger = logging.getLogger(name)
    
    # Tránh gắn handler nhiều lần nếu logger đã được cấu hình
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        formatter = JSONFormatter(service_name=service_name, env=env)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Thêm file handler
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
