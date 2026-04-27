import os
import redis
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    _instance = None
    _redis_client = None
    _mongo_client = None
    _mongo_db = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def connect(self):
        # 1. Connect to Redis
        if self._redis_client is None:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            print(f"[DB] Connecting to Redis at {redis_host}:{redis_port}...")
            self.__class__._redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=0, 
                decode_responses=True
            )
            # Ping test
            self.__class__._redis_client.ping()
            print("[DB] Redis Connected!")

        # 2. Connect to MongoDB
        if self._mongo_client is None:
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            db_name = os.getenv("MONGO_DB_NAME", "legal_rag")
            print(f"[DB] Connecting to MongoDB at {mongo_uri} (DB: {db_name})...")
            self.__class__._mongo_client = MongoClient(mongo_uri)
            
            # Ping test
            self.__class__._mongo_client.admin.command('ping')
            self.__class__._mongo_db = self.__class__._mongo_client[db_name]
            print("[DB] MongoDB Connected!")

    def get_redis(self):
        if self._redis_client is None:
            self.connect()
        return self._redis_client

    def get_mongo_db(self):
        if self._mongo_db is None:
            self.connect()
        return self._mongo_db

if __name__ == "__main__":
    db = DatabaseManager()
    db.connect()
