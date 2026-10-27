"""SQLite 기반 경량 벡터 스토어 모듈

Python 표준 sqlite3와 numpy를 활용하여 별도 외부 DB 없이
벡터 임베딩 저장 및 코사인 유사도 검색을 지원합니다.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document


class SQLiteVectorStore:
    def __init__(self, db_path: str = "rag_sqlite.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """벡터 테이블 초기화"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def clear(self) -> None:
        """기존 저장된 청크 데이터 전체 삭제"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rag_chunks;")
            conn.commit()

    def count(self) -> int:
        """현재 저장된 청크 총 개수 반환"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rag_chunks;")
            return cursor.fetchone()[0]

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> int:
        """청크 문서와 임베딩 벡터 목록을 SQLite DB에 저장"""
        if len(documents) != len(embeddings):
            raise ValueError("문서 개수와 임베딩 개수가 일치하지 않습니다.")

        records = []
        for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
            source_file = str(doc.metadata.get("source", "unknown"))
            page_number = int(doc.metadata.get("page", 1))
            metadata_json = json.dumps(doc.metadata, ensure_ascii=False)
            # float32 바이너리(BLOB)로 압축 변환
            emb_blob = np.array(emb, dtype=np.float32).tobytes()

            records.append(
                (source_file, page_number, idx, doc.page_content, metadata_json, emb_blob)
            )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO rag_chunks (source_file, page_number, chunk_index, content, metadata_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                records,
            )
            conn.commit()

        return len(records)

    def similarity_search_by_vector(
        self, query_embedding: List[float], top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        """질문 임베딩 벡터와 코사인 유사도(Cosine Similarity)를 계산하여 상위 top_k 청크 반환"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, source_file, page_number, chunk_index, content, metadata_json, embedding
                FROM rag_chunks;
                """
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        # 쿼리 벡터 정규화
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        results = []
        for row in rows:
            doc_vec = np.frombuffer(row["embedding"], dtype=np.float32)
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0:
                d_norm = 1.0

            # 코사인 유사도 계산: (A · B) / (||A|| * ||B||)
            similarity = float(np.dot(q_vec, doc_vec) / (q_norm * d_norm))

            metadata = {}
            if row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except Exception:
                    metadata = {
                        "source": row["source_file"],
                        "page": row["page_number"],
                        "chunk_index": row["chunk_index"],
                    }

            metadata["db_id"] = row["id"]
            metadata["chunk_index"] = row["chunk_index"]

            doc = Document(page_content=row["content"], metadata=metadata)
            results.append((doc, similarity))

        # 유사도 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

