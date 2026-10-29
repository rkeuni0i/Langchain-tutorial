"""SQLite 기반 경량 벡터 스토어 모듈

Python 표준 sqlite3와 numpy를 활용하여 별도 외부 DB 없이
벡터 임베딩 저장 및 코사인 유사도 검색을 지원합니다.
"""

import hashlib
import json
import math
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from langchain_core.documents import Document


def _tokenize(text: str) -> List[str]:
    """한글/영문 형태소 및 2-gram 토큰화 (어미/조사 분리 및 복합명사 매칭)"""
    words = re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())
    tokens = []
    for w in words:
        tokens.append(w)
        if len(w) >= 2:
            for i in range(len(w) - 1):
                tokens.append(w[i : i + 2])
    return tokens


class SQLiteVectorStore:
    def __init__(self, db_path: str = "rag_sqlite.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """벡터 테이블 및 인덱스 관리 테이블 초기화"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_hash TEXT,
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # 기존 테이블에 content_hash 컬럼이 없는 경우 자동 마이그레이션
            cursor.execute("PRAGMA table_info(rag_chunks);")
            columns = [r["name"] for r in cursor.fetchall()]
            if "content_hash" not in columns:
                cursor.execute("ALTER TABLE rag_chunks ADD COLUMN content_hash TEXT;")

            # 파일 단위 관리 테이블 (SHA-256 해시 추적)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size_kb REAL,
                    total_pages INTEGER,
                    total_chunks INTEGER,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 페이지 단위 관리 테이블 (PDF 페이지 원본 해시 → 이미 적재된 쪽은 OCR/임베딩/저장 건너뛰기)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_pages (
                    page_hash TEXT PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # 임베딩 캐시 테이블 (동일 텍스트 청크 재임베딩 방지 및 API 비용/시간 절약)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    text_content TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, text_hash)
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_emb_cache_lookup ON embedding_cache(model_name, text_hash);"
            )
            conn.commit()

    def get_cached_embeddings(
        self, model_name: str, texts: List[str]
    ) -> Tuple[Dict[int, List[float]], List[Tuple[int, str]]]:
        """
        texts 목록 중 캐시에 존재하는 임베딩과 캐시 미스(신규 API 호출 필요) 항목을 분리 반환합니다.

        반환값:
            cached_hits: {idx: embedding_vector, ...}
            missing_items: [(idx, text), ...]
        """
        if not texts:
            return {}, []

        def _clean_hash(t: str) -> str:
            cleaned = re.sub(r"\s+", " ", t.strip())
            return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        hashes = [_clean_hash(t) for t in texts]
        hash_to_indices: Dict[str, List[int]] = {}
        for idx, h in enumerate(hashes):
            hash_to_indices.setdefault(h, []).append(idx)

        cached_hits: Dict[int, List[float]] = {}
        missing_indices = set(range(len(texts)))

        with self._get_connection() as conn:
            cur = conn.cursor()
            unique_hashes = list(hash_to_indices.keys())
            batch_size = 500
            for i in range(0, len(unique_hashes), batch_size):
                batch_hashes = unique_hashes[i : i + batch_size]
                placeholders = ",".join("?" for _ in batch_hashes)
                cur.execute(
                    f"SELECT text_hash, embedding FROM embedding_cache WHERE model_name = ? AND text_hash IN ({placeholders})",
                    [model_name] + batch_hashes,
                )
                for row in cur.fetchall():
                    thash = row["text_hash"]
                    blob = row["embedding"]
                    vec = np.frombuffer(blob, dtype=np.float32).tolist()
                    for orig_idx in hash_to_indices.get(thash, []):
                        cached_hits[orig_idx] = vec
                        missing_indices.discard(orig_idx)

        missing_items = [(idx, texts[idx]) for idx in sorted(missing_indices)]
        return cached_hits, missing_items

    def save_cached_embeddings(
        self, model_name: str, items: List[Tuple[str, List[float]]]
    ) -> int:
        """새로 계산된 임베딩들을 영구 캐시 테이블에 저장합니다."""
        if not items:
            return 0

        def _clean_hash(t: str) -> str:
            cleaned = re.sub(r"\s+", " ", t.strip())
            return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

        records = []
        for text, emb in items:
            thash = _clean_hash(text)
            blob = np.array(emb, dtype=np.float32).tobytes()
            records.append((model_name, thash, text[:500], len(emb), blob))

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO embedding_cache (model_name, text_hash, text_content, dimension, embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_name, text_hash) DO NOTHING;
                """,
                records,
            )
            conn.commit()
            return cur.rowcount

    def get_cache_stats(self) -> Dict[str, Any]:
        """임베딩 캐시 통계 조회"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM embedding_cache;")
            total_cached = cur.fetchone()[0]
            cur.execute("SELECT model_name, COUNT(*) FROM embedding_cache GROUP BY model_name;")
            models = {r[0]: r[1] for r in cur.fetchall()}
            return {"total_cached_embeddings": total_cached, "models": models}

    def clear_embedding_cache(self) -> int:
        """임베딩 캐시 전체 비우기"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM embedding_cache;")
            count = cur.rowcount
            conn.commit()
            return count

    def clear(self) -> None:
        """기존 저장된 청크 데이터 및 파일 기록 전체 삭제"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rag_chunks;")
            cursor.execute("DELETE FROM indexed_files;")
            cursor.execute("DELETE FROM indexed_pages;")
            conn.commit()

    def delete_by_source(self, source_file: str) -> int:
        """특정 파일의 기존 청크 데이터만 선별 삭제 (부분 갱신용)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rag_chunks WHERE source_file = ?;", (source_file,))
            deleted = cursor.rowcount
            cursor.execute("DELETE FROM indexed_files WHERE file_name = ?;", (source_file,))
            cursor.execute("DELETE FROM indexed_pages WHERE source_file = ?;", (source_file,))
            conn.commit()
            return deleted

    def get_known_page_hashes(self, page_hashes: List[str]) -> set:
        """주어진 페이지 해시 중 이미 적재된 것들의 집합 반환"""
        known: set = set()
        if not page_hashes:
            return known
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for i in range(0, len(page_hashes), 500):
                batch = page_hashes[i : i + 500]
                placeholders = ",".join("?" for _ in batch)
                cursor.execute(
                    f"SELECT page_hash FROM indexed_pages WHERE page_hash IN ({placeholders});", batch
                )
                known.update(r["page_hash"] for r in cursor.fetchall())
        return known

    def record_pages(self, source_file: str, pages: List[Tuple[int, str]]) -> None:
        """적재 완료된 페이지 기록. pages = [(page_number, page_hash), ...]"""
        if not pages:
            return
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO indexed_pages (page_hash, source_file, page_number) VALUES (?, ?, ?);",
                [(h, source_file, n) for n, h in pages],
            )
            conn.commit()

    def has_recorded_pages(self, source_file: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT 1 FROM indexed_pages WHERE source_file = ? LIMIT 1;", (source_file,))
            return cur.fetchone() is not None

    def record_file(
        self,
        file_name: str,
        file_hash: str,
        file_size_kb: float,
        total_pages: int,
        total_chunks: int,
    ) -> None:
        """인덱싱된 파일 메타데이터 기록 (중복 시 갱신)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO indexed_files (file_name, file_hash, file_size_kb, total_pages, total_chunks)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(file_name) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    file_size_kb = excluded.file_size_kb,
                    total_pages = excluded.total_pages,
                    total_chunks = excluded.total_chunks,
                    indexed_at = CURRENT_TIMESTAMP;
                """,
                (file_name, file_hash, file_size_kb, total_pages, total_chunks),
            )
            conn.commit()

    def get_file_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """동일한 해시(내용 100% 일치)를 가진 파일 정보 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_name, file_size_kb, total_pages, total_chunks, indexed_at FROM indexed_files WHERE file_hash = ? LIMIT 1;",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def check_subset_document(
        self,
        new_text: str,
        threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """
        새로 입력된 문서 텍스트가 기존 DB에 저장된 특정 문서의 부분집합(Subset / 하위 발췌본)인지 검사합니다.
        
        검사 원리:
        1. 새 문서의 15자 이상 구문/문장들을 추출
        2. 기존 DB의 파일별 원문 본문과 구문 일치도 및 어휘 포함률(Inclusion Rate) 계산
        3. 종합 포함율이 threshold (기본 80%) 이상이면 하위 문서(Subset)로 판정
        """
        if not new_text or self.count() == 0:
            return {"is_subset": False, "overlap_ratio": 0.0, "matched_file": None, "message": ""}

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT source_file FROM rag_chunks;")
            source_files = [r["source_file"] for r in cur.fetchall()]

            if not source_files:
                return {"is_subset": False, "overlap_ratio": 0.0, "matched_file": None, "message": ""}

            # 새 텍스트를 문장/단락 단위(약 60~100자) 조각으로 분할
            sentences = [s.strip() for s in re.split(r"[\n.?!]+", new_text) if len(s.strip()) >= 15]
            if not sentences:
                sentences = [new_text.strip()]

            new_words = set(re.findall(r"[가-힣a-zA-Z0-9]+", new_text.lower()))
            if not new_words:
                return {"is_subset": False, "overlap_ratio": 0.0, "matched_file": None, "message": ""}

            best_match_file = None
            highest_ratio = 0.0

            for sfile in source_files:
                cur.execute("SELECT content FROM rag_chunks WHERE source_file = ?;", (sfile,))
                file_contents = [r["content"] for r in cur.fetchall()]
                full_file_text = " ".join(file_contents)
                file_words = set(re.findall(r"[가-힣a-zA-Z0-9]+", full_file_text.lower()))

                # 1. 단어 포함율 (Token Inclusion): 새 문서의 단어가 기존 파일에 얼마나 포함되어 있는가?
                common_words = new_words.intersection(file_words)
                token_inclusion = len(common_words) / len(new_words)

                # 2. 문장/구문 포함율 (Sentence Inclusion): 새 문서의 문장이 기존 파일 본문에 얼마나 포함되어 있는가?
                clean_full_file = re.sub(r"\s+", "", full_file_text)
                matched_sentences = 0
                for sent in sentences:
                    sent_clean = re.sub(r"\s+", "", sent)
                    if sent_clean and sent_clean in clean_full_file:
                        matched_sentences += 1
                    else:
                        sent_tokens = set(re.findall(r"[가-힣a-zA-Z0-9]+", sent.lower()))
                        if sent_tokens and (len(sent_tokens.intersection(file_words)) / len(sent_tokens)) >= 0.85:
                            matched_sentences += 1

                sentence_inclusion = matched_sentences / len(sentences)

                # 종합 포함율 가중치: 구문/문장 일치도 70% + 단어 포함도 30%
                composite_ratio = (sentence_inclusion * 0.7) + (token_inclusion * 0.3)

                if composite_ratio > highest_ratio:
                    highest_ratio = composite_ratio
                    best_match_file = sfile

            is_subset = highest_ratio >= threshold
            msg = ""
            if is_subset:
                msg = (
                    f"업로드된 문서는 이미 기존 등록된 '{best_match_file}'의 내용과 "
                    f"{highest_ratio * 100:.1f}% 일치하는 부분 발췌본(하위 문서)입니다. "
                    f"불필요한 중복 저장을 방지하기 위해 신규 인덱싱을 건너뜁니다."
                )

            return {
                "is_subset": is_subset,
                "overlap_ratio": round(highest_ratio, 3),
                "matched_file": best_match_file,
                "message": msg,
            }

    def count(self) -> int:
        """현재 저장된 청크 총 개수 반환"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rag_chunks;")
            return cursor.fetchone()[0]

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> int:
        """청크 문서와 임베딩 벡터 목록을 SQLite DB에 저장 (이미 같은 내용의 청크가 있으면 건너뜀)

        - content_hash(공백 정규화 MD5)가 이미 DB에 있는 청크는 저장하지 않아 중복 적재를 막는다.
        - chunk_index는 기존 최대값 뒤에 이어 붙여 증분 추가 시에도 겹치지 않게 한다.
        반환값: 실제로 저장된 레코드 수
        """
        if len(documents) != len(embeddings):
            raise ValueError("문서 개수와 임베딩 개수가 일치하지 않습니다.")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM rag_chunks WHERE content_hash IS NOT NULL;")
            seen = {r["content_hash"] for r in cursor.fetchall()}
            cursor.execute("SELECT COALESCE(MAX(chunk_index), -1) AS m FROM rag_chunks;")
            next_idx = cursor.fetchone()["m"] + 1

        records = []
        for doc, emb in zip(documents, embeddings):
            source_file = str(doc.metadata.get("source", "unknown"))
            page_number = int(doc.metadata.get("page", 1))
            content_clean = re.sub(r"\s+", " ", doc.page_content.strip())
            content_hash = hashlib.md5(content_clean.encode("utf-8")).hexdigest()
            if content_hash in seen:
                continue
            seen.add(content_hash)
            metadata_json = json.dumps(doc.metadata, ensure_ascii=False)
            # float32 바이너리(BLOB)로 압축 변환
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            records.append(
                (source_file, page_number, next_idx, content_hash, doc.page_content, metadata_json, emb_blob)
            )
            next_idx += 1

        if records:
            with self._get_connection() as conn:
                conn.executemany(
                    """
                    INSERT INTO rag_chunks (source_file, page_number, chunk_index, content_hash, content, metadata_json, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    records,
                )
                conn.commit()

        return len(records)

    def similarity_search_by_vector(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        query_text: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """질문 임베딩 벡터 및 키워드(BM25)를 기반으로 상위 top_k 청크 반환 (하이브리드 RRF 지원)"""
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

        # 1. 쿼리 벡터 정규화 및 코사인 유사도(Dense Similarity) 계산
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        dense_scores: Dict[int, float] = {}
        for row in rows:
            doc_vec = np.frombuffer(row["embedding"], dtype=np.float32)
            d_norm = np.linalg.norm(doc_vec)
            if d_norm == 0:
                d_norm = 1.0
            similarity = float(np.dot(q_vec, doc_vec) / (q_norm * d_norm))
            dense_scores[row["id"]] = similarity

        # Dense 순위 매기기 (1부터 N까지)
        sorted_dense = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        dense_ranks = {row_id: rank for rank, (row_id, _) in enumerate(sorted_dense, 1)}

        # 2. query_text가 있는 경우 Sparse(BM25) 키워드 검색 및 RRF(Reciprocal Rank Fusion) 적용
        use_hybrid = bool(query_text and query_text.strip())
        bm25_ranks: Dict[int, int] = {}
        bm25_scores: Dict[int, float] = {}

        if use_hybrid:
            q_tokens = set(_tokenize(query_text))
            N = len(rows)

            # Document Frequency (DF) 계산
            df: Dict[str, int] = {}
            doc_tokens_map: Dict[int, set] = {}
            for row in rows:
                t_set = set(_tokenize(row["content"]))
                doc_tokens_map[row["id"]] = t_set
                for t in t_set:
                    df[t] = df.get(t, 0) + 1

            # BM25 스타일 점수 계산
            for row in rows:
                row_id = row["id"]
                score = 0.0
                doc_tokens = doc_tokens_map[row_id]
                for t in q_tokens:
                    if t in doc_tokens:
                        # IDF 가중치 계산
                        idf = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
                        score += idf
                bm25_scores[row_id] = score

            sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            bm25_ranks = {row_id: rank for rank, (row_id, _) in enumerate(sorted_bm25, 1)}

        # 3. 최종 점수 및 순위 조합
        results = []
        max_rrf = 2.0 / 61.0  # RRF 최대 가능 점수 (Dense 1위 + BM25 1위)
        min_rrf = 2.0 / (60.0 + len(rows))

        for row in rows:
            row_id = row["id"]
            d_sim = dense_scores[row_id]
            d_rank = dense_ranks[row_id]

            if use_hybrid:
                b_rank = bm25_ranks.get(row_id, len(rows))
                b_score = bm25_scores.get(row_id, 0.0)
                # Reciprocal Rank Fusion (k=60)
                rrf = (1.0 / (60.0 + d_rank)) + (1.0 / (60.0 + b_rank))
                # 0.0 ~ 1.0 스케일로 정규화
                norm_score = max(0.0, min(1.0, (rrf - min_rrf) / (max_rrf - min_rrf)))
                final_score = round(norm_score, 4)
            else:
                final_score = d_sim

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
            metadata["dense_similarity"] = round(d_sim, 4)
            metadata["dense_rank"] = d_rank
            if use_hybrid:
                metadata["bm25_score"] = round(bm25_scores.get(row_id, 0.0), 2)
                metadata["bm25_rank"] = bm25_ranks.get(row_id, len(rows))
                metadata["search_type"] = "hybrid_rrf"

            doc = Document(page_content=row["content"], metadata=metadata)
            results.append((doc, final_score))

        # 유사도/융합점수 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

