"""테스트용 유효한 샘플 PDF 생성 유틸리티"""

import os


def create_sample_pdf(file_path: str = "sample_knowledge.pdf") -> str:
    """PDF 1.4 표준에 맞추어 정확한 오프셋을 갖는 샘플 텍스트 PDF 파일 생성"""
    lines = [
        "Artificial Intelligence and RAG Guide",
        "Document Overview: RAG (Retrieval-Augmented Generation) Architecture",
        "",
        "1. Concept of RAG:",
        "Retrieval-Augmented Generation combines LLMs with external knowledge bases.",
        "Instead of relying solely on parametric training weights, RAG retrieves",
        "relevant context at runtime and supplies it to the model prompt.",
        "",
        "2. Key Benefits of RAG:",
        "- Hallucination Reduction: Grounding responses in verified documents.",
        "- Fresh Information: Keeping knowledge updated without expensive retraining.",
        "- Verifiable Sources: Enabling explicit citations and page references.",
        "- Security & Access Control: Keeping sensitive data in private databases.",
        "",
        "3. Document Chunking Strategies:",
        "Chunk size determines the size of text segments sent to the embedding model.",
        "A typical chunk size ranges between 400 and 1000 characters.",
        "Chunk overlap (usually 10 to 20 percent of chunk size) ensures that important",
        "information at boundaries is not lost between adjacent chunks.",
        "",
        "4. Vector Database and Cosine Similarity:",
        "Embeddings map text to high-dimensional vectors (e.g., 3072 dimensions in Gemini).",
        "Cosine similarity measures the angular distance between query and document vectors.",
        "A similarity score close to 1.0 represents high contextual relevance.",
    ]

    stream_content = "BT\n/F1 12 Tf\n40 750 Td\n18 TL\n"
    for line in lines:
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_content += f"({safe_line}) '\n"
    stream_content += "ET\n"

    stream_bytes = stream_content.encode("latin-1")
    stream_len = len(stream_bytes)

    objects = []
    # Obj 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Obj 2: Pages
    objects.append("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # Obj 3: Page
    objects.append(
        "3 0 obj\n"
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        "endobj\n"
    )
    # Obj 4: Contents Stream
    objects.append(
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n{stream_content}endstream\nendobj\n"
    )
    # Obj 5: Font
    objects.append(
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    header = "%PDF-1.4\n"
    offsets = [0]
    current_offset = len(header.encode("latin-1"))

    body_bytes = b""
    for obj_str in objects:
        offsets.append(current_offset)
        obj_b = obj_str.encode("latin-1")
        body_bytes += obj_b
        current_offset += len(obj_b)

    xref_offset = current_offset
    xref = f"xref\n0 {len(offsets)}\n"
    xref += "0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n"

    trailer = (
        f"trailer\n"
        f"<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n"
        f"{xref_offset}\n"
        f"%%EOF\n"
    )

    pdf_bytes = header.encode("latin-1") + body_bytes + xref.encode("latin-1") + trailer.encode("latin-1")

    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    return os.path.abspath(file_path)


if __name__ == "__main__":
    p = create_sample_pdf()
    print("Created:", p)

