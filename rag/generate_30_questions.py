import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

pdf_path = Path("rag/uploads/han-gang1-100.pdf")
output_json_path = Path("rag/han_gang_30.json")

print(f"Uploading {pdf_path} for question generation...")
uploaded_file = client.files.upload(file=str(pdf_path))
print(f"File uploaded: {uploaded_file.name}, State: {uploaded_file.state}")

while uploaded_file.state.name == "PROCESSING":
    print("Waiting for file processing...")
    time.sleep(4)
    uploaded_file = client.files.get(name=uploaded_file.name)

print(f"File ready: {uploaded_file.state.name}")

system_prompt = """당신은 RAG(검색 증강 생성) 시스템의 검색 정확도와 답변 품질을 검증하기 위한 'RAG 평가 데이터셋 설계 전문가'입니다.
제공된 100페이지 PDF 문서(한강의 다리 이야기: 프롤로그, 1장 양화대교, 2장 원효대교, 3장 한강철교)의 본문 내용을 면밀히 분석하여,
RAG 시스템에 질의해볼 만한 **가장 가치 있고 정밀한 30개의 평가 질문 데이터셋**을 작성해 주세요.

[질문 선정 및 배분 기준]
1. 문서 전반의 균형 잡힌 배분 (총 30개):
   - 프롤로그 (pp. 1 ~ 12): 한강의 지리, 발원지 물길, 삼국시대 명칭, 8개 다리 개요 (6~7문항)
   - 제1장 양화대교 (pp. 13 ~ 40): 제2한강교와 군사 목적, 1965년 완공, 절두산 순교성지/병인박해, 선유도, 강 플레이트 거더교/국부좌굴/스티프너, UHPC 선유교 (8문항)
   - 제2장 원효대교 (pp. 41 ~ 79): 효창공원/원효로 역사, 괴물 영화, 백범 김구/안두희/박기서, 동아건설 민자시공/통행료, 프리스트레스트 콘크리트(PSC) 원리, 디비닥 공법, 콘크리트 발전사(에디스톤 등대, 조제프 모니에, 프레시네) (8~9문항)
   - 제3장 한강철교 (pp. 80 ~ 100): 제1철교(1900년 완공)와 모스, 표준궤, 1925년 을축년 대홍수, 6·25 한강철교 폭파 비극, 트러스 구조 원리, 공진(Resonance) 현상, 베세머 전로/평로와 에즈교/포스교, 전함 드레드노트와 러일전쟁 (6~7문항)

2. 질문 유형의 다양성:
   - 사실 확인형(Factoid): 연도, 인물, 지명, 명칭 유래 (예: "양화대교는 몇년도에 완공되었는가?")
   - 공학/기술 원리형(Engineering): PSC 원리, 트러스 구조, 국부좌굴과 보강재, 공진 현상
   - 인과 관계 및 배경 설명형(Context/Reason): 군사 목적 건설 이유, 민자 통행료 징수 논란, 병인박해 발발 배경
   - 비교 및 다중 팩트 종합형(Comparative): 센강/템스강과 한강 비교, 고구려/백제/신라의 한강 명칭 비교

3. JSON 스키마 준수 (순수 JSON 배열만 출력):
각 질문 객체는 다음 필드를 반드시 포함해야 합니다:
{
  "id": 1,
  "category": "역사/유래" | "공학/구조" | "인물/사회" | "지리/하천" | "사건/사고",
  "chapter": "프롤로그" | "1장 양화대교" | "2장 원효대교" | "3장 한강철교",
  "page_range": "해당 내용이 등장하는 페이지 (예: pp. 6~8, p. 20 등)",
  "question": "RAG 시스템에 입력할 자연스럽고 명확한 질문",
  "difficulty": "상" | "중" | "하",
  "key_keywords": ["검색_핵심_키워드1", "키워드2", "키워드3"],
  "expected_answer": "문서에 근거한 구체적이고 정확한 정답 요약 (연도, 고유명사, 수치 명시)",
  "evaluation_intent": "이 질문을 통해 RAG 시스템의 어떤 검색/생성 능력을 평가하고자 하는지 의도"
}

반드시 백틱(```json) 없이 유효한 JSON 형식으로만 파싱 가능하게 출력해 주세요.
"""

print("Generating 30 high-quality RAG evaluation questions with gemini-3.8-flash...")
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=[uploaded_file, system_prompt],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )
)

raw_text = response.text.strip()
data = json.loads(raw_text)

# Save to json file with pretty print and utf-8
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n[성공] 총 {len(data)}개의 질의 질문이 '{output_json_path}'에 저장되었습니다!")

