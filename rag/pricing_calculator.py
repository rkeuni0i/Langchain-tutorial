"""Google Cloud Model Garden & Vertex AI 토큰 및 과금 계산기 모듈

참조 공식 가격표:
1. Google Cloud Model Garden Pricing:
   https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform
2. Google Cloud Model Garden Partner Models Pricing:
   https://cloud.google.com/model-garden/partner-models/pricing
3. Vertex AI Generative AI Pricing:
   https://cloud.google.com/vertex-ai/generative-ai/pricing
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

# 환율 기준 (1 USD = 1,400 KRW)
USD_TO_KRW = 1400.0

# 100만(1M) 토큰 당 가격 (USD)
MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    "claude-haiku-5-5": {
        "display_name": "Claude Haiku 5.5",
        "provider": "Anthropic",
        "deployment": "Vertex AI Partner Model",
        "input_price_per_m": 0.10,    # $0.10 / 1M tokens (prompts <= 100K)
        "output_price_per_m": 0.50,   # $0.50 / 1M tokens
        "cache_read_per_m": 0.01,
        "pricing_source": "https://cloud.google.com/model-garden/partner-models/pricing",
        "badge_color": "orange",
        "description": "초저지연, 고성능 경량 추론 및 분류 모델 (Anthropic)",
    },
    "gemma-4-26b": {
        "display_name": "Gemma 4 26B",
        "provider": "Google DeepMind",
        "deployment": "Model Garden MaaS",
        "input_price_per_m": 0.15,    # $0.15 / 1M tokens
        "output_price_per_m": 0.60,   # $0.60 / 1M tokens
        "cache_read_per_m": 0.015,
        "pricing_source": "https://cloud.google.com/model-garden/pricing",
        "badge_color": "indigo",
        "description": "25.2B MoE 구조 (3.8B active) 고효율 오픈 가중치 모델",
    },
    "gemini-3.5-flash-lite": {
        "display_name": "Gemini 3.5 Flash-Lite",
        "provider": "Google DeepMind",
        "deployment": "Vertex AI First-Party",
        "input_price_per_m": 0.30,    # $0.30 / 1M tokens
        "output_price_per_m": 2.50,   # $2.50 / 1M tokens
        "cache_read_per_m": 0.03,
        "pricing_source": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "badge_color": "emerald",
        "description": "문서 파싱 및 라우팅에 최적화된 초경량 고속 모델",
    },
    "deepseek-v4.1-flash": {
        "display_name": "DeepSeek V4.1 Flash",
        "provider": "DeepSeek AI",
        "deployment": "Model Garden MaaS",
        "input_price_per_m": 0.30,    # $0.30 / 1M tokens
        "output_price_per_m": 1.20,   # $1.20 / 1M tokens
        "cache_read_per_m": 0.03,
        "pricing_source": "https://cloud.google.com/model-garden/partner-models/pricing",
        "badge_color": "teal",
        "description": "552B MoE 기반 초고속 대용량 컨텍스트 모델",
    },
    "gemini-3.8-flash": {
        "display_name": "Gemini 3.8 Flash",
        "provider": "Google",
        "deployment": "Google AI / Vertex AI",
        "input_price_per_m": 0.75,    # 프로모션 할인가 ($0.75 / 1M, 표준 $1.50)
        "output_price_per_m": 3.75,   # 프로모션 할인가 ($3.75 / 1M, 표준 $7.50)
        "cache_read_per_m": 0.075,
        "pricing_source": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "badge_color": "blue",
        "description": "다단계 복합 추론 및 플래그십 범용 고성능 Flash 모델",
    },
    "gemma4:e2b": {
        "display_name": "Gemma 4:e2b (Ollama)",
        "provider": "Local Host",
        "deployment": "Local Hardware",
        "input_price_per_m": 0.00,
        "output_price_per_m": 0.00,
        "cache_read_per_m": 0.00,
        "pricing_source": "Free / Local Execution",
        "badge_color": "purple",
        "description": "로컬 GPU/CPU 구동 완전 무료 온프레미스 모델",
    },
    "gemini-embedding-2": {
        "display_name": "Gemini Embedding 2",
        "provider": "Google DeepMind",
        "deployment": "Vertex AI Embeddings",
        "input_price_per_m": 0.20,    # $0.20 / 1M tokens
        "output_price_per_m": 0.00,   # 임베딩 출력 무료
        "cache_read_per_m": 0.00,
        "pricing_source": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        "badge_color": "slate",
        "description": "텍스트 및 멀티모달 벡터 임베딩 생성 모델",
    },
}


def get_pricing_key(model_name: str) -> str:
    """임의의 모델명을 정규화하여 가격표 키와 매핑"""
    m = model_name.lower().strip()
    if "deepseek" in m:
        return "deepseek-v4.1-flash"
    elif "haiku" in m or "claude" in m:
        return "claude-haiku-5-5"
    elif "3.5-flash-lite" in m or "3.5-lite" in m:
        return "gemini-3.5-flash-lite"
    elif "26b" in m or ("gemma" in m and "e2b" not in m and ":" not in m and "ollama" not in m):
        return "gemma-4-26b"
    elif ":" in m or "ollama" in m or "e2b" in m:
        return "gemma4:e2b"
    elif "embedding" in m:
        return "gemini-embedding-2"
    else:
        return "gemini-3.8-flash"


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> Dict[str, Any]:
    """주어진 모델명과 입출력 토큰 수를 바탕으로 USD 및 KRW 과금액 계산"""
    key = get_pricing_key(model_name)
    pricing = MODEL_PRICING.get(key, MODEL_PRICING["gemini-3.8-flash"])

    input_cost = (input_tokens / 1_000_000.0) * pricing["input_price_per_m"]
    output_cost = (output_tokens / 1_000_000.0) * pricing["output_price_per_m"]
    cache_cost = (cache_read_tokens / 1_000_000.0) * pricing.get("cache_read_per_m", 0.0)

    total_cost_usd = input_cost + output_cost + cache_cost
    total_cost_krw = total_cost_usd * USD_TO_KRW

    return {
        "model_key": key,
        "display_name": pricing["display_name"],
        "provider": pricing["provider"],
        "deployment": pricing["deployment"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 7),
        "output_cost_usd": round(output_cost, 7),
        "cost_usd": round(total_cost_usd, 7),
        "cost_krw": round(total_cost_krw, 4),
        "pricing": {
            "input_price_per_m": pricing["input_price_per_m"],
            "output_price_per_m": pricing["output_price_per_m"],
            "cache_read_per_m": pricing.get("cache_read_per_m", 0.0),
            "pricing_source": pricing["pricing_source"],
        },
    }


class UsageTracker:
    """실시간 RAG 질의응답 세션의 토큰 사용량 및 과금액 누적 추적 클래스"""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_queries = 0
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self._init_models()

    def _init_models(self):
        for k, v in MODEL_PRICING.items():
            self.model_stats[k] = {
                "model_key": k,
                "display_name": v["display_name"],
                "provider": v["provider"],
                "deployment": v["deployment"],
                "badge_color": v["badge_color"],
                "query_count": 0,
                "aux_calls": 0,  # 질의 확장/리랭킹/임베딩 등 보조 호출 횟수 (질의 수에는 포함하지 않음)
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_krw": 0.0,
                "input_price_per_m": v["input_price_per_m"],
                "output_price_per_m": v["output_price_per_m"],
                "pricing_source": v["pricing_source"],
                "description": v["description"],
            }

    def record_query(
        self,
        model: str,
        query: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cost_krw: float,
        latency: float,
    ):
        with self._lock:
            key = get_pricing_key(model)
            self.total_queries += 1
            if key not in self.model_stats:
                self.model_stats[key] = {
                    "model_key": key,
                    "display_name": model,
                    "provider": "Unknown",
                    "deployment": "Unknown",
                    "badge_color": "slate",
                    "query_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_krw": 0.0,
                    "input_price_per_m": 0.0,
                    "output_price_per_m": 0.0,
                    "pricing_source": "#",
                    "description": "",
                }
            m = self.model_stats[key]
            m["query_count"] += 1
            m["input_tokens"] += input_tokens
            m["output_tokens"] += output_tokens
            m["total_tokens"] += (input_tokens + output_tokens)
            m["cost_usd"] = round(m["cost_usd"] + cost_usd, 6)
            m["cost_krw"] = round(m["cost_krw"] + cost_krw, 3)

            # 최근 50건 히스토리 큐 관리
            self.history.insert(
                0,
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "model": model,
                    "model_key": key,
                    "display_name": m.get("display_name", model),
                    "query_preview": query[:60] + ("..." if len(query) > 60 else ""),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": round(cost_usd, 6),
                    "cost_krw": round(cost_krw, 3),
                    "latency": latency,
                },
            )
            if len(self.history) > 50:
                self.history.pop()

    def record_usage(
        self,
        model: str,
        stage: str,
        query: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cost_krw: float,
    ):
        """보조 호출(질의 확장/리랭킹/임베딩) 사용량 누적. 질의 수(total_queries, query_count)는 올리지 않는다."""
        with self._lock:
            key = get_pricing_key(model)
            m = self.model_stats.get(key)
            if m is None:
                return
            m["aux_calls"] = m.get("aux_calls", 0) + 1
            m["input_tokens"] += input_tokens
            m["output_tokens"] += output_tokens
            m["total_tokens"] += input_tokens + output_tokens
            m["cost_usd"] = round(m["cost_usd"] + cost_usd, 6)
            m["cost_krw"] = round(m["cost_krw"] + cost_krw, 3)

            label = f"[{stage}] {query}"
            self.history.insert(
                0,
                {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "model": model,
                    "model_key": key,
                    "display_name": m.get("display_name", model),
                    "query_preview": label[:60] + ("..." if len(label) > 60 else ""),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": round(cost_usd, 6),
                    "cost_krw": round(cost_krw, 3),
                    "latency": 0.0,
                },
            )
            if len(self.history) > 50:
                self.history.pop()

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_input = sum(m["input_tokens"] for m in self.model_stats.values())
            total_output = sum(m["output_tokens"] for m in self.model_stats.values())
            total_tokens = total_input + total_output
            total_usd = sum(m["cost_usd"] for m in self.model_stats.values())
            total_krw = sum(m["cost_krw"] for m in self.model_stats.values())

            # 가성비 최고 모델 (사용된 모델 중 토큰당 비용이 가장 저렴한 모델)
            used_models = [m for m in self.model_stats.values() if m["query_count"] > 0]
            cheapest_model = "None"
            if used_models:
                # 0원 모델 제외 시 유료 최저가, 혹은 전체 최저가
                sorted_by_price = sorted(
                    used_models, key=lambda x: (x["input_price_per_m"] + x["output_price_per_m"])
                )
                cheapest_model = sorted_by_price[0]["display_name"]

            return {
                "total_queries": self.total_queries,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_usd, 6),
                "total_cost_krw": round(total_krw, 2),
                "usd_to_krw_rate": USD_TO_KRW,
                "cheapest_model": cheapest_model,
                "models": list(self.model_stats.values()),
                "recent_history": self.history[:30],
                "pricing_references": [
                    {
                        "title": "Google Cloud Model Garden Pricing",
                        "url": "https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform",
                        "description": "Agent Platform 및 Model Garden MaaS (Gemma, DeepSeek 등) 서버리스 과금 기준",
                    },
                    {
                        "title": "Google Cloud Model Garden Partner Models Pricing",
                        "url": "https://cloud.google.com/model-garden/partner-models/pricing",
                        "description": "Anthropic Claude, DeepSeek 등 파트너 모델 토큰별 종량제 요금표",
                    },
                    {
                        "title": "Vertex AI Generative AI Pricing",
                        "url": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
                        "description": "Gemini 3.5 Flash-Lite, Gemini 3.8 Flash 및 임베딩 모델 공식 요금표",
                    },
                ],
            }

    def reset(self):
        with self._lock:
            self.total_queries = 0
            self.history.clear()
            self._init_models()


# 전역 싱글톤 트래커 인스턴스
global_usage_tracker = UsageTracker()

