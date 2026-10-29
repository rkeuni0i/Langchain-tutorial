"""Gemini API 429 (RESOURCE_EXHAUSTED) 에러 정밀 진단 및 디버깅 유틸리티

구글 Gemini API 호출 시 발생하는 429 오류의 원인을 정밀 분석하여
1) 사용자 프로젝트의 할당량 한도 초과 (Project Quota Exceeded)
2) 구글 클라우드 리전 리소스 일시 고갈 / DSQ 순간 가속 한도 (Google Cloud Capacity Exhaustion / DSQ Spike)
를 명확하게 구분하고 상세 디버깅 메시지를 출력합니다.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional

# Windows 콘솔 utf-8 출력 인코딩 보장
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def extract_gemini_error_payload(err: Exception) -> Optional[Dict[str, Any]]:
    """예외 객체 및 체인(__cause__, __context__)에서 Google API 에러 응답 페이로드(JSON/Dict)를 추출합니다."""
    # 1. 예외 객체 및 체인(__cause__, __context__) 속성 탐색
    target_err: Optional[BaseException] = err
    while target_err:
        for attr in ["details", "response_json", "error_details", "body"]:
            val = getattr(target_err, attr, None)
            if isinstance(val, dict):
                return val
            elif isinstance(val, list):
                return {"details": val}

        # response 객체(requests, httpx 등) 내 json() 탐색
        resp = getattr(target_err, "response", None)
        if resp is not None:
            try:
                if hasattr(resp, "json") and callable(resp.json):
                    j = resp.json()
                    if isinstance(j, dict):
                        return j
            except Exception:
                pass

        target_err = getattr(target_err, "__cause__", None) or getattr(
            target_err, "__context__", None
        )

    # 2. 문자열(str) 내 포함된 JSON 파싱 폴백
    err_str = str(err)
    json_patterns = [
        r"(\{.*\"error\"\s*:\s*\{.*\}\s*\})",
        r"(\{\s*\"code\"\s*:\s*429.*?\})",
    ]
    for pattern in json_patterns:
        match = re.search(pattern, err_str, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    return None


def diagnose_gemini_error(err: Exception) -> Dict[str, Any]:
    """Gemini API 에러(특히 429 RESOURCE_EXHAUSTED)를 분석하여
    '프로젝트 할당량 초과'와 '구글 클라우드 리소스 고갈'을 구분합니다.
    """
    err_str = str(err)
    payload = extract_gemini_error_payload(err)

    # 429 / RESOURCE_EXHAUSTED 여부 확인
    is_429 = (
        "429" in err_str
        or "RESOURCE_EXHAUSTED" in err_str
        or "RATE_LIMIT_EXCEEDED" in err_str
        or "GoogleRateLimitError" in type(err).__name__
        or (
            isinstance(payload, dict)
            and str(
                payload.get("code") or payload.get("error", {}).get("code")
            )
            == "429"
        )
    )

    if not is_429:
        return {
            "is_429": False,
            "error_type": "NOT_429",
            "category_name": "일반 오류 (Non-429 Error)",
            "message": err_str,
            "raw_payload": payload,
        }

    # details 배열 추출
    details_list: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        if "error" in payload and isinstance(payload["error"], dict):
            raw_details = payload["error"].get("details", [])
        else:
            raw_details = payload.get("details", [])
        if isinstance(raw_details, list):
            details_list = [d for d in raw_details if isinstance(d, dict)]
    elif isinstance(payload, list):
        details_list = [d for d in payload if isinstance(d, dict)]

    has_quota_failure = False
    quota_metrics: List[str] = []
    quota_limits: List[str] = []
    violation_descs: List[str] = []
    has_retry_info = False
    retry_delay: Optional[str] = None
    domain: Optional[str] = None
    reason: Optional[str] = None

    for d in details_list:
        dtype = str(d.get("@type", ""))

        # 1. QuotaFailure 검사
        if "QuotaFailure" in dtype or "quotaFailure" in dtype:
            has_quota_failure = True
            for v in d.get("violations", []):
                if isinstance(v, dict):
                    desc = v.get("description", "")
                    if desc:
                        violation_descs.append(desc)
                    if "quota_metric" in v:
                        quota_metrics.append(str(v["quota_metric"]))

        # 2. ErrorInfo 검사
        if "ErrorInfo" in dtype or "errorInfo" in dtype:
            reason = d.get("reason", reason)
            domain = d.get("domain", domain)
            metadata = d.get("metadata", {})
            if isinstance(metadata, dict):
                qm = metadata.get("quota_metric")
                ql = metadata.get("quota_limit")
                if qm:
                    quota_metrics.append(str(qm))
                if ql:
                    quota_limits.append(str(ql))
            if reason in ("RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED"):
                has_quota_failure = True

        # 3. RetryInfo 검사
        if "RetryInfo" in dtype or "retryInfo" in dtype:
            has_retry_info = True
            retry_delay = d.get("retry_delay", retry_delay)

    # 4. 페이로드 외 문자열 내의 Quota 메트릭 키워드 보조 검사
    known_metrics = [
        "GenerateContentRequestsPerMinutePerProject",
        "GenerateContentRequestsPerDayPerProject",
        "tokens_per_minute",
        "queries_per_minute",
        "quota_metric",
        "Quota exceeded",
    ]
    for kw in known_metrics:
        if kw.lower() in err_str.lower() and kw not in quota_metrics:
            quota_metrics.append(kw)
            has_quota_failure = True

    # 5. 핵심 판정 로직
    # - error.details 안에 QuotaFailure 객체나 구체적인 quota_metric이 명시되어 있다면 프로젝트 한도 초과
    # - 구체적인 메트릭 없이 재시도를 권고하는 메시지(RetryInfo 등)라면 리전 리소스 고갈이거나 DSQ 순간 가속 한도
    if has_quota_failure or quota_metrics or violation_descs:
        category = "PROJECT_QUOTA_EXCEEDED"
        category_name = "🚨 [프로젝트 할당량 초과] (Project Quota Limit Exceeded)"
        root_cause_summary = (
            "사용자 Google AI Studio / GCP 프로젝트의 API 키 할당량 한도(RPM/TPM/일일 RPD)를 초과했습니다."
        )
        action_guide = (
            "1. 분당 요청 한도(15 RPM / 1M TPM)의 경우: 약 1분간 대기 후 다시 시도하세요.\n"
            "2. 일일 한도(1,500 RPD) 소진 시: 태평양 표준시(PST) 자정 리셋을 기다리거나 유료 계정(Pay-as-you-go)으로 전환하세요.\n"
            "3. Google AI Studio 콘솔(https://aistudio.google.com/)에서 프로젝트 사용량과 한도를 확인하세요."
        )
    else:
        category = "GOOGLE_CLOUD_RESOURCE_EXHAUSTION"
        category_name = "☁️ [구글 클라우드 리전 리소스 일시 고갈] (Cloud Resource Exhaustion / DSQ Spike Limit)"
        root_cause_summary = (
            "사용자 프로젝트의 할당량 문제가 아닙니다!\n"
            "구글 클라우드 해당 리전의 AI 인프라(GPU/TPU) 순간 트래픽 폭증으로 인한 동적 서빙 큐(DSQ) 스파이크 제한이거나 일시적 서버 자원 고갈입니다."
        )
        delay_text = f"권장 재시도 대기시간: {retry_delay}" if retry_delay else "수 초~수십 초 대기 후 재시도 권장"
        action_guide = (
            f"1. {delay_text}\n"
            "2. 일시적인 클라우드 트래픽 급증이므로 지수 백오프(Exponential Backoff)를 두고 잠시 후 재호출하면 대부분 즉시 정상 처리됩니다."
        )

    # 포맷팅된 디버깅 리포트 생성 (Windows 콘솔 호환)
    report_lines = [
        "",
        "=" * 74,
        "       [Google Gemini API 429 RESOURCE_EXHAUSTED 원인 정밀 진단]",
        "=" * 74,
        f"  ▶ 판정 결과 : {category_name}",
        f"  ▶ 상세 원인 : {root_cause_summary}",
    ]

    if quota_metrics:
        report_lines.append(f"  ▶ 위반 메트릭(quota_metric)  : {', '.join(quota_metrics)}")
    if quota_limits:
        report_lines.append(f"  ▶ 위반 한도명(quota_limit)   : {', '.join(quota_limits)}")
    if violation_descs:
        report_lines.append(f"  ▶ 위반 세부설명(violations)   : {violation_descs[0]}")
    if retry_delay:
        report_lines.append(f"  ▶ 권장 재시도 대기(retry_delay): {retry_delay}")

    report_lines.extend([
        "-" * 74,
        "  💡 권장 조치 사항:",
    ])
    for guide_line in action_guide.split("\n"):
        report_lines.append(f"     {guide_line}")

    if details_list:
        report_lines.extend([
            "-" * 74,
            "  📋 에러 응답 페이로드 (error.details 세부 내용):",
        ])
        for idx, d in enumerate(details_list, 1):
            type_name = d.get("@type", "UnknownType").split(".")[-1]
            report_lines.append(f"     [{idx}] {type_name}: {json.dumps(d, ensure_ascii=False)}")

    report_lines.append("=" * 74 + "\n")
    debug_report_str = "\n".join(report_lines)

    return {
        "is_429": True,
        "category": category,
        "category_name": category_name,
        "root_cause": root_cause_summary,
        "quota_metrics": list(set(quota_metrics)),
        "quota_limits": list(set(quota_limits)),
        "violations": violation_descs,
        "retry_delay": retry_delay,
        "has_retry_info": has_retry_info,
        "action_guide": action_guide,
        "raw_details": details_list or payload,
        "debug_report": debug_report_str,
    }


def print_gemini_429_diagnosis(err: Exception) -> Dict[str, Any]:
    """Gemini 429 에러를 진단하고 콘솔에 서식화된 디버깅 메시지를 출력합니다."""
    diag = diagnose_gemini_error(err)
    if diag.get("is_429"):
        print(diag["debug_report"])
    else:
        print(f"[알림] Gemini API 오류 발생 (429 아님): {err}")
    return diag


if __name__ == "__main__":
    # 자체 진단 데모 테스트
    print(">>> [테스트 1] 프로젝트 한도 초과 (QuotaFailure 포함 시뮬레이션)")
    mock_project_error = Exception(
        '429 RESOURCE_EXHAUSTED: {"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED", "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"subject": "project:12345678", "description": "Quota exceeded for quota metric GenerateContentRequestsPerMinutePerProject and limit 15"}]}, {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "RATE_LIMIT_EXCEEDED", "metadata": {"quota_metric": "generativelanguage.googleapis.com/generate_content_requests", "quota_limit": "GenerateContentRequestsPerMinutePerProject"}}]}}'
    )
    print_gemini_429_diagnosis(mock_project_error)

    print(">>> [테스트 2] 구글 클라우드 리전 리소스 일시 고갈 (RetryInfo만 포함 시뮬레이션)")
    mock_cloud_error = Exception(
        '429 RESOURCE_EXHAUSTED: {"error": {"code": 429, "message": "The service is temporarily unavailable in this region due to traffic spike. Please retry with backoff.", "status": "RESOURCE_EXHAUSTED", "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retry_delay": "15s"}]}}'
    )
    print_gemini_429_diagnosis(mock_cloud_error)
