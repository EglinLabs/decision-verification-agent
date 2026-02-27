import os
import time
import certifi
import json
from typing import Any, Dict, List, Optional, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, AnyHttpUrl

APP_NAME = "Decision + Verification Agent"

# ✅ Always load .env from the same folder as this file (works regardless of where uvicorn is launched)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# --- Security (read at import; you rarely change this)
API_KEY = os.getenv("API_KEY", "")

# --- Runtime controls
DEFAULT_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "12"))
MAX_CHECKS = int(os.getenv("MAX_CHECKS", "10"))
DECISION_TTL_S = int(os.getenv("DECISION_TTL_S", "300"))
MAX_LLM_TOKENS = int(os.getenv("MAX_LLM_TOKENS", "260"))

app = FastAPI(title=APP_NAME)


# ----------------------
# Models
# ----------------------
class VerifyHTTPCheck(BaseModel):
    type: Literal["http"] = "http"
    url: AnyHttpUrl
    method: Literal["GET", "HEAD"] = "GET"
    expect_status: int = 200
    must_contain: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)


class DecideRequest(BaseModel):
    goal: str = Field(..., description="What decision is needed and why")
    constraints: Dict[str, Any] = Field(default_factory=dict)
    checks: List[VerifyHTTPCheck] = Field(default_factory=list)
    allow_decide_on_partial: bool = False
    context: Dict[str, Any] = Field(default_factory=dict)


class DecideResponse(BaseModel):
    verdict: Literal["proceed", "wait", "block"]
    confidence: float
    risk: Literal["low", "medium", "high"]
    expires_in: int
    decision: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    failure_modes: List[str]


# ----------------------
# Auth
# ----------------------
def require_api_key(x_api_key: Optional[str]):
    # For local dev you *can* leave API_KEY empty. For production, set it.
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----------------------
# Verification: HTTP checks
# ----------------------
async def run_http_check(client: httpx.AsyncClient, check: VerifyHTTPCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    status = None
    error = None
    body_snippet = None

    try:
        resp = await client.request(
            method=check.method,
            url=str(check.url),
            headers=check.headers,
            timeout=DEFAULT_TIMEOUT_S,
            follow_redirects=True,
        )
        status = resp.status_code

        if check.method == "GET":
            text = resp.text or ""
            if check.must_contain:
                ok = (status == check.expect_status) and (check.must_contain in text)
            else:
                ok = (status == check.expect_status)
            body_snippet = text[:300] if text else None
        else:
            ok = (status == check.expect_status)

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)

    # classify transient failures (useful for verdict=wait)
    transient = False
    if error:
        transient = True
    if status and 500 <= status <= 599:
        transient = True
    if status == 429:
        transient = True

    return {
        "type": "http",
        "url": str(check.url),
        "method": check.method,
        "expect_status": check.expect_status,
        "must_contain": check.must_contain,
        "status": status,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "body_snippet": body_snippet,
    }


# ----------------------
# OpenRouter: decision compression (read env at call time; no caching)
# ----------------------
async def llm_decide_openrouter(payload: Dict[str, Any]) -> Dict[str, Any]:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set (check .env location/format)")

    system = (
        "You are a Decision+Verification agent for OTHER agents.\n"
        "Return ONLY a valid JSON object. No markdown. No extra text.\n"
        "Required keys:\n"
        "- verdict: one of ['proceed','wait','block']\n"
        "- confidence: number 0..1\n"
        "- risk: one of ['low','medium','high']\n"
        "- decision: { action: string, reason: string (<=20 words), constraints: object }\n"
        "- failure_modes: array of strings\n"
        "Rules:\n"
        "- If any non-transient check failed AND partial is not allowed: verdict='block'\n"
        "- If failures look transient (timeouts/5xx/429/DNS), prefer verdict='wait'\n"
    )

    user_obj = {
        "goal": payload.get("goal"),
        "constraints": payload.get("constraints", {}),
        "allow_decide_on_partial": payload.get("allow_decide_on_partial", False),
        "context": payload.get("context", {}),
        "evidence": payload.get("evidence", []),
    }

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": payload.get("referer", "http://localhost"),
        "X-Title": "DecisionVerificationAgent",
    }

    body = {
        "model": openrouter_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj)},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_LLM_TOKENS,
        # NOTE: Do not send response_format here to maximize compatibility on OpenRouter.
    }

    async with httpx.AsyncClient(verify=certifi.where()) as client:
        resp = await client.post(
            f"{openrouter_base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=45,
        )

        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:1200]}")

        data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()

    # Robust JSON extraction: parse directly, otherwise extract first {...} block
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
        raise


# ----------------------
# API
# ----------------------
@app.get("/health")
async def health():
    return {"ok": True, "name": APP_NAME}


@app.get("/debug/env")
async def debug_env(x_api_key: Optional[str] = Header(default=None)):
    """
    Temporary debugging endpoint.
    Delete before deploying publicly, or protect with API_KEY.
    """
    require_api_key(x_api_key)

    # Show only booleans / non-sensitive info
    return {
        "cwd": os.getcwd(),
        "base_dir": BASE_DIR,
        "env_file_exists": os.path.exists(os.path.join(BASE_DIR, ".env")),
        "openrouter_key_present": bool(os.getenv("OPENROUTER_API_KEY", "")),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "openrouter_base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    }


@app.post("/v1/decide", response_model=DecideResponse)
async def decide(req: DecideRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    if len(req.checks) > MAX_CHECKS:
        raise HTTPException(status_code=400, detail=f"Too many checks (max {MAX_CHECKS})")

    evidence: List[Dict[str, Any]] = []
    failure_modes: List[str] = []

    async with httpx.AsyncClient() as client:
        for c in req.checks:
            evidence.append(await run_http_check(client, c))

    any_failed = any(e.get("ok") is False for e in evidence)
    any_transient_fail = any((e.get("ok") is False) and (e.get("transient") is True) for e in evidence)
    any_hard_fail = any((e.get("ok") is False) and (e.get("transient") is False) for e in evidence)

    if any_failed:
        failure_modes.append("one_or_more_checks_failed")

    # Cheap guardrail:
    # If partial not allowed and there is any hard failure -> block without spending LLM tokens.
    if (not req.allow_decide_on_partial) and any_hard_fail:
        return DecideResponse(
            verdict="block",
            confidence=0.90,
            risk="high",
            expires_in=DECISION_TTL_S,
            decision={
                "action": "halt",
                "reason": "Verification failed",
                "constraints": req.constraints,
            },
            evidence=evidence,
            failure_modes=failure_modes + ["hard_failure"],
        )

    # If only transient failures and partial not allowed -> wait (still no LLM spend)
    if (not req.allow_decide_on_partial) and any_transient_fail and (not any_hard_fail):
        return DecideResponse(
            verdict="wait",
            confidence=0.75,
            risk="medium",
            expires_in=60,
            decision={
                "action": "retry",
                "reason": "Transient verification failure",
                "constraints": req.constraints,
            },
            evidence=evidence,
            failure_modes=failure_modes + ["transient_failure"],
        )

    # Otherwise: compress with LLM into structured decision
    try:
        out = await llm_decide_openrouter(
            {
                "goal": req.goal,
                "constraints": req.constraints,
                "allow_decide_on_partial": req.allow_decide_on_partial,
                "context": req.context,
                "evidence": evidence,
                "referer": req.context.get("referer", "http://localhost"),
            }
        )
    except Exception as e:
        # Fail-safe: block on LLM error
        return DecideResponse(
            verdict="block",
            confidence=0.85,
            risk="high",
            expires_in=DECISION_TTL_S,
            decision={
                "action": "halt",
                "reason": "Decision engine error",
                "constraints": req.constraints,
            },
            evidence=evidence,
            failure_modes=failure_modes + [f"llm_error:{str(e)[:220]}"],
        )

    return DecideResponse(
        verdict=out["verdict"],
        confidence=float(out["confidence"]),
        risk=out["risk"],
        expires_in=DECISION_TTL_S,
        decision=out["decision"],
        evidence=evidence,
        failure_modes=out.get("failure_modes", []) + failure_modes,
    )