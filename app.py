import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Union, Annotated

import certifi
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, AnyHttpUrl, EmailStr

APP_NAME = "Decision + Verification Agent"

# ----------------------
# Env / config
# ----------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

API_KEY = os.getenv("API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://decision-verification-agent.onrender.com")
DOCS_URL = os.getenv("DOCS_URL", "https://eglin-labs-decision-twu5.bolt.host/docs")

DEFAULT_TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "12"))
MAX_CHECKS = int(os.getenv("MAX_CHECKS", "10"))
ACP_MAX_CHECKS = int(os.getenv("ACP_MAX_CHECKS", "3"))
DECISION_TTL_S = int(os.getenv("DECISION_TTL_S", "300"))
MAX_LLM_TOKENS = int(os.getenv("MAX_LLM_TOKENS", "260"))
ISSUE_SIGNUP_KEYS = os.getenv("ISSUE_SIGNUP_KEYS", "true").lower() == "true"

DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "eglin.db")))

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


class VerifyRPCCheck(BaseModel):
    type: Literal["rpc"] = "rpc"
    rpc_url: AnyHttpUrl
    method_name: str
    params: List[Any] = Field(default_factory=list)
    expect_hex_result: bool = False
    min_block_number: Optional[int] = None


class VerifyPriceCheck(BaseModel):
    type: Literal["price"] = "price"
    source: Literal["coingecko"] = "coingecko"
    asset_id: str
    quote: str = "usd"
    min_price: Optional[float] = None
    max_price: Optional[float] = None


class VerifyTxCheck(BaseModel):
    type: Literal["tx"] = "tx"
    rpc_url: AnyHttpUrl
    tx_hash: str
    require_success: bool = True


class VerifyDexPriceCheck(BaseModel):
    type: Literal["dex_price"] = "dex_price"
    chain: str
    pair_address: str
    min_price_usd: Optional[float] = None
    max_price_usd: Optional[float] = None
    min_liquidity_usd: Optional[float] = None
    min_volume_h24_usd: Optional[float] = None


class VerifyStablecoinDepegCheck(BaseModel):
    type: Literal["stablecoin_depeg"] = "stablecoin_depeg"
    asset_id: str
    quote: str = "usd"
    warn_deviation_pct: float = 0.005
    block_deviation_pct: float = 0.02


class VerifyBridgeExploitMonitorCheck(BaseModel):
    type: Literal["bridge_exploit_monitor"] = "bridge_exploit_monitor"
    status_url: AnyHttpUrl
    expect_status: int = 200
    safe_keywords: List[str] = Field(default_factory=lambda: ["operational", "all systems operational"])
    danger_keywords: List[str] = Field(default_factory=lambda: [
        "exploit", "incident", "degraded", "outage", "down", "halted", "paused", "security"
    ])
    headers: Dict[str, str] = Field(default_factory=dict)


class VerifyRugPullRiskCheck(BaseModel):
    type: Literal["rug_pull_risk"] = "rug_pull_risk"
    chain: str
    pair_address: str
    min_liquidity_usd: float = 50000
    min_pair_age_minutes: int = 1440
    max_fdv_to_liquidity_ratio: float = 20.0
    min_buys_h24: Optional[int] = None
    min_sells_h24: Optional[int] = None


CheckType = Annotated[
    Union[
        VerifyHTTPCheck,
        VerifyRPCCheck,
        VerifyPriceCheck,
        VerifyTxCheck,
        VerifyDexPriceCheck,
        VerifyStablecoinDepegCheck,
        VerifyBridgeExploitMonitorCheck,
        VerifyRugPullRiskCheck,
    ],
    Field(discriminator="type"),
]


class DecideRequest(BaseModel):
    goal: str = Field(..., description="What decision is needed and why")
    constraints: Dict[str, Any] = Field(default_factory=dict)
    checks: List[CheckType] = Field(default_factory=list)
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


class GuardRequest(BaseModel):
    action: Literal["swap", "transfer", "bridge", "yield_deposit", "generic"]
    chain: Optional[str] = "base"
    pair_address: Optional[str] = None
    stablecoin_asset_id: Optional[str] = "usd-coin"
    rpc_url: Optional[AnyHttpUrl] = None
    tx_hash: Optional[str] = None
    bridge_status_url: Optional[AnyHttpUrl] = None
    amount_usd: Optional[float] = None
    strict_mode: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    name: str
    slug: str
    status: Literal["live", "coming_soon", "in_development"]
    description: str
    endpoints: List[str] = Field(default_factory=list)


class ApiKeySignupRequest(BaseModel):
    email: EmailStr
    company: Optional[str] = None
    full_name: Optional[str] = None
    use_case: Optional[str] = None
    website: Optional[AnyHttpUrl] = None
    requested_plan: Optional[Literal["Basic", "DeFi", "Execution"]] = None


class ApiKeySignupResponse(BaseModel):
    ok: bool
    message: str
    api_key: Optional[str] = None
    docs_url: str
    base_url: str


class StablecoinIntelligenceRequest(BaseModel):
    asset_ids: List[str] = Field(default_factory=lambda: ["usd-coin", "tether", "dai"])
    quote: str = "usd"
    warn_deviation_pct: float = 0.005
    block_deviation_pct: float = 0.02


class BridgeRiskRequest(BaseModel):
    bridge_status_urls: List[AnyHttpUrl]
    strict_mode: bool = True


class YieldRouteOption(BaseModel):
    name: str
    apy: float
    chain: str = "base"
    protocol: Optional[str] = None
    risk_score: float = 50.0
    liquidity_usd: Optional[float] = None
    stablecoin_asset_id: Optional[str] = "usd-coin"


class YieldRoutingRequest(BaseModel):
    options: List[YieldRouteOption]
    min_liquidity_usd: float = 50000
    max_risk_score: float = 70.0
    prefer_chain: Optional[str] = None


# ----------------------
# Product catalog
# ----------------------
AGENTS: List[AgentInfo] = [
    AgentInfo(
        name="Decision Verification Agent",
        slug="decision-verification-agent",
        status="live",
        description="AI infrastructure agent that verifies conditions before autonomous systems act.",
        endpoints=["/v1/decide", "/v1/acp/decide", "/v1/guard", "/v1/acp/guard"],
    ),
    AgentInfo(
        name="Execution Guard Engine",
        slug="execution-guard-engine",
        status="live",
        description="Universal guard endpoint that automatically builds the correct safety checks before execution.",
        endpoints=["/v1/guard", "/v1/acp/guard", "/v1/agents/execution-guard/execute"],
    ),
    AgentInfo(
        name="DeFi Risk Verification Module",
        slug="defi-risk-verification-module",
        status="live",
        description="DeFi-focused checks for stablecoin depeg, DEX liquidity, rug-pull risk, and bridge monitoring.",
        endpoints=["/v1/agents/defi-risk/verify"],
    ),
    AgentInfo(
        name="Stablecoin Intelligence Agent",
        slug="stablecoin-intelligence-agent",
        status="coming_soon",
        description="Stablecoin monitoring and intelligence agent.",
        endpoints=["/v1/agents/stablecoin-intelligence/analyze"],
    ),
    AgentInfo(
        name="Bridge Risk Agent",
        slug="bridge-risk-agent",
        status="coming_soon",
        description="Bridge monitoring and route risk analysis.",
        endpoints=["/v1/agents/bridge-risk/analyze"],
    ),
    AgentInfo(
        name="Yield Routing Agent",
        slug="yield-routing-agent",
        status="coming_soon",
        description="Yield routing and allocation agent.",
        endpoints=["/v1/agents/yield-routing/recommend"],
    ),
    AgentInfo(
        name="Agent-to-Agent Verification Layer",
        slug="agent-to-agent-verification-layer",
        status="coming_soon",
        description="ACP-native machine-to-machine verification layer.",
        endpoints=["/v1/acp/decide", "/v1/acp/guard"],
    ),
]


# ----------------------
# Database helpers
# ----------------------
def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_key_signups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                company TEXT,
                full_name TEXT,
                use_case TEXT,
                website TEXT,
                requested_plan TEXT,
                issued_key TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS issued_api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                api_key TEXT NOT NULL UNIQUE,
                label TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def db_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@app.on_event("startup")
async def startup_event():
    init_db()


# ----------------------
# Auth
# ----------------------
def require_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----------------------
# Helpers
# ----------------------
def _is_hex_string(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("0x")


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def _default_rpc_url_for_chain(chain: Optional[str]) -> Optional[str]:
    mapping = {
        "base": "https://mainnet.base.org",
        "ethereum": "https://ethereum-rpc.publicnode.com",
        "arbitrum": "https://arbitrum-one-rpc.publicnode.com",
        "optimism": "https://optimism-rpc.publicnode.com",
        "polygon": "https://polygon-bor-rpc.publicnode.com",
    }
    return mapping.get((chain or "").lower())


def build_guard_checks(payload: GuardRequest) -> List[CheckType]:
    checks: List[CheckType] = []

    rpc_url = str(payload.rpc_url) if payload.rpc_url else _default_rpc_url_for_chain(payload.chain)

    if rpc_url:
        checks.append(
            VerifyRPCCheck(
                type="rpc",
                rpc_url=rpc_url,
                method_name="eth_blockNumber",
                params=[],
                expect_hex_result=True,
            )
        )

    if payload.action in ("swap", "yield_deposit", "generic"):
        if payload.stablecoin_asset_id:
            checks.append(
                VerifyStablecoinDepegCheck(
                    type="stablecoin_depeg",
                    asset_id=payload.stablecoin_asset_id,
                    quote="usd",
                    warn_deviation_pct=0.005,
                    block_deviation_pct=0.02,
                )
            )

        if payload.pair_address and payload.chain:
            checks.append(
                VerifyDexPriceCheck(
                    type="dex_price",
                    chain=payload.chain,
                    pair_address=payload.pair_address,
                    min_liquidity_usd=50000,
                )
            )
            checks.append(
                VerifyRugPullRiskCheck(
                    type="rug_pull_risk",
                    chain=payload.chain,
                    pair_address=payload.pair_address,
                    min_liquidity_usd=50000,
                    min_pair_age_minutes=1440,
                    max_fdv_to_liquidity_ratio=20.0,
                )
            )

    if payload.action == "bridge" and payload.bridge_status_url:
        checks.append(
            VerifyBridgeExploitMonitorCheck(
                type="bridge_exploit_monitor",
                status_url=payload.bridge_status_url,
                expect_status=200,
            )
        )

    if payload.action == "transfer" and payload.tx_hash and rpc_url:
        checks.append(
            VerifyTxCheck(
                type="tx",
                rpc_url=rpc_url,
                tx_hash=payload.tx_hash,
                require_success=True,
            )
        )

    return checks


# ----------------------
# Verification checks
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
                ok = status == check.expect_status
            body_snippet = text[:300] if text else None
        else:
            ok = status == check.expect_status

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error) or (status is not None and (500 <= status <= 599 or status == 429))

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


async def run_rpc_check(client: httpx.AsyncClient, check: VerifyRPCCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    error = None
    result = None
    status_code = None

    try:
        resp = await client.post(
            str(check.rpc_url),
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": check.method_name,
                "params": check.params,
            },
            timeout=DEFAULT_TIMEOUT_S,
        )
        status_code = resp.status_code
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            error = json.dumps(payload["error"])[:300]
        else:
            result = payload.get("result")

            if check.expect_hex_result and not _is_hex_string(result):
                error = "Expected hex result but got non-hex result"
            elif check.min_block_number is not None:
                if not _is_hex_string(result):
                    error = "Expected hex block number result"
                else:
                    block_number = int(result, 16)
                    ok = block_number >= check.min_block_number
            else:
                ok = result is not None

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error)

    return {
        "type": "rpc",
        "rpc_url": str(check.rpc_url),
        "method_name": check.method_name,
        "params": check.params,
        "expect_hex_result": check.expect_hex_result,
        "min_block_number": check.min_block_number,
        "status": status_code,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "result": result,
    }


async def run_price_check(client: httpx.AsyncClient, check: VerifyPriceCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    error = None
    price = None
    status_code = None

    try:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": check.asset_id, "vs_currencies": check.quote},
            timeout=DEFAULT_TIMEOUT_S,
        )
        status_code = resp.status_code
        resp.raise_for_status()
        payload = resp.json()

        price = payload.get(check.asset_id, {}).get(check.quote)

        if price is None:
            error = "Price not found in source response"
        else:
            meets_min = check.min_price is None or price >= check.min_price
            meets_max = check.max_price is None or price <= check.max_price
            ok = meets_min and meets_max

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error)

    return {
        "type": "price",
        "source": check.source,
        "asset_id": check.asset_id,
        "quote": check.quote,
        "min_price": check.min_price,
        "max_price": check.max_price,
        "status": status_code,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "price": price,
    }


async def run_tx_check(client: httpx.AsyncClient, check: VerifyTxCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    error = None
    receipt = None
    status_code = None

    try:
        resp = await client.post(
            str(check.rpc_url),
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getTransactionReceipt",
                "params": [check.tx_hash],
            },
            timeout=DEFAULT_TIMEOUT_S,
        )
        status_code = resp.status_code
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            error = json.dumps(payload["error"])[:300]
        else:
            receipt = payload.get("result")
            if receipt is None:
                error = "Transaction receipt not found yet"
            else:
                tx_status = receipt.get("status")
                if check.require_success:
                    ok = tx_status == "0x1"
                    if not ok:
                        error = f"Transaction status was {tx_status}"
                else:
                    ok = True

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error)

    return {
        "type": "tx",
        "rpc_url": str(check.rpc_url),
        "tx_hash": check.tx_hash,
        "require_success": check.require_success,
        "status": status_code,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "receipt": receipt,
    }


async def run_dex_price_check(client: httpx.AsyncClient, check: VerifyDexPriceCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    error = None
    status_code = None
    pair = None
    price_usd = None
    liquidity_usd = None
    volume_h24 = None

    try:
        resp = await client.get(
            f"https://api.dexscreener.com/latest/dex/pairs/{check.chain}/{check.pair_address}",
            timeout=DEFAULT_TIMEOUT_S,
        )
        status_code = resp.status_code
        resp.raise_for_status()
        payload = resp.json()

        pairs = payload.get("pairs") or []
        if not pairs:
            error = "Pair not found"
        else:
            pair = pairs[0]
            price_usd = _to_float(pair.get("priceUsd"))
            liquidity_usd = _to_float((pair.get("liquidity") or {}).get("usd"))
            volume_h24 = _to_float((pair.get("volume") or {}).get("h24"))

            checks_ok = [
                price_usd is not None,
                check.min_price_usd is None or (price_usd is not None and price_usd >= check.min_price_usd),
                check.max_price_usd is None or (price_usd is not None and price_usd <= check.max_price_usd),
                check.min_liquidity_usd is None or (liquidity_usd is not None and liquidity_usd >= check.min_liquidity_usd),
                check.min_volume_h24_usd is None or (volume_h24 is not None and volume_h24 >= check.min_volume_h24_usd),
            ]
            ok = all(checks_ok)

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error)

    return {
        "type": "dex_price",
        "chain": check.chain,
        "pair_address": check.pair_address,
        "min_price_usd": check.min_price_usd,
        "max_price_usd": check.max_price_usd,
        "min_liquidity_usd": check.min_liquidity_usd,
        "min_volume_h24_usd": check.min_volume_h24_usd,
        "status": status_code,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "price_usd": price_usd,
        "liquidity_usd": liquidity_usd,
        "volume_h24_usd": volume_h24,
        "pair_url": pair.get("url") if pair else None,
        "dex_id": pair.get("dexId") if pair else None,
    }


async def run_stablecoin_depeg_check(client: httpx.AsyncClient, check: VerifyStablecoinDepegCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    error = None
    price = None
    deviation_pct = None
    severity = "unknown"
    status_code = None

    try:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": check.asset_id, "vs_currencies": check.quote},
            timeout=DEFAULT_TIMEOUT_S,
        )
        status_code = resp.status_code
        resp.raise_for_status()
        payload = resp.json()

        price = payload.get(check.asset_id, {}).get(check.quote)
        if price is None:
            error = "Stablecoin price not found"
        else:
            deviation_pct = abs(price - 1.0)
            ok = deviation_pct <= check.block_deviation_pct

            if deviation_pct <= check.warn_deviation_pct:
                severity = "safe"
            elif deviation_pct <= check.block_deviation_pct:
                severity = "warning"
            else:
                severity = "danger"

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error)

    return {
        "type": "stablecoin_depeg",
        "asset_id": check.asset_id,
        "quote": check.quote,
        "warn_deviation_pct": check.warn_deviation_pct,
        "block_deviation_pct": check.block_deviation_pct,
        "status": status_code,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "price": price,
        "deviation_pct": deviation_pct,
        "severity": severity,
    }


async def run_bridge_exploit_monitor_check(client: httpx.AsyncClient, check: VerifyBridgeExploitMonitorCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    status = None
    error = None
    body_snippet = None
    risk_level = "unknown"

    try:
        resp = await client.get(
            str(check.status_url),
            headers=check.headers,
            timeout=DEFAULT_TIMEOUT_S,
            follow_redirects=True,
        )
        status = resp.status_code
        text = resp.text or ""
        body_snippet = text[:300] if text else None

        has_danger = _contains_any(text, check.danger_keywords)
        has_safe = _contains_any(text, check.safe_keywords)

        if status == check.expect_status and not has_danger:
            ok = True
            risk_level = "low" if has_safe else "medium"
        else:
            ok = False
            risk_level = "high" if has_danger else "medium"

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error) or (status is not None and (500 <= status <= 599 or status == 429))

    return {
        "type": "bridge_exploit_monitor",
        "status_url": str(check.status_url),
        "expect_status": check.expect_status,
        "status": status,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "risk_level": risk_level,
        "body_snippet": body_snippet,
    }


async def run_rug_pull_risk_check(client: httpx.AsyncClient, check: VerifyRugPullRiskCheck) -> Dict[str, Any]:
    t0 = time.time()
    ok = False
    error = None
    status_code = None
    pair = None
    liquidity_usd = None
    fdv = None
    fdv_to_liquidity_ratio = None
    pair_age_minutes = None
    buys_h24 = None
    sells_h24 = None
    risk_level = "unknown"

    try:
        resp = await client.get(
            f"https://api.dexscreener.com/latest/dex/pairs/{check.chain}/{check.pair_address}",
            timeout=DEFAULT_TIMEOUT_S,
        )
        status_code = resp.status_code
        resp.raise_for_status()
        payload = resp.json()

        pairs = payload.get("pairs") or []
        if not pairs:
            error = "Pair not found"
        else:
            pair = pairs[0]
            liquidity_usd = _to_float((pair.get("liquidity") or {}).get("usd"))
            fdv = _to_float(pair.get("fdv"))
            buys_h24 = (pair.get("txns") or {}).get("h24", {}).get("buys")
            sells_h24 = (pair.get("txns") or {}).get("h24", {}).get("sells")

            created_at = pair.get("pairCreatedAt")
            if created_at:
                pair_age_minutes = int((time.time() * 1000 - created_at) / 60000)

            if liquidity_usd and fdv:
                fdv_to_liquidity_ratio = fdv / liquidity_usd if liquidity_usd > 0 else None

            checks_ok = [
                liquidity_usd is not None and liquidity_usd >= check.min_liquidity_usd,
                pair_age_minutes is not None and pair_age_minutes >= check.min_pair_age_minutes,
                fdv_to_liquidity_ratio is not None and fdv_to_liquidity_ratio <= check.max_fdv_to_liquidity_ratio,
            ]

            if check.min_buys_h24 is not None:
                checks_ok.append(buys_h24 is not None and buys_h24 >= check.min_buys_h24)

            if check.min_sells_h24 is not None:
                checks_ok.append(sells_h24 is not None and sells_h24 >= check.min_sells_h24)

            ok = all(checks_ok)
            risk_level = "low" if ok else "high"

    except Exception as e:
        error = str(e)

    latency_ms = int((time.time() - t0) * 1000)
    transient = bool(error)

    return {
        "type": "rug_pull_risk",
        "chain": check.chain,
        "pair_address": check.pair_address,
        "min_liquidity_usd": check.min_liquidity_usd,
        "min_pair_age_minutes": check.min_pair_age_minutes,
        "max_fdv_to_liquidity_ratio": check.max_fdv_to_liquidity_ratio,
        "min_buys_h24": check.min_buys_h24,
        "min_sells_h24": check.min_sells_h24,
        "status": status_code,
        "latency_ms": latency_ms,
        "ok": ok,
        "transient": transient,
        "error": error,
        "liquidity_usd": liquidity_usd,
        "fdv": fdv,
        "fdv_to_liquidity_ratio": fdv_to_liquidity_ratio,
        "pair_age_minutes": pair_age_minutes,
        "buys_h24": buys_h24,
        "sells_h24": sells_h24,
        "risk_level": risk_level,
        "pair_url": pair.get("url") if pair else None,
    }


async def run_check(client: httpx.AsyncClient, check: CheckType) -> Dict[str, Any]:
    if isinstance(check, VerifyHTTPCheck):
        return await run_http_check(client, check)
    if isinstance(check, VerifyRPCCheck):
        return await run_rpc_check(client, check)
    if isinstance(check, VerifyPriceCheck):
        return await run_price_check(client, check)
    if isinstance(check, VerifyTxCheck):
        return await run_tx_check(client, check)
    if isinstance(check, VerifyDexPriceCheck):
        return await run_dex_price_check(client, check)
    if isinstance(check, VerifyStablecoinDepegCheck):
        return await run_stablecoin_depeg_check(client, check)
    if isinstance(check, VerifyBridgeExploitMonitorCheck):
        return await run_bridge_exploit_monitor_check(client, check)
    if isinstance(check, VerifyRugPullRiskCheck):
        return await run_rug_pull_risk_check(client, check)

    return {
        "type": "unknown",
        "ok": False,
        "transient": False,
        "error": "Unsupported check type",
    }


# ----------------------
# LLM decision logic
# ----------------------
async def llm_decide_openrouter(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

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
        "- If all checks passed, prefer verdict='proceed'\n"
    )

    user_obj = {
        "goal": payload.get("goal"),
        "constraints": payload.get("constraints", {}),
        "allow_decide_on_partial": payload.get("allow_decide_on_partial", False),
        "context": payload.get("context", {}),
        "evidence": payload.get("evidence", []),
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": payload.get("referer", "http://localhost"),
        "X-Title": "DecisionVerificationAgent",
    }

    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj)},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_LLM_TOKENS,
    }

    async with httpx.AsyncClient(verify=certifi.where()) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=body,
            timeout=45,
        )

        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:1200]}")

        data = resp.json()

    content = data["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start:end + 1])
        raise


# ----------------------
# Core decision logic
# ----------------------
async def process_decision(req: DecideRequest, max_checks_allowed: int) -> DecideResponse:
    if len(req.checks) > max_checks_allowed:
        raise HTTPException(status_code=400, detail=f"Too many checks (max {max_checks_allowed})")

    evidence: List[Dict[str, Any]] = []
    failure_modes: List[str] = []

    async with httpx.AsyncClient(verify=certifi.where()) as client:
        for c in req.checks:
            evidence.append(await run_check(client, c))

    any_failed = any(e.get("ok") is False for e in evidence)
    any_transient_fail = any((e.get("ok") is False) and (e.get("transient") is True) for e in evidence)
    any_hard_fail = any((e.get("ok") is False) and (e.get("transient") is False) for e in evidence)

    if any_failed:
        failure_modes.append("one_or_more_checks_failed")

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


async def run_guard_payload(payload: GuardRequest, max_checks_allowed: int) -> DecideResponse:
    checks = build_guard_checks(payload)

    req = DecideRequest(
        goal=f"Verify safety before executing action: {payload.action}",
        constraints={
            "strict_mode": payload.strict_mode,
            "amount_usd": payload.amount_usd,
            "source": "guard",
        },
        checks=checks,
        allow_decide_on_partial=not payload.strict_mode,
        context=payload.context,
    )

    return await process_decision(req, max_checks_allowed)


async def run_stablecoin_intelligence(payload: StablecoinIntelligenceRequest, max_checks_allowed: int):
    checks: List[CheckType] = [
        VerifyStablecoinDepegCheck(
            type="stablecoin_depeg",
            asset_id=asset_id,
            quote=payload.quote,
            warn_deviation_pct=payload.warn_deviation_pct,
            block_deviation_pct=payload.block_deviation_pct,
        )
        for asset_id in payload.asset_ids
    ]

    req = DecideRequest(
        goal="Analyze stablecoin health and peg deviations",
        checks=checks[:max_checks_allowed],
        allow_decide_on_partial=True,
        context={"source": "stablecoin_intelligence"},
    )
    result = await process_decision(req, max_checks_allowed)
    return {
        "ok": True,
        "summary": result,
        "assets": payload.asset_ids,
    }


async def run_bridge_risk(payload: BridgeRiskRequest, max_checks_allowed: int):
    checks: List[CheckType] = [
        VerifyBridgeExploitMonitorCheck(
            type="bridge_exploit_monitor",
            status_url=url,
            expect_status=200,
        )
        for url in payload.bridge_status_urls
    ]

    req = DecideRequest(
        goal="Analyze bridge risk and operational status",
        checks=checks[:max_checks_allowed],
        allow_decide_on_partial=not payload.strict_mode,
        context={"source": "bridge_risk"},
    )
    result = await process_decision(req, max_checks_allowed)
    return {
        "ok": True,
        "summary": result,
        "bridges_checked": [str(x) for x in payload.bridge_status_urls[:max_checks_allowed]],
    }


async def run_yield_routing(payload: YieldRoutingRequest, max_checks_allowed: int):
    eligible: List[Dict[str, Any]] = []

    for option in payload.options:
        if option.risk_score > payload.max_risk_score:
            continue
        if option.liquidity_usd is not None and option.liquidity_usd < payload.min_liquidity_usd:
            continue

        score = option.apy - (option.risk_score / 100.0)
        if payload.prefer_chain and option.chain == payload.prefer_chain:
            score += 0.25

        eligible.append(
            {
                "name": option.name,
                "protocol": option.protocol,
                "chain": option.chain,
                "apy": option.apy,
                "risk_score": option.risk_score,
                "liquidity_usd": option.liquidity_usd,
                "score": round(score, 4),
            }
        )

    eligible.sort(key=lambda x: x["score"], reverse=True)

    return {
        "ok": True,
        "recommended": eligible[:max_checks_allowed],
        "count_considered": len(payload.options),
        "count_eligible": len(eligible),
    }


# ----------------------
# Routes
# ----------------------
@app.get("/")
async def root():
    return {
        "name": APP_NAME,
        "status": "running",
        "docs": "/v1/capabilities",
    }


@app.get("/health")
async def health():
    return {"ok": True, "name": APP_NAME}


@app.get("/v1/capabilities")
async def capabilities():
    return {
        "name": APP_NAME,
        "description": "Verify external conditions before an AI agent executes actions.",
        "checks_supported": [
            "http",
            "rpc",
            "price",
            "tx",
            "dex_price",
            "stablecoin_depeg",
            "bridge_exploit_monitor",
            "rug_pull_risk",
        ],
        "guard_actions_supported": ["swap", "transfer", "bridge", "yield_deposit", "generic"],
        "max_checks_private": MAX_CHECKS,
        "max_checks_acp": ACP_MAX_CHECKS,
        "decision_types": ["proceed", "wait", "block"],
        "base_url": PUBLIC_BASE_URL,
        "endpoints": {
            "private_decision": "/v1/decide",
            "acp_decision": "/v1/acp/decide",
            "private_guard": "/v1/guard",
            "acp_guard": "/v1/acp/guard",
            "api_key_signup": "/v1/api-keys/signup",
            "agents": "/v1/agents",
            "health": "/health",
        },
    }


@app.get("/v1/agents", response_model=List[AgentInfo])
async def list_agents():
    return AGENTS


@app.post("/v1/api-keys/signup", response_model=ApiKeySignupResponse)
async def api_key_signup(payload: ApiKeySignupRequest):
    issued_key = None
    if ISSUE_SIGNUP_KEYS:
        issued_key = f"eglin_{secrets.token_urlsafe(24)}"

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO api_key_signups
            (email, company, full_name, use_case, website, requested_plan, issued_key)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.email,
                payload.company,
                payload.full_name,
                payload.use_case,
                str(payload.website) if payload.website else None,
                payload.requested_plan,
                issued_key,
            ),
        )
        if issued_key:
            conn.execute(
                """
                INSERT INTO issued_api_keys (email, api_key, label, is_active)
                VALUES (?, ?, ?, 1)
                """,
                (payload.email, issued_key, payload.requested_plan or "signup"),
            )
        conn.commit()

    return ApiKeySignupResponse(
        ok=True,
        message=(
            "Signup received. Use the issued key for private endpoints."
            if issued_key
            else "Signup received. Eglin Labs will follow up with access."
        ),
        api_key=issued_key,
        docs_url=DOCS_URL,
        base_url=PUBLIC_BASE_URL,
    )


# Core endpoints
@app.post("/v1/decide", response_model=DecideResponse)
async def decide(req: DecideRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    return await process_decision(req, MAX_CHECKS)


@app.post("/v1/acp/decide", response_model=DecideResponse)
async def acp_decide(req: DecideRequest):
    return await process_decision(req, ACP_MAX_CHECKS)


@app.post("/v1/guard", response_model=DecideResponse)
async def universal_guard(payload: GuardRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    return await run_guard_payload(payload, MAX_CHECKS)


@app.post("/v1/acp/guard", response_model=DecideResponse)
async def acp_universal_guard(payload: GuardRequest):
    return await run_guard_payload(payload, ACP_MAX_CHECKS)


# Website-listed product aliases
@app.post("/v1/agents/decision-verification/decide", response_model=DecideResponse)
async def decision_verification_agent(req: DecideRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    return await process_decision(req, MAX_CHECKS)


@app.post("/v1/agents/execution-guard/execute", response_model=DecideResponse)
async def execution_guard_agent(payload: GuardRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    return await run_guard_payload(payload, MAX_CHECKS)


@app.post("/v1/agents/defi-risk/verify", response_model=DecideResponse)
async def defi_risk_verify(req: DecideRequest):
    allowed = {"dex_price", "stablecoin_depeg", "bridge_exploit_monitor", "rug_pull_risk"}
    for check in req.checks:
        if getattr(check, "type", None) not in allowed:
            raise HTTPException(
                status_code=400,
                detail="defi-risk endpoint only accepts dex_price, stablecoin_depeg, bridge_exploit_monitor, and rug_pull_risk checks",
            )
    return await process_decision(req, ACP_MAX_CHECKS)


@app.post("/v1/agents/stablecoin-intelligence/analyze")
async def stablecoin_intelligence_agent(payload: StablecoinIntelligenceRequest):
    return await run_stablecoin_intelligence(payload, ACP_MAX_CHECKS)


@app.post("/v1/agents/bridge-risk/analyze")
async def bridge_risk_agent(payload: BridgeRiskRequest):
    return await run_bridge_risk(payload, ACP_MAX_CHECKS)


@app.post("/v1/agents/yield-routing/recommend")
async def yield_routing_agent(payload: YieldRoutingRequest):
    return await run_yield_routing(payload, ACP_MAX_CHECKS)