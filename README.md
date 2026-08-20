# VerifyProceed — Guard API

> Pre-execution safety checks for AI DeFi agents. Binary verdict in under 300ms.

[![Live](https://img.shields.io/badge/status-live-brightgreen?style=flat-square)](https://decision-verification-agent.onrender.com/health) [![Base Mainnet](https://img.shields.io/badge/chain-Base%20Mainnet-0052ff?style=flat-square)](https://base.org) [![Solana](https://img.shields.io/badge/chain-Solana%20coming-9945FF?style=flat-square)](https://verifyproceed.com) [![x402](https://img.shields.io/badge/payments-x402%20USDC-00c896?style=flat-square)](https://x402.org) [![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://python.org) [![ACP](https://img.shields.io/badge/Coinbase-ACP%20compatible-0052ff?style=flat-square)](https://www.coinbase.com/developer-platform/products/agentkit)

---

## The problem

Every major AI agent framework — LangChain, ElizaOS, Coinbase AgentKit — lets agents execute on-chain with zero pre-execution safety checks.

A bridge exploit goes live. The agent keeps bridging.
A stablecoin depegs. The agent keeps swapping.
An RPC returns corrupted data. The agent acts on it.

**$2.8B+ was lost to DeFi exploits in 2024. The data existed. Nobody was checking it.**

---

## The solution

One API call before any on-chain action. Binary answer. Under 300ms.

```
curl -X POST https://decision-verification-agent.onrender.com/v1/acp/guard \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"action":"bridge","chain":"base","amount_usd":50000}'
```

```
{
  "verdict": "proceed",
  "confidence": 0.95,
  "risk": "low",
  "expires_in": 300,
  "decision": {
    "action": "execute",
    "reason": "All safety checks passed."
  },
  "evidence": [...],
  "failure_modes": []
}
```

If `verdict` is `block` — the agent stops. No human needed. No post-mortem needed.

---

## Live on Virtuals ACP as a neutral Evaluator

VerifyProceed also runs as a **third-party Evaluator agent** on Virtuals Protocol's Agent Commerce Protocol — the neutral referee that approves or rejects other agents' escrowed job deliverables before payment releases, using this same verification engine.

Both directions confirmed on Base mainnet, with the client, provider, and evaluator as three separate wallets throughout — genuine third-party evaluation, not self-approval:

- **Reject / refund** — verified across multiple jobs, escrow correctly returned to the client each time.
- **Complete / release** — verified with a real requirement, a real deliverable, a genuine passing verdict, and a confirmed single on-chain payout to the provider.

Runs on `@virtuals-protocol/acp-node-v2` with a non-custodial Privy-managed signer — built to run unattended on server infrastructure, not just as a local test harness.

Verify directly: [VerifyProceed's agent wallet on BaseScan](https://basescan.org/address/0x5a37a8ae00c884a864253dde3fe11f68364fcdf1)

---

## Quick start

### 1. Get a free API key

```
https://verifyproceed.com/get-api-key
```

100 calls/month. No card required.

### 2. Test immediately — no key needed

```
# Returns HTTP 402 (expected) — proves the API is live
curl -X POST https://decision-verification-agent.onrender.com/v1/acp/guard \
  -H "Content-Type: application/json" \
  -d '{"action":"generic","chain":"base"}'
```

### 3. With your key

```
curl -X POST https://decision-verification-agent.onrender.com/v1/acp/guard \
  -H "Authorization: Bearer vp_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"action":"generic","chain":"base"}'
```

---

## Code examples

### Python

```python
# pip install requests
import requests

response = requests.post(
    "https://decision-verification-agent.onrender.com/v1/acp/guard",
    headers={
        "Authorization": "Bearer vp_your_key_here",
        "Content-Type": "application/json",
    },
    json={
        "action":     "bridge",
        "chain":      "base",
        "amount_usd": 50000,
    },
)

result = response.json()

if result["verdict"] == "proceed":
    execute_bridge()
else:
    print("Blocked:", result["decision"]["reason"])
```

### TypeScript

```typescript
const response = await fetch(
  "https://decision-verification-agent.onrender.com/v1/acp/guard",
  {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.VERIFYPROCEED_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      action:     "swap",
      chain:      "base",
      amount_usd: 10000,
    }),
  }
);

const { verdict, decision } = await response.json();

if (verdict !== "proceed") {
  console.log("Blocked:", decision.reason);
  return;
}

// safe to execute
```

### ElizaOS plugin

```typescript
import type { Action, IAgentRuntime, Memory } from "@elizaos/core";

export const guardAction: Action = {
  name: "GUARD_CHECK",
  description: "Run VerifyProceed pre-execution safety check before any on-chain action",
  async handler(runtime: IAgentRuntime, message: Memory) {
    const res = await fetch(
      "https://decision-verification-agent.onrender.com/v1/acp/guard",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.VERIFYPROCEED_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action:     message.content.action ?? "generic",
          chain:      "base",
          amount_usd: message.content.amount_usd ?? 0,
        }),
      }
    );

    const verdict = await res.json();

    if (verdict.verdict !== "proceed") {
      return { text: `Action blocked: ${verdict.decision.reason}` };
    }

    return { text: "Guard check passed — proceeding." };
  },
};
```

### LangChain tool

```python
from langchain.tools import tool
import requests

@tool
def guard_check(action: str, chain: str = "base", amount_usd: float = 0) -> str:
    """Run a pre-execution safety check before any DeFi action."""
    response = requests.post(
        "https://decision-verification-agent.onrender.com/v1/acp/guard",
        headers={"Authorization": f"Bearer {VERIFYPROCEED_API_KEY}"},
        json={"action": action, "chain": chain, "amount_usd": amount_usd},
    )
    result = response.json()
    return f"verdict:{result['verdict']} confidence:{result['confidence']}"
```

---

## Endpoints

**Base URL:** `https://decision-verification-agent.onrender.com`

| Method | Path                           | Auth           | Cost      | Description                |
| ------ | ------------------------------ | -------------- | --------- | -------------------------- |
| GET    | `/health`                      | None           | Free      | Liveness check             |
| GET    | `/v1/capabilities`             | None           | Free      | Capability manifest        |
| GET    | `/v1/agents`                   | None           | Free      | List available agents      |
| POST   | `/v1/acp/guard`                | Bearer or x402 | $0.01     | ACP pre-execution guard    |
| POST   | `/v1/acp/decide`               | Bearer or x402 | $0.01     | ACP policy-driven decision |
| POST   | `/functions/v1/guard`          | Bearer         | Free tier | Key-authenticated guard    |
| POST   | `/functions/v1/api-key-signup` | None           | Free      | Get API key                |

---

## Guard request schema

```json
{
  "action":               "swap | transfer | bridge | yield_deposit | approve | mint | stake | add_liquidity | contract_call | generic",
  "chain":                "base | ethereum | arbitrum | optimism | polygon",
  "pair_address":         "0x... (optional — for DEX checks)",
  "stablecoin_asset_id":  "usd-coin (optional — for depeg checks)",
  "rpc_url":              "https://... (optional — overrides default RPC)",
  "bridge_status_url":    "https://... (optional — for bridge exploit check)",
  "amount_usd":           50000,
  "strict_mode":          true
}
```

---

## What we check

| Check                  | What it detects                            |
| ---------------------- | ------------------------------------------ |
| RPC health             | Corrupted or lagging RPC node data         |
| Stablecoin depeg       | USDC / USDT / DAI deviation from $1.00 peg |
| Bridge exploit monitor | Live exploits and active incidents         |
| DEX price integrity    | Manipulation signals and liquidity drain   |
| Rug pull risk          | Pair age, FDV ratio, liquidity depth       |
| Approval risk          | Unlimited/unusual token approvals          |
| Contract verification  | Unverified or suspicious contract bytecode |

All checks run in parallel. Total latency: **<300ms p99**.

---

## Verdict schema

```json
{
  "verdict":      "proceed | wait | block",
  "confidence":   0.95,
  "risk":         "low | medium | high",
  "expires_in":   300,
  "decision": {
    "action":     "execute | retry | halt",
    "reason":     "All safety checks passed.",
    "constraints": {}
  },
  "evidence":     [...],
  "failure_modes": []
}
```

| Verdict   | Meaning                                                                 |
| --------- | ----------------------------------------------------------------------- |
| `proceed` | All checks passed — safe to execute                                     |
| `wait`    | Transient failure (timeout, rate limit) — retry in `expires_in` seconds |
| `block`   | Hard failure detected — do not execute                                  |

---

## Payments

### Free tier

Get an API key at [verifyproceed.com/get-api-key](https://verifyproceed.com/get-api-key).
**100 calls/month. No card. No expiry.**

### x402 — pay per call

Agents pay **$0.01 USDC on Base** per call, autonomously, via the [x402 protocol](https://x402.org).
No accounts. No billing portals. Machine-native payments for machine-native infrastructure.

```
POST /v1/acp/guard  (no key)
← HTTP 402 + payment challenge
→ Agent pays $0.01 USDC on Base
→ Retry with X-PAYMENT header
← HTTP 200 + verdict
```

---

## Compatible with

| Framework        | Integration                          |
| ---------------- | ------------------------------------ |
| Virtuals ACP     | Native — Evaluator agent + `/v1/acp/*` endpoints |
| Coinbase ACP     | Native — ACP endpoints (`/v1/acp/*`) |
| ElizaOS / ai16z  | Plugin (see example above)           |
| LangChain        | Tool (see example above)             |
| Solana Agent Kit | HTTP call before any action          |
| Any HTTP client  | curl, requests, fetch, axios         |

---

## Self-hosting

### Requirements

- Python 3.11+
- [OpenRouter](https://openrouter.ai) API key (for LLM verdicts)
- Optional: [CoinGecko Demo API key](https://coingecko.com/en/api) (higher rate limits)

### Run locally

```
git clone https://github.com/verifyproceed/verifyproceed.git
cd verifyproceed

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your keys

uvicorn app:app --reload --port 8000
```

### Environment variables

| Variable               | Required    | Description                        |
| ----------------------- | ----------- | ----------------------------------- |
| `OPENROUTER_API_KEY`   | Yes         | LLM verdict generation             |
| `SUPABASE_URL`         | Yes         | API key validation                 |
| `SUPABASE_SERVICE_KEY` | Yes         | Supabase service role key          |
| `COINGECKO_API_KEY`    | Recommended | Higher rate limits on price checks |
| `PAYMENT_WALLET`       | Yes         | USDC payment recipient address     |
| `OPENROUTER_MODEL`     | No          | Default: `openai/gpt-4o-mini`      |
| `API_KEY`              | No          | Global key for private endpoints   |

### Docker

```
docker build -t verifyproceed-guard .
docker run -p 8000:8000 --env-file .env verifyproceed-guard
```

---

## API reference and tools

| Resource               | Link                                                                                    |
| ---------------------- | --------------------------------------------------------------------------------------- |
| Website                | [verifyproceed.com](https://verifyproceed.com)                                          |
| Full documentation     | [verifyproceed.com/docs](https://verifyproceed.com/docs)                                |
| Interactive playground | [verifyproceed.com/playground](https://verifyproceed.com/playground)                    |
| Marketplace / ACP kit  | [verifyproceed.com/marketplace-kit](https://verifyproceed.com/marketplace-kit)          |
| Get API key            | [verifyproceed.com/get-api-key](https://verifyproceed.com/get-api-key)                  |

*(Confirm exact paths match your live site's routing before publishing — inferred from the site's page structure, not independently verified.)*

---

## Support

- **Website:** [verifyproceed.com](https://verifyproceed.com)
- **X / Twitter:** *(add current handle)*
- **Email:** *(add current support email)*

---

## License

MIT
