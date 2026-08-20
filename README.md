# VerifyProceed — Guard API

> Pre-execution safety checks for AI DeFi agents. Binary verdict in under 300ms.

[![Live](https://camo.githubusercontent.com/653a2d81930df2a18483cc2ac973f4c6004d152778456e3a17b8d4051ea1efea/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f7374617475732d6c6976652d627269676874677265656e3f7374796c653d666c61742d737175617265)](https://api.verifyproceed.com/health) [![Base Mainnet](https://camo.githubusercontent.com/10100a553b5f5e74aebaf8a1c9b562a06f9fc9dd980cac084cfa63cbc54898c4/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636861696e2d426173652532304d61696e6e65742d3030353266663f7374796c653d666c61742d737175617265)](https://base.org) [![Solana](https://camo.githubusercontent.com/391e87158f8e26ce4e646a4ea3b5d456947474ef6610033c4487209b08b4d9a3/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636861696e2d536f6c616e61253230636f6d696e672d3939343546463f7374796c653d666c61742d737175617265)](https://verifyproceed.com) [![x402](https://camo.githubusercontent.com/8fe78a7af8053931dc3e2798b9158baa085c75e3c8bc9e4df0945866330c5c00/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f7061796d656e74732d78343032253230555344432d3030633839363f7374796c653d666c61742d737175617265)](https://x402.org) [![Python](https://camo.githubusercontent.com/3fe3e79b0f8fceaa55a511c82c259fb2e61db1733967582543ea1a15c10752ff/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d332e31312b2d626c75653f7374796c653d666c61742d737175617265)](https://python.org) [![ACP](https://camo.githubusercontent.com/1be88869d8e354d5cc6257953b7dd8f4ad82a6c0fc90ebda78aaaf194a59a2e8/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f696e626173652d414350253230636f6d70617469626c652d3030353266663f7374796c653d666c61742d737175617265)](https://www.coinbase.com/developer-platform/products/agentkit)

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
curl -X POST https://api.verifyproceed.com/v1/acp/guard \
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

## Quick start

### 1. Get a free API key

```
https://verifyproceed.com/get-api-key
```

100 calls/month. No card required.

### 2. Test immediately — no key needed

```
# Returns HTTP 402 (expected) — proves the API is live
curl -X POST https://api.verifyproceed.com/v1/acp/guard \
  -H "Content-Type: application/json" \
  -d '{"action":"generic","chain":"base"}'
```

### 3. With your key

```
curl -X POST https://api.verifyproceed.com/v1/acp/guard \
  -H "Authorization: Bearer vp_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"action":"generic","chain":"base"}'
```

---

## Code examples

### Python

```
# pip install requests
import requests

response = requests.post(
    "https://api.verifyproceed.com/v1/acp/guard",
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

```
const response = await fetch(
  "https://api.verifyproceed.com/v1/acp/guard",
  {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.VP_API_KEY}`,
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

```
import type { Action, IAgentRuntime, Memory } from "@elizaos/core";

export const guardAction: Action = {
  name: "GUARD_CHECK",
  description: "Run VerifyProceed pre-execution safety check before any on-chain action",
  async handler(runtime: IAgentRuntime, message: Memory) {
    const res = await fetch(
      "https://api.verifyproceed.com/v1/acp/guard",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${process.env.VP_API_KEY}`,
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

```
from langchain.tools import tool
import requests

@tool
def guard_check(action: str, chain: str = "base", amount_usd: float = 0) -> str:
    """Run a pre-execution safety check before any DeFi action."""
    response = requests.post(
        "https://api.verifyproceed.com/v1/acp/guard",
        headers={"Authorization": f"Bearer {VP_API_KEY}"},
        json={"action": action, "chain": chain, "amount_usd": amount_usd},
    )
    result = response.json()
    return f"verdict:{result['verdict']} confidence:{result['confidence']}"
```

---

## Endpoints

**Base URL:** `https://api.verifyproceed.com`
**Private URL:** `https://dupfyqqbvkrmzjexwukd.supabase.co`

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

```
{
  "action":               "swap | transfer | bridge | yield_deposit | generic",
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

All checks run in parallel. Total latency: **<300ms p99**.

---

## Verdict schema

```
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
| ---------------------- | ----------- | ----------------------------------- |
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
| Full documentation     | [verifyproceed.com/docs](https://verifyproceed.com/docs)                                |
| Interactive playground | [verifyproceed.com/playground](https://verifyproceed.com/playground)                    |
| OpenAPI spec           | [verifyproceed.com/openapi.json](https://verifyproceed.com/openapi.json)                |
| Postman collection     | [verifyproceed.com/postman_collection.json](https://verifyproceed.com/postman_collection.json) |
| Get API key            | [verifyproceed.com/get-api-key](https://verifyproceed.com/get-api-key)                  |

---

## Support

- **Email:** <api@verifyproceed.com>
- **Website:** [verifyproceed.com](https://verifyproceed.com)
- **X / Twitter:** [@proceedapi](https://x.com/proceedapi)

---

## License

MIT
