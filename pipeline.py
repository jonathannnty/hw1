import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import dotenv
from anthropic import AsyncAnthropic
import openai
import websockets

URI = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"

BASE_PROMPT = """You are a poetry detector for short social media posts.
Given a post, determine whether it is likely an original poem.

Respond with strict JSON:
{
  \"is_poem\": true/false,
  \"confidence\": 0.0-1.0,
  \"explanation\": \"brief reason (<50 words)\"
}

Post text:
%s
"""

FEWSHOT_PROMPT = """You are a poetry detector for short social media posts.
Use the examples to classify the final post.

Example 1:
Text: moonlight breaks / over empty train tracks / one lantern waits
Output: {\"is_poem\": true, \"confidence\": 0.97, \"explanation\": \"Line breaks and imagery suggest poetry.\"}

Example 2:
Text: New internship posted at ACME Labs for summer software interns. Apply here: example.com
Output: {\"is_poem\": false, \"confidence\": 0.99, \"explanation\": \"Job announcement format, not poetic writing.\"}

Now classify this post with strict JSON only:
%s
"""

COT_STYLE_PROMPT = """You are a poetry detector for short social media posts.
Think carefully about poetic features like line breaks, figurative language, rhythm, and imagery.
Then provide a concise explanation.

Respond with strict JSON:
{
  \"is_poem\": true/false,
  \"confidence\": 0.0-1.0,
  \"explanation\": \"brief reason (<50 words)\"
}

Post text:
%s
"""


@dataclass
class PipelineConfig:
    provider: str = "openai"  # openai|anthropic
    model: str = "gpt-4o-mini"
    min_chars: int = 12
    max_chars: int = 600
    keywords: Tuple[str, ...] = (
        "poem",
        "poetry",
        "haiku",
        "verse",
        "stanza",
        "sonnet",
        "rhyme",
        "ode",
        "line break",
    )
    heuristic_min_score: float = 2.5
    prompt_variant: str = "base"  # base|fewshot|cot


KEYWORD_PROFILES: Dict[str, Tuple[str, ...]] = {
    "strict": (
        "poem",
        "haiku",
        "sonnet",
        "stanza",
        "verse",
    ),
    "balanced": (
        "poem",
        "poetry",
        "haiku",
        "verse",
        "stanza",
        "sonnet",
        "rhyme",
        "ode",
        "line break",
    ),
    "recall": (
        "poem",
        "poetry",
        "haiku",
        "verse",
        "stanza",
        "sonnet",
        "rhyme",
        "ode",
        "line break",
        "metaphor",
        "imagery",
        "lyric",
        "free verse",
        "couplet",
    ),
}

HEURISTIC_THRESHOLDS: Dict[str, float] = {
    "strict": 3.0,
    "balanced": 2.5,
    "recall": 1.5,
}

POETRY_CUE_TERMS: Tuple[str, ...] = (
    "moon",
    "dawn",
    "night",
    "rain",
    "wind",
    "shadow",
    "star",
    "ocean",
    "river",
    "silence",
    "heart",
    "light",
    "dream",
    "whisper",
    "lantern",
    "feather",
    "sunset",
    "snow",
)

NON_POEM_CUE_TERMS: Tuple[str, ...] = (
    "apply",
    "internship",
    "meeting",
    "announcement",
    "discount",
    "status",
    "traffic",
    "submit",
    "buy",
    "selling",
    "roommate",
)


def default_model_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return "claude-haiku-4-5-20251001"
    return "gpt-4o-mini"


def get_keywords(profile: str) -> Tuple[str, ...]:
    return KEYWORD_PROFILES[profile]


def get_heuristic_threshold(profile: str) -> float:
    return HEURISTIC_THRESHOLDS[profile]


def _get_prompt(variant: str, text: str) -> str:
    if variant == "fewshot":
        return FEWSHOT_PROMPT % text
    if variant == "cot":
        return COT_STYLE_PROMPT % text
    return BASE_PROMPT % text


def _estimate_cost(provider: str, usage: Optional[Any]) -> float:
    if usage is None:
        return 0.0

    if provider == "anthropic":
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        in_price = float(os.getenv("ANTHROPIC_INPUT_COST_PER_1M", "1.00"))
        out_price = float(os.getenv("ANTHROPIC_OUTPUT_COST_PER_1M", "5.00"))
    else:
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        in_price = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0.15"))
        out_price = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "0.60"))

    return (in_tok / 1_000_000.0) * in_price + (out_tok / 1_000_000.0) * out_price


def _parse_json_output(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def create_llm_client(provider: str) -> Any:
    if provider == "anthropic":
        return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return openai.AsyncOpenAI()


def _extract_text(msgjson: Dict[str, Any]) -> Optional[str]:
    try:
        return msgjson["commit"]["record"]["text"]
    except (KeyError, TypeError):
        return None


def _passes_symbolic_filter(text: str, cfg: PipelineConfig) -> bool:
    n = len(text.strip())
    return cfg.min_chars <= n <= cfg.max_chars


def _passes_keyword_filter(text: str, cfg: PipelineConfig) -> bool:
    lower = text.lower()
    return any(k in lower for k in cfg.keywords)


def _poetry_heuristic_score(text: str) -> float:
    lower = text.lower()
    score = 0.0

    # Line separators are a strong cheap signal for short-form poetry.
    if "\n" in text:
        score += 2.0
    if "/" in text:
        score += 2.0

    if 12 <= len(text.strip()) <= 220:
        score += 0.5

    if any(cue in lower for cue in POETRY_CUE_TERMS):
        score += 1.0

    # Lightweight figurative-language cue.
    if " like " in lower or " as " in lower:
        score += 0.5

    # Penalize obvious non-poetry content patterns.
    if "http://" in lower or "https://" in lower:
        score -= 1.0
    if any(cue in lower for cue in NON_POEM_CUE_TERMS):
        score -= 1.0

    return score


def _passes_stage2_filter(text: str, cfg: PipelineConfig) -> bool:
    if _passes_keyword_filter(text, cfg):
        return True
    return _poetry_heuristic_score(text) >= cfg.heuristic_min_score


async def classify_post(
    client: Any,
    text: str,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    prompt = _get_prompt(cfg.prompt_variant, text)
    if cfg.provider == "anthropic":
        response = await client.messages.create(
            model=cfg.model,
            max_tokens=220,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        parsed = _parse_json_output(raw)
        parsed["cost_usd"] = _estimate_cost(cfg.provider, response.usage)
        return parsed

    response = await client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=140,
        temperature=0,
    )
    parsed = _parse_json_output(response.choices[0].message.content)
    parsed["cost_usd"] = _estimate_cost(cfg.provider, response.usage)
    return parsed


async def run_pipeline_on_text(
    client: Any,
    text: str,
    cfg: PipelineConfig,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "text": text,
        "stage_symbolic": False,
        "stage_keyword": False,
        "used_llm": False,
        "prediction": False,
        "confidence": 0.0,
        "explanation": "",
        "cost_usd": 0.0,
    }

    if not _passes_symbolic_filter(text, cfg):
        out["explanation"] = "Rejected by symbolic length filter"
        return out
    out["stage_symbolic"] = True

    if not _passes_stage2_filter(text, cfg):
        out["explanation"] = "Rejected by stage-2 keyword/heuristic filter"
        return out
    out["stage_keyword"] = True

    llm = await classify_post(client, text, cfg)
    out["used_llm"] = True
    out["prediction"] = bool(llm.get("is_poem", False))
    out["confidence"] = float(llm.get("confidence", 0.0))
    out["explanation"] = str(llm.get("explanation", ""))
    out["cost_usd"] = float(llm.get("cost_usd", 0.0))
    return out


async def listen_stream(cfg: PipelineConfig) -> None:
    dotenv.load_dotenv()
    client = create_llm_client(cfg.provider)

    seen = 0
    llm_calls = 0
    total_cost = 0.0

    async with websockets.connect(URI) as websocket:
        while True:
            try:
                message = await websocket.recv()
                msgjson = json.loads(message)
                text = _extract_text(msgjson)
                if not text:
                    continue

                result = await run_pipeline_on_text(client, text, cfg)
                seen += 1
                if result["used_llm"]:
                    llm_calls += 1
                    total_cost += result["cost_usd"]

                if result["prediction"]:
                    print("-" * 60)
                    print("MATCH")
                    print(text)
                    print(f"confidence={result['confidence']:.2f} cost=${result['cost_usd']:.6f}")
                    print(result["explanation"])

                if seen % 50 == 0:
                    print(
                        f"seen={seen} llm_calls={llm_calls} llm_rate={llm_calls/max(seen,1):.2%} cost=${total_cost:.4f}"
                    )
            except (json.JSONDecodeError, KeyError, TypeError, websockets.WebSocketException) as exc:
                print(f"stream error: {exc}")


async def batch_classify_file(
    in_jsonl: str,
    out_jsonl: str,
    cfg: PipelineConfig,
) -> None:
    dotenv.load_dotenv()
    client = create_llm_client(cfg.provider)

    with open(in_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            text = row.get("text", "")
            result = await run_pipeline_on_text(client, text, cfg)
            merged = {**row, **result}
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW1 multi-stage social monitor pipeline")
    p.add_argument("--mode", choices=["stream", "batch"], default="stream")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--model", default="")
    p.add_argument("--prompt-variant", choices=["base", "fewshot", "cot"], default="base")
    p.add_argument("--keyword-profile", choices=["strict", "balanced", "recall"], default="balanced")
    p.add_argument("--input", help="Input JSONL with a text field (batch mode)")
    p.add_argument("--output", help="Output JSONL (batch mode)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or default_model_for_provider(args.provider)
    cfg = PipelineConfig(
        provider=args.provider,
        model=model,
        keywords=get_keywords(args.keyword_profile),
        heuristic_min_score=get_heuristic_threshold(args.keyword_profile),
        prompt_variant=args.prompt_variant,
    )

    if args.mode == "stream":
        asyncio.run(listen_stream(cfg))
        return

    if not args.input or not args.output:
        raise ValueError("--input and --output are required in batch mode")
    asyncio.run(batch_classify_file(args.input, args.output, cfg))


if __name__ == "__main__":
    main()
