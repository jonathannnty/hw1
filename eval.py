import argparse
import asyncio
import json
from dataclasses import asdict
from typing import Dict, List

from pipeline import (
    PipelineConfig,
    create_llm_client,
    default_model_for_provider,
    get_heuristic_threshold,
    get_keywords,
    run_pipeline_on_text,
)
import dotenv


def _metrics(rows: List[Dict]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for r in rows:
        y = bool(r["label"])
        yhat = bool(r["prediction"])
        if y and yhat:
            tp += 1
        elif (not y) and yhat:
            fp += 1
        elif (not y) and (not yhat):
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


async def evaluate(gold_path: str, cfg: PipelineConfig, save_path: str = "") -> Dict:
    dotenv.load_dotenv()
    client = create_llm_client(cfg.provider)

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]

    out_rows = []
    for row in gold:
        text = row.get("text", "")
        res = await run_pipeline_on_text(client, text, cfg)
        out_rows.append({"label": bool(row.get("label", False)), **res})

    m = _metrics(out_rows)
    cost = sum(r.get("cost_usd", 0.0) for r in out_rows)
    symbolic_pass = sum(1 for r in out_rows if r.get("stage_symbolic"))
    keyword_pass = sum(1 for r in out_rows if r.get("stage_keyword"))
    llm_calls = sum(1 for r in out_rows if r.get("used_llm"))

    report = {
        "config": asdict(cfg),
        "n": len(out_rows),
        "stage_counts": {
            "symbolic_pass": symbolic_pass,
            "keyword_pass": keyword_pass,
            "llm_calls": llm_calls,
        },
        "metrics": m,
        "total_cost_usd": cost,
        "avg_cost_per_item_usd": (cost / len(out_rows)) if out_rows else 0.0,
    }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"report": report, "rows": out_rows}, f, indent=2)

    return report


async def compare(gold_path: str, model: str, save_dir: str = "") -> None:
    variants = ["base", "fewshot", "cot"]
    print("variant\tprecision\trecall\tf1\tllm_calls\ttotal_cost_usd")
    for v in variants:
        cfg = PipelineConfig(model=model, prompt_variant=v)
        save_path = f"{save_dir}/eval_{v}.json" if save_dir else ""
        report = await evaluate(gold_path, cfg, save_path=save_path)
        print(
            f"{v}\t{report['metrics']['precision']:.3f}\t{report['metrics']['recall']:.3f}\t"
            f"{report['metrics']['f1']:.3f}\t{report['stage_counts']['llm_calls']}\t{report['total_cost_usd']:.6f}"
        )


async def compare_with_profile(
    gold_path: str,
    provider: str,
    model: str,
    keyword_profile: str,
    save_dir: str = "",
) -> None:
    variants = ["base", "fewshot", "cot"]
    print("variant\tprecision\trecall\tf1\tllm_calls\ttotal_cost_usd")
    for v in variants:
        cfg = PipelineConfig(
            provider=provider,
            model=model,
            keywords=get_keywords(keyword_profile),
            heuristic_min_score=get_heuristic_threshold(keyword_profile),
            prompt_variant=v,
        )
        save_path = f"{save_dir}/eval_{provider}_{keyword_profile}_{v}.json" if save_dir else ""
        report = await evaluate(gold_path, cfg, save_path=save_path)
        print(
            f"{v}\t{report['metrics']['precision']:.3f}\t{report['metrics']['recall']:.3f}\t"
            f"{report['metrics']['f1']:.3f}\t{report['stage_counts']['llm_calls']}\t{report['total_cost_usd']:.6f}"
        )


async def tune(
    gold_path: str,
    provider: str,
    model: str,
    cost_cap: float,
    target_recall: float,
    save_dir: str = "",
) -> None:
    variants = ["base", "fewshot", "cot"]
    profiles = ["strict", "balanced", "recall"]
    rows = []

    print("profile\tvariant\tprecision\trecall\tf1\tllm_calls\ttotal_cost_usd")
    for p in profiles:
        for v in variants:
            cfg = PipelineConfig(
                provider=provider,
                model=model,
                keywords=get_keywords(p),
                heuristic_min_score=get_heuristic_threshold(p),
                prompt_variant=v,
            )
            save_path = f"{save_dir}/eval_{provider}_{p}_{v}.json" if save_dir else ""
            report = await evaluate(gold_path, cfg, save_path=save_path)
            m = report["metrics"]
            cost = report["total_cost_usd"]
            llm_calls = report["stage_counts"]["llm_calls"]
            print(
                f"{p}\t{v}\t{m['precision']:.3f}\t{m['recall']:.3f}\t{m['f1']:.3f}\t{llm_calls}\t{cost:.6f}"
            )
            rows.append({
                "profile": p,
                "variant": v,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "llm_calls": llm_calls,
                "total_cost_usd": cost,
            })

    eligible = [r for r in rows if r["total_cost_usd"] <= cost_cap]
    threshold_ok = [r for r in eligible if r["recall"] >= target_recall]

    if threshold_ok:
        # Minimize cost while maintaining target recall.
        best = sorted(
            threshold_ok,
            key=lambda r: (r["total_cost_usd"], -r["precision"], -r["f1"]),
        )[0]
    else:
        target = eligible if eligible else rows
        best = sorted(
            target,
            key=lambda r: (-r["recall"], -r["precision"], r["total_cost_usd"]),
        )[0]

    print("\nBest config for target recall under cost cap:")
    print(json.dumps({"cost_cap": cost_cap, "target_recall": target_recall, "best": best}, indent=2))

    if save_dir:
        with open(f"{save_dir}/tune_summary_{provider}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cost_cap": cost_cap,
                    "target_recall": target_recall,
                    "best": best,
                    "results": rows,
                },
                f,
                indent=2,
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW1 eval harness")
    p.add_argument("--gold", default="hw1/data/gold_dataset.jsonl")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--model", default="")
    p.add_argument("--variant", choices=["base", "fewshot", "cot"], default="base")
    p.add_argument("--keyword-profile", choices=["strict", "balanced", "recall"], default="balanced")
    p.add_argument("--compare", action="store_true", help="Run base/fewshot/cot comparison")
    p.add_argument("--tune", action="store_true", help="Run keyword-profile + prompt-variant tuning")
    p.add_argument("--cost-cap", type=float, default=0.02, help="Cost cap used by --tune")
    p.add_argument("--target-recall", type=float, default=0.5, help="Target recall used by --tune")
    p.add_argument("--save", default="", help="Save single-run report JSON")
    p.add_argument("--save-dir", default="", help="Save per-variant reports when --compare")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = args.model or default_model_for_provider(args.provider)

    if args.tune:
        asyncio.run(
            tune(
                args.gold,
                provider=args.provider,
                model=model,
                cost_cap=args.cost_cap,
                target_recall=args.target_recall,
                save_dir=args.save_dir,
            )
        )
        return

    if args.compare:
        asyncio.run(
            compare_with_profile(
                args.gold,
                provider=args.provider,
                model=model,
                keyword_profile=args.keyword_profile,
                save_dir=args.save_dir,
            )
        )
        return

    cfg = PipelineConfig(
        provider=args.provider,
        model=model,
        keywords=get_keywords(args.keyword_profile),
        heuristic_min_score=get_heuristic_threshold(args.keyword_profile),
        prompt_variant=args.variant,
    )
    report = asyncio.run(evaluate(args.gold, cfg, save_path=args.save))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
