# Assignment 1 - Social Media Monitor

## Project Description

This project monitors the BlueSky firehose for potential poetry posts using a cost-aware multi-stage pipeline.
The system first applies cheap symbolic and keyword filters, then only sends likely candidates to an LLM classifier.
An evaluation harness measures quality (precision/recall/F1) and estimated API cost on a labeled gold dataset.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Set either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

## How To Run

Run online streaming monitor (pipeline):

- OpenAI: `python firehose.py --mode stream --provider openai --model gpt-4o-mini --keyword-profile balanced --prompt-variant base`
- Anthropic: `python firehose.py --mode stream --provider anthropic --model claude-haiku-4-5-20251001 --keyword-profile balanced --prompt-variant base`

Run eval harness (single configuration):

- `python hw1/eval.py --gold hw1/data/gold_dataset.jsonl --provider anthropic --keyword-profile balanced --variant base --save hw1/eval_anthropic_base.json`

Run eval comparison across prompting styles (base/fewshot/cot):

- `python hw1/eval.py --gold hw1/data/gold_dataset.jsonl --provider anthropic --keyword-profile balanced --compare --save-dir hw1`

Run tuning across keyword profiles and prompt variants:

- `python hw1/eval.py --gold hw1/data/gold_dataset.jsonl --provider anthropic --tune --cost-cap 0.02 --save-dir hw1`

## Demo Video

Add your unlisted YouTube link here:

- TODO: [Demo link](https://youtube.com/)

## Eval Results

After running eval, fill this table with your measured values:

| Configuration | Precision | Recall |    F1 | LLM Calls | Total Cost (USD) |
| ------------- | --------: | -----: | ----: | --------: | ---------------: |
| base          |     1.000 |  0.160 | 0.276 |        11 |         0.005043 |
| fewshot       |     1.000 |  0.160 | 0.276 |        11 |         0.005514 |
| cot           |     1.000 |  0.160 | 0.276 |        11 |         0.005432 |

Tuning summary (cost cap 0.02 USD): best config was `keyword-profile=strict` + `prompt-variant=base`
with precision=1.000, recall=0.160, f1=0.276, llm_calls=6, total_cost_usd=0.002810.

## Pipeline Stages

1. Symbolic filter: keep posts with plausible text length.
2. Keyword filter: keep posts containing poetry-related terms.
3. LLM classifier: classify candidates as poem/non-poem with confidence and explanation.

These stages reduce cost by avoiding LLM calls for obviously irrelevant posts.

## AI Transcripts

Store exported assistant transcripts in the `transcripts/` folder and reference them here.

## Notes

- The provided `hw1/data/gold_dataset.jsonl` is a starter labeled dataset scaffold (50 examples).
- You should revise and expand it with task-specific examples for stronger evaluation quality.
