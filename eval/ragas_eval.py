"""
RAGAS Evaluation Script
────────────────────────
Measures 4 metrics across a test set:
  - Faithfulness       (is answer grounded in context?)
  - Answer Relevancy   (does answer address the question?)
  - Context Recall     (did retrieval get the right chunks?)
  - Context Precision  (were fetched chunks useful?)

Usage:
    python eval/ragas_eval.py --questions eval/test_questions.json --output eval/results.json

Test questions JSON format:
[
  {
    "question": "What is the main contribution of the Attention Is All You Need paper?",
    "ground_truth": "The paper introduces the Transformer architecture..."
  },
  ...
]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_test_questions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_pipeline_for_eval(question: str) -> dict:
    """Run the RAG pipeline and return structured output for RAGAS."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.graph import run_query

    result = run_query(question)
    return result


def evaluate(test_path: str, output_path: str):
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError:
        print("Install ragas and datasets: pip install ragas datasets")
        return

    test_cases = load_test_questions(test_path)
    print(f"Evaluating {len(test_cases)} questions...\n")

    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['question'][:60]}...")
        try:
            result = run_pipeline_for_eval(tc["question"])
            rows["question"].append(tc["question"])
            rows["answer"].append(result["answer"])
            rows["contexts"].append([c.get("text", "") for c in result.get("approved_chunks", [])])
            rows["ground_truth"].append(tc.get("ground_truth", ""))
        except Exception as e:
            print(f"  ⚠ Error: {e}")

    dataset = Dataset.from_dict(rows)

    print("\nRunning RAGAS scoring...")
    scores = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(rows["question"]),
        "metrics": {
            "faithfulness": round(scores["faithfulness"], 4),
            "answer_relevancy": round(scores["answer_relevancy"], 4),
            "context_recall": round(scores["context_recall"], 4),
            "context_precision": round(scores["context_precision"], 4),
        },
    }

    print("\n" + "="*40)
    print("RAGAS Evaluation Results")
    print("="*40)
    for metric, val in results["metrics"].items():
        bar = "█" * int(val * 20)
        print(f"  {metric:<22} {val:.4f}  {bar}")
    print("="*40)

    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ─── Sample test questions (AI papers domain) ────────────────────────────────

SAMPLE_QUESTIONS = [
    {
        "question": "What is the transformer architecture and what problem does it solve?",
        "ground_truth": "The Transformer is a sequence-to-sequence model based entirely on attention mechanisms, eliminating recurrence and convolutions. It solves the parallelization limitations of RNNs and LSTMs while better capturing long-range dependencies.",
    },
    {
        "question": "How does RAG differ from fine-tuning for knowledge injection in LLMs?",
        "ground_truth": "RAG retrieves relevant documents at inference time and conditions generation on them, while fine-tuning bakes knowledge into model weights. RAG is more flexible and avoids catastrophic forgetting.",
    },
    {
        "question": "What are the main challenges in evaluating large language models?",
        "ground_truth": "Key challenges include benchmark contamination, lack of standardized evaluation frameworks, difficulty measuring reasoning vs memorization, and the gap between automated metrics and human judgment.",
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/test_questions.json")
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--generate-sample", action="store_true", help="Write sample questions file")
    args = parser.parse_args()

    if args.generate_sample:
        Path(args.questions).parent.mkdir(exist_ok=True)
        with open(args.questions, "w") as f:
            json.dump(SAMPLE_QUESTIONS, f, indent=2)
        print(f"Sample questions written to {args.questions}")
    else:
        evaluate(args.questions, args.output)