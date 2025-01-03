"""
Main script to run the semantic entropy evaluation on XSum dataset
"""

import logging
import torch
from tqdm import tqdm
import numpy as np
from model import load_model_and_tokenizer, EntailmentDeberta
from data import get_dataset
from scores import (
    context_entails_response,
    get_semantic_ids,
    predictive_entropy,
    predictive_entropy_rao,
    cluster_assignment_entropy,
)

logging.basicConfig(level=logging.INFO)


def generate_summaries(model, tokenizer, text, num_samples=10):
    """Generate multiple summaries for a given text"""
    inputs = tokenizer(
        f"Summarize this text in one sentence: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(model.device)

    summaries = []
    log_probs = []

    for _ in range(num_samples):
        outputs = model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        summary = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        summaries.append(summary)

        # Calculate log probabilities
        scores = torch.stack(outputs.scores, dim=1)
        log_prob = torch.sum(
            torch.log_softmax(scores[0], dim=-1).gather(
                -1, outputs.sequences[0, 1:].unsqueeze(-1)
            )
        ).item()
        log_probs.append(log_prob)

    return summaries, log_probs


def evaluate_document(
    document, reference, model, tokenizer, entailment_model, num_samples=10
):
    """Evaluate semantic entropy metrics for a single document"""
    # Generate multiple summaries
    summaries, log_probs = generate_summaries(model, tokenizer, document, num_samples)

    # Get semantic IDs for generated summaries
    semantic_ids = get_semantic_ids(summaries, entailment_model)

    # Calculate metrics
    metrics = {
        "predictive_entropy": predictive_entropy(log_probs),
        "rao_entropy": predictive_entropy_rao(log_probs),
        "cluster_entropy": cluster_assignment_entropy(semantic_ids),
        "context_entailment": context_entails_response(
            document, summaries, entailment_model
        ),
        "reference_alignment": entailment_model.check_implication(
            reference, summaries[0]
        ),
        "generated_summaries": summaries,
    }

    return metrics


def main():
    """Main function to run the evaluation"""
    # Load models and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name)
    entailment_model = EntailmentDeberta()

    # Load dataset
    dataset = get_dataset("xsum")
    eval_dataset = dataset["validation"].select(
        range(100)
    )  # Evaluate on subset for efficiency

    # Initialize results storage
    results = []

    # Evaluate each document
    for idx, item in enumerate(tqdm(eval_dataset)):
        try:
            metrics = evaluate_document(
                item["document"], item["summary"], model, tokenizer, entailment_model
            )

            metrics["document_id"] = item["id"]
            results.append(metrics)

            # Log progress
            if (idx + 1) % 10 == 0:
                logging.info(f"Processed {idx + 1} documents")

        except Exception as e:
            logging.error(f"Error processing document {item['id']}: {e}")
            continue

    # Calculate and log aggregate metrics
    agg_metrics = {
        "avg_predictive_entropy": np.mean([r["predictive_entropy"] for r in results]),
        "avg_rao_entropy": np.mean([r["rao_entropy"] for r in results]),
        "avg_cluster_entropy": np.mean([r["cluster_entropy"] for r in results]),
        "avg_context_entailment": np.mean([r["context_entailment"] for r in results]),
        "avg_reference_alignment": np.mean([r["reference_alignment"] for r in results]),
    }

    logging.info("Aggregate Metrics:")
    for metric, value in agg_metrics.items():
        logging.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
