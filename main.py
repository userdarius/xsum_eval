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


def generate_summaries(model, tokenizer, text, doc_id, num_samples=10):
    """Generate multiple summaries for a given text"""
    prompt = (
        "Generate a single-sentence summary using ONLY facts explicitly stated in this text. Only generate the summary, do not include any other text. "
        "Mention only events directly described: "
        f"{text}"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(model.device)

    summaries = []
    log_probs = []

    for _ in range(num_samples):
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=40,  # Further reduced
            min_new_tokens=10,
            temperature=0.4,  # Reduced temperature for more focused outputs
            top_p=0.85,  # Slightly reduced for more conservative sampling
            no_repeat_ngram_size=3,
            length_penalty=0.8,  # Increased penalty for longer sequences
            num_beams=1,  # Use greedy decoding
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode summary and remove the prompt from the output
        # Decode and clean up the summary
        full_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # Remove prompt and clean up common artifacts
        summary = (
            full_output.replace(prompt, "")
            .replace("Source: BBC News", "")
            .replace("In one sentence,", "")
            .replace("Here is a summary:", "")
            .replace("Source:", "")
            .split("(")[0]  # Remove any parenthetical metadata
            .strip()
        )

        # Only take the first sentence if multiple were generated
        if "." in summary:
            summary = summary.split(".")[0].strip() + "."

        summaries.append(summary)

        logging.info(f"Generated summary for document {doc_id}: {summary}")

        if (
            len(summary) > 10  # Must be longer than 10 chars
            and not summary.startswith("http")  # No URLs
            and not any(
                x in summary.lower() for x in ["source:", "note:", "see more"]
            )  # No metadata
            and not summary.isspace()  # Not just whitespace
            and summary != "."  # Not just a period
        ):
            summaries.append(summary)

            # Calculate log probabilities
            scores = torch.stack(outputs.scores, dim=1)
            seq_length = outputs.sequences[0, 1:].size(0)

            if seq_length > scores.size(1):
                seq_length = scores.size(1)
                sequence = outputs.sequences[0, 1 : seq_length + 1]
            else:
                sequence = outputs.sequences[0, 1 : seq_length + 1]

            log_softmax_scores = torch.log_softmax(scores[0, :seq_length], dim=-1)
            token_log_probs = log_softmax_scores.gather(-1, sequence.unsqueeze(-1))
            log_prob = torch.sum(token_log_probs).item()

            log_probs.append(log_prob)

        # If we don't have enough valid summaries, continue generating
        if len(summaries) < num_samples:
            continue

    return summaries, log_probs


def evaluate_document(
    document, reference, doc_id, model, tokenizer, entailment_model, num_samples=10
):
    """Evaluate semantic entropy metrics for a single document"""
    # Generate multiple summaries
    summaries, log_probs = generate_summaries(
        model, tokenizer, document, doc_id, num_samples
    )

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
        range(1)
    )  # Evaluate on subset for efficiency

    # Initialize results storage
    results = []

    # Evaluate each document
    for idx, item in enumerate(tqdm(eval_dataset)):
        try:
            metrics = evaluate_document(
                item["document"],
                item["summary"],
                item["id"],
                model,
                tokenizer,
                entailment_model,
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
