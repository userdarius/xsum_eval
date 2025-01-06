"""
Main script to run the semantic entropy evaluation on XSum dataset with branching generation
"""

import sys
from datetime import datetime
import logging
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from model import (
    load_model_and_tokenizer,
    EntailmentDeberta,
    generate_branching_responses,
)
from data import get_dataset
from scores import (
    context_entails_response,
    get_semantic_ids,
    predictive_entropy,
    cluster_assignment_entropy,
)


class ResultsVisualizer:
    def __init__(self, results_list):
        """Initialize with a list of result dictionaries"""
        cleaned_results = [
            {
                k: v
                for k, v in r.items()
                if k != "generated_summaries" and not isinstance(v, (list, dict))
            }
            for r in results_list
        ]
        self.results = pd.DataFrame(cleaned_results)

    def plot_metric_distribution(self, metric_name, title=None):
        """Create a distribution plot for a specific metric"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.results[metric_name], kde=True)
        plt.title(title or f"Distribution of {metric_name}")
        plt.xlabel(metric_name)
        plt.ylabel("Count")
        plt.savefig(f"{metric_name}_distribution.png")
        plt.close()

    def plot_metric_correlations(self):
        """Create a correlation heatmap between different metrics"""
        plt.figure(figsize=(12, 10))
        numeric_cols = self.results.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.results[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Between Uncertainty Metrics")
        plt.tight_layout()
        plt.savefig("metric_correlations.png")
        plt.close()

    def plot_metrics_scatter(self, x_metric, y_metric):
        """Create a scatter plot comparing two metrics"""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.results, x=x_metric, y=y_metric)
        plt.title(f"{x_metric} vs {y_metric}")
        plt.xlabel(x_metric)
        plt.ylabel(y_metric)
        plt.savefig(f"{x_metric}_vs_{y_metric}_scatter.png")
        plt.close()

    def plot_metrics_over_documents(self):
        """Plot all metrics across documents"""
        metrics = [col for col in self.results.columns if col != "document_id"]
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            plt.plot(
                range(len(self.results)), self.results[metric], label=metric, marker="o"
            )
        plt.title("Metrics across Documents")
        plt.xlabel("Document Index")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("metrics_across_documents.png")
        plt.close()

    def generate_summary_statistics(self):
        """Generate and save summary statistics"""
        summary_stats = self.results.describe()
        summary_stats.to_csv("summary_statistics.csv")
        return summary_stats


def setup_logging():
    """Configure logging with detailed formatting and both file and console handlers"""
    log_filename = f"semantic_entropy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    console_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename


def generate_summaries(
    model, tokenizer, text, reference_summary, doc_id, num_branches=10
):
    """Generate multiple summaries using branching generation"""
    logging.info(f"Generating {num_branches} summaries for document ID: {doc_id}")
    logging.debug(f"Input text length: {len(text)} characters")

    prompt = "Write a simple one-sentence summary of this news article: " + text

    try:
        # Use branching generation method
        responses = generate_branching_responses(
            model,
            tokenizer,
            prompt,
            max_length=30,
            num_branches=num_branches,
        )

        # Sort responses by confidence score
        responses.sort(key=lambda x: x[1], reverse=True)

        summaries = []
        confidence_scores = []
        log_probs = []

        logging.info(f"Document: {text}")
        logging.info(f"Reference summary: {reference_summary}")

        for response, confidence_score, log_prob in responses:
            # Clean up the summary
            summary = response.replace(prompt, "").strip()

            # Remove "Source: BBC News" and similar variations
            source_patterns = ["Source: BBC News", "Source: BBC", "BBC News:", "BBC:"]
            for pattern in source_patterns:
                summary = summary.replace(pattern, "").strip()

            # Take first sentence and ensure it ends with a period
            summary = summary.split(".")[0].strip() + "."

            # Validate summary
            if len(summary) > 15 and summary != "." and not summary.startswith("http"):
                logging.info(f"Valid summary generated: {summary}")
                logging.debug(f"Confidence score: {confidence_score}")
                logging.debug(f"Log probability: {log_prob}")

                summaries.append(summary)
                confidence_scores.append(confidence_score)
                log_probs.append(log_prob)
            else:
                logging.warning(f"Summary failed validation: {summary}")

        logging.info(
            f"Successfully generated {len(summaries)}/{num_branches} valid summaries"
        )
        return summaries, confidence_scores, log_probs

    except Exception as e:
        logging.error(f"Error in generate_summaries: {str(e)}")
        return [], [], []


def evaluate_document(
    document, reference, doc_id, model, tokenizer, entailment_model, num_branches=10
):
    """Evaluate semantic entropy metrics for a single document"""
    logging.info(f"Starting evaluation for document ID: {doc_id}")
    logging.debug(
        f"Document length: {len(document)} chars, Reference length: {len(reference)} chars"
    )

    try:
        # Generate multiple summaries using branching
        summaries, confidence_scores, log_probs = generate_summaries(
            model, tokenizer, document, reference, doc_id, num_branches
        )

        if not summaries:
            logging.error(f"No valid summaries generated for document {doc_id}")
            return None

        # Calculate semantic IDs
        logging.info("Calculating semantic IDs")
        semantic_ids = get_semantic_ids(summaries, entailment_model)
        semantic_cluster_counts = np.bincount(semantic_ids)
        logging.info(f"Semantic IDs distribution: {semantic_cluster_counts}")

        context_entailment_score = context_entails_response(
            document, summaries, entailment_model
        )
        answer_entailment_score = context_entails_response(
            reference, summaries, entailment_model
        )

        # Print entailment scores (existing logic)
        print(f"Context entailment score: {context_entailment_score}")
        if context_entailment_score == 0:
            print(f"Contradiction")
        elif context_entailment_score == 1:
            print(f"Neutral")
        else:
            print(f"Entailment")

        print(f"Answer entailment score: {answer_entailment_score}")
        if answer_entailment_score == 0:
            print(f"Contradiction")
        elif answer_entailment_score == 1:
            print(f"Neutral")
        else:
            print(f"Entailment")

        # Calculate basic metrics first
        predictive_ent = predictive_entropy(log_probs)
        cluster_ent = cluster_assignment_entropy(semantic_ids)

        metrics = {
            "document_id": doc_id,
            "predictive_entropy": predictive_ent,
            "cluster_entropy": cluster_ent,
            "context_entailment": context_entails_response(
                document, summaries, entailment_model
            ),
            "reference_alignment": entailment_model.check_implication(
                reference, summaries[0]
            ),
            "generated_summaries": summaries,
            # New metrics from SQuAD implementation
            "mean_sequence_length": np.mean([len(s.split()) for s in summaries]),
            "response_diversity": len(set(summaries)) / len(summaries),
            "max_logprob": max(log_probs),
            "min_logprob": min(log_probs),
            "logprob_range": max(log_probs) - min(log_probs),
            "num_semantic_clusters": len(set(semantic_ids)),
            "largest_cluster_size": max(semantic_cluster_counts),
            "cluster_size_std": np.std(semantic_cluster_counts),
            "context_answer_entailment_gap": abs(
                context_entailment_score - answer_entailment_score
            ),
            "majority_summary_frequency": max(semantic_cluster_counts)
            / len(semantic_ids),
            "semantic_agreement_score": len(set(semantic_ids)) / len(summaries),
            "logprob_confidence_correlation": np.corrcoef(log_probs, confidence_scores)[
                0, 1
            ],
            "entropy_cluster_correlation": abs(predictive_ent - cluster_ent),
            "high_confidence_entailment": np.mean(
                [
                    c
                    for c, s in zip(confidence_scores, semantic_ids)
                    if s == semantic_ids[0]
                ]
            ),
        }

        # Log metrics
        for metric, value in metrics.items():
            if metric != "generated_summaries":
                logging.info(f"Document {doc_id} - {metric}: {value:.4f}")

        return metrics

    except Exception as e:
        logging.error(f"Failed to evaluate document {doc_id}: {str(e)}", exc_info=True)
        return None


def main():
    """Main function to run the evaluation"""
    log_filename = setup_logging()
    logging.info("Starting semantic entropy evaluation with branching generation")

    try:
        # Load models and tokenizer
        logging.info("Loading models and tokenizer")
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model, tokenizer = load_model_and_tokenizer(model_name)
        entailment_model = EntailmentDeberta()
        logging.info("Models loaded successfully")

        # Load dataset
        logging.info("Loading XSum dataset")
        dataset = get_dataset("xsum")
        eval_dataset = dataset["validation"].select(range(5))
        logging.info(f"Evaluation dataset size: {len(eval_dataset)} documents")

        # Initialize results storage
        results = []

        # Track memory usage
        if torch.cuda.is_available():
            logging.info(
                f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )

        # Evaluate each document
        for idx, item in enumerate(tqdm(eval_dataset)):
            logging.info(
                f"\nProcessing document {idx + 1}/{len(eval_dataset)}: {item['id']}"
            )

            metrics = evaluate_document(
                item["document"],
                item["summary"],
                item["id"],
                model,
                tokenizer,
                entailment_model,
            )

            if metrics:
                results.append(metrics)

                if torch.cuda.is_available():
                    logging.debug(
                        f"GPU memory after document {idx + 1}: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    )

        # Generate visualizations and save results
        if results:
            logging.info("Generating visualizations and saving results...")
            visualizer = ResultsVisualizer(results)

            # Save results to CSV
            output_file = f"xsum_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df = pd.DataFrame(
                [
                    {k: v for k, v in r.items() if k != "generated_summaries"}
                    for r in results
                ]
            )
            df.to_csv(output_file, index=False)
            logging.info(f"Results saved to {output_file}")

            # Generate all visualizations
            try:
                visualizer.plot_metric_correlations()
                visualizer.plot_metrics_over_documents()
                summary_stats = visualizer.generate_summary_statistics()
                logging.info(f"Generated summary statistics:\n{summary_stats}")

                for metric in df.select_dtypes(include=[np.number]).columns:
                    visualizer.plot_metric_distribution(metric)

            except Exception as e:
                logging.error(f"Error generating visualizations: {str(e)}")

            logging.info(
                f"Successfully processed {len(results)}/{len(eval_dataset)} documents"
            )
        else:
            logging.error("No results generated - all documents failed processing")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
