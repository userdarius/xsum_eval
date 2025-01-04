"""
Main script to run the semantic entropy evaluation on XSum dataset with enhanced logging
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
    speculative_sampling_v2,
    load_approx_and_target_model_and_tokenizer,
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
        # Convert results list to DataFrame, excluding generated_summaries
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
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.results.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation between Metrics")
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


# Enhanced logging configuration
def setup_logging():
    """Configure logging with detailed formatting and both file and console handlers"""
    log_filename = f"semantic_entropy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create formatters and handlers
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    console_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {log_filename}")

    return log_filename


def generate_summaries(
    approx_model,
    target_model,
    tokenizer,
    text,
    reference_summary,
    doc_id,
    num_samples=10,
):
    """Generate multiple summaries for a given text using speculative sampling"""
    logging.info(f"Generating {num_samples} summaries for document ID: {doc_id}")
    logging.debug(f"Input text length: {len(text)} characters")

    prompt = "Write a simple one-sentence summary of this news article: " f"{text}"

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(target_model.device)
        logging.debug(f"Tokenized input length: {inputs['input_ids'].size()}")
    except Exception as e:
        logging.error(f"Tokenization failed: {str(e)}")
        raise

    summaries = []
    log_probs = []
    logging.info(f"Document: {text}")
    logging.info(f"Reference summary: {reference_summary}")

    for sample_idx in range(num_samples):
        logging.debug(f"Generating sample {sample_idx + 1}/{num_samples}")
        try:
            # Use speculative sampling instead of regular generate
            output_ids = speculative_sampling_v2(
                prefix=inputs.input_ids,
                approx_model=approx_model,
                target_model=target_model,
                max_len=30,  # max_new_tokens
                gamma=4,  # number of tokens to speculate
                temperature=0.5,
                top_k=20,
                top_p=0.85,
                random_seed=None,
            )

            full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logging.debug(f"Raw output length: {len(full_output)} characters")

            # Clean up the summary
            summary = (
                full_output.replace(prompt, "")
                .strip()
                .split(".")[0]  # Take first sentence
                .strip()
                + "."
            )

            # Validate summary
            if len(summary) > 15 and summary != "." and not summary.startswith("http"):
                logging.info(
                    f"Sample {sample_idx + 1} - Valid summary generated: {summary}"
                )
                summaries.append(summary)

                # Calculate approximate log probabilities
                # Since speculative sampling doesn't return scores directly,
                # we'll need to calculate them separately
                with torch.no_grad():
                    input_ids = tokenizer(summary, return_tensors="pt").input_ids.to(
                        target_model.device
                    )
                    outputs = target_model(input_ids, labels=input_ids)
                    log_prob = -outputs.loss.item() * input_ids.size(1)
                    log_probs.append(log_prob)

                logging.debug(
                    f"Sample {sample_idx + 1} log probability: {log_prob:.4f}"
                )
            else:
                logging.warning(f"Sample {sample_idx + 1} failed validation: {summary}")
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error generating sample {sample_idx + 1}: {str(e)}")
            continue

    logging.info(
        f"Successfully generated {len(summaries)}/{num_samples} valid summaries"
    )
    return summaries, log_probs


def evaluate_document(
    document,
    reference,
    doc_id,
    approx_model,
    target_model,
    tokenizer,
    entailment_model,
    num_samples=10,
):
    """Evaluate semantic entropy metrics for a single document"""
    logging.info(f"Starting evaluation for document ID: {doc_id}")
    logging.debug(
        f"Document length: {len(document)} chars, Reference length: {len(reference)} chars"
    )

    try:
        # Generate multiple summaries
        summaries, log_probs = generate_summaries(
            approx_model,
            target_model,
            tokenizer,
            document,
            reference,
            doc_id,
            num_samples,
        )

        if not summaries:
            logging.error(f"No valid summaries generated for document {doc_id}")
            return None

        # Calculate semantic IDs
        logging.info("Calculating semantic IDs")
        semantic_ids = get_semantic_ids(summaries, entailment_model)
        logging.info(f"Semantic IDs distribution: {np.bincount(semantic_ids)}")

        # Calculate metrics
        logging.debug("Computing evaluation metrics")
        metrics = {
            "predictive_entropy": predictive_entropy(log_probs),
            "cluster_entropy": cluster_assignment_entropy(semantic_ids),
            "context_entailment": context_entails_response(
                document, summaries, entailment_model
            ),
            "reference_alignment": entailment_model.check_implication(
                reference, summaries[0]
            ),
            "generated_summaries": summaries,
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
    logging.info("Starting semantic entropy evaluation")

    try:
        # Load models and tokenizer
        logging.info("Loading models and tokenizer")
        approx_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        target_model_name = "meta-llama/Llama-3.2-3B-Instruct"
        approx_model, target_model, tokenizer = (
            load_approx_and_target_model_and_tokenizer(
                approx_model_name, target_model_name
            )
        )
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
                approx_model,
                target_model,
                tokenizer,
                entailment_model,
            )

            if metrics:
                metrics["document_id"] = item["id"]
                results.append(metrics)

                if torch.cuda.is_available():
                    logging.debug(
                        f"GPU memory after document {idx + 1}: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    )

        # Calculate and log aggregate metrics
        logging.info("\nCalculating aggregate metrics")
        if results:
            agg_metrics = {
                "avg_predictive_entropy": np.mean(
                    [r["predictive_entropy"] for r in results]
                ),
                "avg_cluster_entropy": np.mean([r["cluster_entropy"] for r in results]),
                "avg_context_entailment": np.mean(
                    [r["context_entailment"] for r in results]
                ),
                "avg_reference_alignment": np.mean(
                    [r["reference_alignment"] for r in results]
                ),
            }

            logging.info("Final Aggregate Metrics:")
            for metric, value in agg_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            logging.info(
                f"Successfully processed {len(results)}/{len(eval_dataset)} documents"
            )
        else:
            logging.error("No results generated - all documents failed processing")

        # Generate visualizations
        try:
            logging.info("Generating visualizations...")
            visualizer = ResultsVisualizer(results)  # Pass the results list directly

            # Generate visualizations
            for metric in [
                "predictive_entropy",
                "cluster_entropy",
                "context_entailment",
                "reference_alignment",
            ]:
                try:
                    visualizer.plot_metric_distribution(metric)
                    logging.info(f"Generated distribution plot for {metric}")
                except Exception as e:
                    logging.error(
                        f"Failed to generate distribution plot for {metric}: {str(e)}"
                    )

            try:
                visualizer.plot_metric_correlations()
                logging.info("Generated correlation heatmap")
            except Exception as e:
                logging.error(f"Failed to generate correlation heatmap: {str(e)}")

            # Scatter plots
            metric_pairs = [
                ("predictive_entropy", "cluster_entropy"),
                ("context_entailment", "reference_alignment"),
            ]
            for x_metric, y_metric in metric_pairs:
                try:
                    visualizer.plot_metrics_scatter(x_metric, y_metric)
                    logging.info(f"Generated scatter plot for {x_metric} vs {y_metric}")
                except Exception as e:
                    logging.error(
                        f"Failed to generate scatter plot for {x_metric} vs {y_metric}: {str(e)}"
                    )

            try:
                visualizer.plot_metrics_over_documents()
                logging.info("Generated metrics over documents plot")
            except Exception as e:
                logging.error(
                    f"Failed to generate metrics over documents plot: {str(e)}"
                )

            try:
                summary_stats = visualizer.generate_summary_statistics()
                logging.info(f"Generated summary statistics:\n{summary_stats}")
            except Exception as e:
                logging.error(f"Failed to generate summary statistics: {str(e)}")

        except Exception as e:
            logging.error(f"Failed to generate visualizations: {str(e)}")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
