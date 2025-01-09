"""
Main script to run the semantic entropy evaluation on XSum dataset with branching generation
"""

import sys
import os
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
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calculate_rouge(reference, candidates):
    """
    Calculate ROUGE scores for a set of candidate summaries against a reference

    Args:
        reference (str): The reference summary
        candidates (list): List of generated candidate summaries

    Returns:
        dict: Average ROUGE scores across all candidates
    """
    # Initialize the ROUGE scorer with multiple ROUGE variants
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Calculate ROUGE for each candidate
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for candidate in candidates:
        scores = scorer.score(reference, candidate)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    # Calculate average scores
    avg_scores = {
        "rouge1": np.mean(rouge_scores["rouge1"]),
        "rouge2": np.mean(rouge_scores["rouge2"]),
        "rougeL": np.mean(rouge_scores["rougeL"]),
    }

    return avg_scores


def calculate_bleu(reference, candidates):
    """
    Calculate BLEU score for a set of candidate summaries against a reference

    Args:
        reference (str): The reference summary
        candidates (list): List of generated candidate summaries

    Returns:
        float: Average BLEU score across all candidates
    """
    # Tokenize reference
    reference_tokens = nltk.word_tokenize(reference.lower())

    # Initialize smoothing function for BLEU
    smoother = SmoothingFunction().method1

    # Calculate BLEU for each candidate
    bleu_scores = []
    for candidate in candidates:
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        # Calculate BLEU with equal weights for 1-4 grams
        score = sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoother,
        )
        bleu_scores.append(score)

    # Return average BLEU score
    return np.mean(bleu_scores)


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_experiment_dir():
    """Create a timestamped directory for the current experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(RESULTS_DIR, f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


class ResultsVisualizer:
    def __init__(self, results_list, experiment_dir):
        """Initialize with a list of result dictionaries and experiment directory"""
        cleaned_results = [
            {
                k: v
                for k, v in r.items()
                if k != "generated_summaries" and not isinstance(v, (list, dict))
            }
            for r in results_list
        ]
        self.results = pd.DataFrame(cleaned_results)
        self.experiment_dir = experiment_dir

    def plot_metric_distribution(self, metric_name, title=None):
        """Create a distribution plot for a specific metric"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.results[metric_name], kde=True)
        plt.title(title or f"Distribution of {metric_name}")
        plt.xlabel(metric_name)
        plt.ylabel("Count")
        plt.savefig(
            os.path.join(self.experiment_dir, f"{metric_name}_distribution.png")
        )
        plt.close()

    def plot_metric_correlations(self):
        """Create a correlation heatmap between different metrics"""
        plt.figure(figsize=(12, 10))
        numeric_cols = self.results.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.results[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Between Uncertainty Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "metric_correlations.png"))
        plt.close()

    def plot_metrics_scatter(self, x_metric, y_metric):
        """Create a scatter plot comparing two metrics"""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.results, x=x_metric, y=y_metric)
        plt.title(f"{x_metric} vs {y_metric}")
        plt.xlabel(x_metric)
        plt.ylabel(y_metric)
        plt.savefig(
            os.path.join(self.experiment_dir, f"{x_metric}_vs_{y_metric}_scatter.png")
        )
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
        plt.savefig(os.path.join(self.experiment_dir, "metrics_across_documents.png"))
        plt.close()

    def plot_semantic_clustering_metrics(self):
        """Create a combined plot of semantic clustering metrics"""
        semantic_metrics = [
            "num_semantic_clusters",
            "largest_cluster_size",
            "cluster_size_std",
            "semantic_agreement_score",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Semantic Clustering Metrics Analysis")

        for idx, metric in enumerate(semantic_metrics):
            row = idx // 2
            col = idx % 2
            sns.boxplot(data=self.results[metric], ax=axes[row, col])
            axes[row, col].set_title(f"{metric} Distribution")
            axes[row, col].set_xlabel(metric)

        plt.tight_layout()
        plt.savefig("semantic_clustering_analysis.png")
        plt.close()

    def plot_probability_metrics(self):
        """Create visualizations for probability-based metrics"""
        prob_metrics = ["max_logprob", "min_logprob", "logprob_range"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Probability Metrics Analysis")

        for idx, metric in enumerate(prob_metrics):
            sns.violinplot(data=self.results[metric], ax=axes[idx])
            axes[idx].set_title(f"{metric} Distribution")
            axes[idx].set_xlabel(metric)

        plt.tight_layout()
        plt.savefig("probability_metrics_analysis.png")
        plt.close()

    def plot_entailment_analysis(self):
        """Create visualizations for entailment-based metrics"""
        plt.figure(figsize=(12, 8))

        metrics = [
            "context_answer_entailment_gap",
            "high_confidence_entailment",
            "entropy_cluster_correlation",
        ]

        plt.figure(figsize=(15, 5))
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, idx)
            sns.kdeplot(data=self.results[metric], fill=True)
            plt.title(f"{metric} Distribution")
            plt.xlabel(metric)

        plt.tight_layout()
        plt.savefig("entailment_analysis.png")
        plt.close()

    def plot_sequence_metrics(self):
        """Create visualizations for sequence-based metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Sequence Metrics Analysis")

        sns.boxplot(data=self.results["mean_sequence_length"], ax=axes[0])
        axes[0].set_title("Mean Sequence Length Distribution")
        axes[0].set_xlabel("Mean Sequence Length")

        sns.histplot(data=self.results["response_diversity"], ax=axes[1], kde=True)
        axes[1].set_title("Response Diversity Distribution")
        axes[1].set_xlabel("Response Diversity")

        plt.tight_layout()
        plt.savefig("sequence_metrics_analysis.png")
        plt.close()

    def plot_comprehensive_metric_relationships(self):
        """Create a comprehensive analysis of relationships between different metric types"""
        # Select key metrics from each category
        key_metrics = {
            "Semantic": ["semantic_agreement_score", "num_semantic_clusters"],
            "Probability": ["max_logprob", "logprob_range"],
            "Entailment": [
                "context_answer_entailment_gap",
                "high_confidence_entailment",
            ],
            "Sequence": ["mean_sequence_length", "response_diversity"],
        }

        # Create correlation matrix for these metrics
        selected_metrics = [m for metrics in key_metrics.values() for m in metrics]
        correlation_matrix = self.results[selected_metrics].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Cross-Category Metric Correlations")
        plt.tight_layout()
        plt.savefig("comprehensive_metric_relationships.png")
        plt.close()

    def generate_summary_statistics(self):
        """Generate and save summary statistics"""
        summary_stats = self.results.describe()
        summary_stats.to_csv(
            os.path.join(self.experiment_dir, "summary_statistics.csv")
        )
        return summary_stats


def setup_logging(experiment_dir):
    """Configure logging with detailed formatting and both file and console handlers"""
    log_filename = os.path.join(experiment_dir, f"semantic_entropy.log")

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


def analyze_sequence_probs(log_probs, length_normalized=False):
    """Analyze sequence-level probability statistics"""
    sequence_stats = {
        "raw_mean_logprob": np.mean(log_probs),
        "raw_std_logprob": np.std(log_probs),
        "raw_median_logprob": np.median(log_probs),
    }
    return sequence_stats


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
        logging.info(f"Semantic IDs distribution: {np.bincount(semantic_ids)}")

        # Calculate BLEU score
        logging.info("Calculating BLEU score")
        bleu_score = calculate_bleu(reference, summaries)
        logging.info(f"BLEU score: {bleu_score:.4f}")

        # Calculate ROUGE score
        logging.info("Calculating ROUGE score")
        rouge_scores = calculate_rouge(reference, summaries)
        logging.info(f"ROUGE score: {rouge_scores}")

        # Calculate entailment scores
        context_entailment_score = context_entails_response(
            document, summaries, entailment_model
        )
        answer_entailment_score = context_entails_response(
            reference, summaries, entailment_model
        )

        predictive_ent = predictive_entropy(log_probs)
        cluster_ent = cluster_assignment_entropy(semantic_ids)

        sequence_stats = analyze_sequence_probs(log_probs, length_normalized=True)

        # Generate confidence scores (using log probabilities as proxy)
        confidence_scores = [np.exp(lp) for lp in log_probs]

        # Calculate metrics
        logging.debug("Computing evaluation metrics")
        metrics = {
            "predictive_entropy": predictive_entropy(log_probs),
            "cluster_entropy": cluster_assignment_entropy(semantic_ids),
            "context_entailment": context_entails_response(
                document, summaries, entailment_model
            ),
            "bleu_score": bleu_score,
            "rouge1_score": rouge_scores["rouge1"],
            "rouge2_score": rouge_scores["rouge2"],
            "rougeL_score": rouge_scores["rougeL"],
            "reference_alignment": entailment_model.check_implication(
                reference, summaries[0]
            ),
            # New sequence-based metrics
            "mean_sequence_length": np.mean([len(s.split()) for s in summaries]),
            "response_diversity": len(set(summaries)) / len(summaries),
            # New probability-based metrics
            "max_logprob": max(log_probs),
            "min_logprob": min(log_probs),
            "logprob_range": max(log_probs) - min(log_probs),
            # New semantic clustering metrics
            "num_semantic_clusters": len(set(semantic_ids)),
            "largest_cluster_size": max(semantic_cluster_counts),
            "cluster_size_std": np.std(semantic_cluster_counts),
            "majority_summary_frequency": max(semantic_cluster_counts)
            / len(semantic_ids),
            "semantic_agreement_score": len(set(semantic_ids)) / len(summaries),
            # New entailment-based metrics
            "context_answer_entailment_gap": abs(
                context_entailment_score - answer_entailment_score
            ),
            "high_confidence_entailment": np.mean(
                [
                    c
                    for c, s in zip(confidence_scores, semantic_ids)
                    if s == semantic_ids[0]
                ]
            ),
            "entropy_cluster_correlation": abs(predictive_ent - cluster_ent),
            # Store generated summaries
            "generated_summaries": summaries,
            "sequence_stats": sequence_stats,
        }

        # Log metrics with type checking
        for metric, value in metrics.items():
            if metric != "generated_summaries":
                if isinstance(value, (float, np.floating)):
                    logging.info(f"Document {doc_id} - {metric}: {value:.4f}")
                elif isinstance(value, (int, np.integer)):
                    logging.info(f"Document {doc_id} - {metric}: {value}")
                else:
                    logging.info(f"Document {doc_id} - {metric}: {value}")

        return metrics

    except Exception as e:
        logging.error(f"Failed to evaluate document {doc_id}: {str(e)}", exc_info=True)
        return None


def main():
    """Main function to run the evaluation"""
    experiment_dir = create_experiment_dir()
    log_filename = setup_logging(experiment_dir)
    logging.info(f"Created experiment directory: {experiment_dir}")
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
        eval_dataset = dataset["validation"].select(range(500))
        logging.info(f"Evaluation dataset size: {len(eval_dataset)} documents")

        # NLTK punkt tokenizers
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            logging.info("Downloading NLTK punkt_tab tokenizer")
            nltk.download("punkt_tab")

        # Also ensure we have the regular punkt tokenizer
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logging.info("Downloading NLTK punkt tokenizer")
            nltk.download("punkt")

        # Initialize results storage
        results = []

        # Track memory usage
        if torch.cuda.is_available():
            logging.info(
                f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
            )

        # Shuffle the dataset
        eval_dataset = eval_dataset.shuffle(seed=42)

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
                metrics["document_id"] = item["id"]
                results.append(metrics)

                if torch.cuda.is_available():
                    logging.debug(
                        f"GPU memory after document {idx + 1}: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    )

                torch.cuda.empty_cache()

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
                "avg_bleu_score": np.mean([r["bleu_score"] for r in results]),
                "avg_rouge1_score": np.mean([r["rouge1_score"] for r in results]),
                "avg_rouge2_score": np.mean([r["rouge2_score"] for r in results]),
                "avg_rougeL_score": np.mean([r["rougeL_score"] for r in results]),
            }

            logging.info("Final Aggregate Metrics:")
            for metric, value in agg_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            logging.info(
                f"Successfully processed {len(results)}/{len(eval_dataset)} documents"
            )
        else:
            logging.error("No results generated - all documents failed processing")

        try:
            logging.info("Generating visualizations...")
            visualizer = ResultsVisualizer(results, experiment_dir)  # Pass the results list directly

            # Generate visualizations
            for metric in [
                "predictive_entropy",
                "cluster_entropy",
                "context_entailment",
                "reference_alignment",
                "bleu_score",
                "rouge1_score",
                "rouge2_score",
                "rougeL_score",
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

            try:
                visualizer.plot_semantic_clustering_metrics()
                logging.info("Generated semantic clustering metrics plot")
            except Exception as e:
                logging.error(
                    f"Failed to generate semantic clustering metrics plot: {str(e)}"
                )

            try:
                visualizer.plot_probability_metrics()
                logging.info("Generated probability metrics plot")
            except Exception as e:
                logging.error(f"Failed to generate probability metrics plot: {str(e)}")

            try:
                visualizer.plot_entailment_analysis()
                logging.info("Generated entailment analysis plot")
            except Exception as e:
                logging.error(f"Failed to generate entailment analysis plot: {str(e)}")

            try:
                visualizer.plot_sequence_metrics()
                logging.info("Generated sequence metrics plot")
            except Exception as e:
                logging.error(f"Failed to generate sequence metrics plot: {str(e)}")

            try:
                visualizer.plot_comprehensive_metric_relationships()
                logging.info("Generated comprehensive metric relationships plot")
            except Exception as e:
                logging.error(
                    f"Failed to generate comprehensive metric relationships plot: {str(e)}"
                )

            # Scatter plots
            metric_pairs = [
                # Uncertainty correlations
                ("predictive_entropy", "cluster_entropy"),
                ("predictive_entropy", "rouge1_score"),
                ("cluster_entropy", "rouge1_score"),
                # Semantic correlations
                ("context_entailment", "reference_alignment"),
                ("context_entailment", "rougeL_score"),
                ("reference_alignment", "rouge1_score"),
                # N-gram metrics correlations
                ("bleu_score", "rouge1_score"),
                ("bleu_score", "rouge2_score"),
                ("bleu_score", "rougeL_score"),
                ("rouge1_score", "rouge2_score"),
                ("rouge2_score", "rougeL_score"),
                # Cross-category key correlations
                ("cluster_entropy", "rougeL_score"),
                ("predictive_entropy", "reference_alignment"),
                # Semantic clustering correlations
                ("num_semantic_clusters", "semantic_agreement_score"),
                ("largest_cluster_size", "cluster_size_std"),
                # Probability metric correlations
                ("max_logprob", "logprob_range"),
                ("min_logprob", "logprob_range"),
                # Entailment correlations
                ("context_answer_entailment_gap", "high_confidence_entailment"),
                ("entropy_cluster_correlation", "semantic_agreement_score"),
                # Sequence metric correlations
                ("mean_sequence_length", "response_diversity"),
                # Cross-category correlations
                ("semantic_agreement_score", "high_confidence_entailment"),
                ("response_diversity", "num_semantic_clusters"),
                ("logprob_range", "context_answer_entailment_gap"),
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

            additional_agg_metrics = {
                "avg_num_semantic_clusters": np.mean(
                    [r["num_semantic_clusters"] for r in results]
                ),
                "avg_semantic_agreement": np.mean(
                    [r["semantic_agreement_score"] for r in results]
                ),
                "avg_response_diversity": np.mean(
                    [r["response_diversity"] for r in results]
                ),
                "avg_sequence_length": np.mean(
                    [r["mean_sequence_length"] for r in results]
                ),
                "avg_logprob_range": np.mean([r["logprob_range"] for r in results]),
                "avg_entailment_gap": np.mean(
                    [r["context_answer_entailment_gap"] for r in results]
                ),
                "avg_high_confidence": np.mean(
                    [r["high_confidence_entailment"] for r in results]
                ),
            }

            # Add new metrics to existing aggregate metrics
            agg_metrics.update(additional_agg_metrics)

            # Log new aggregate metrics
            logging.info("\nAdditional Aggregate Metrics:")
            for metric, value in additional_agg_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

        except Exception as e:
            logging.error(f"Failed to generate visualizations: {str(e)}")

    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
