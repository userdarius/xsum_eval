"""
Main script to run the evaluation script
"""

import logging
import torch
from tqdm import tqdm
from model import load_model_and_tokenizer, EntailmentDeberta
from data import get_dataset


logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function to run the evaluation script
    """
    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Load entailment model
    entailment_model = EntailmentDeberta()

    # Load dataset
    dataset_name = "xsum"
    dataset = get_dataset(dataset_name)

    # Print model and tokenizer
    print(model)
    print(tokenizer)

    # Print the entailment model
    print(entailment_model)

    # Print the first 10 rows of the dataset
    print(dataset["train"]["document"][:10])
    print(dataset["train"]["summary"][:10])
    print(dataset["train"]["id"][:10])


if __name__ == "__main__":
    main()
