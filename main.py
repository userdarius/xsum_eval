import logging
import torch
from model import load_model_and_tokenizer, EntailmentDeberta
from data import get_dataset
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


def main():
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
    print(dataset["document"][:10])
    print(dataset["summary"][:10])
    print(dataset["id"][:10])


if __name__ == "__main__":
    main()
