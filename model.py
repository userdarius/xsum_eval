""" Load huggingface model and tokenizer """

import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch.nn.functional as F
import os
from typing import Dict, Tuple, List


### Main model ###
def load_model(model_name):
    """
    Load a model from HuggingFace
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def load_tokenizer(model_name):
    """
    Load a tokenizer from HuggingFace
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token to eos token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        raise


def load_model_and_tokenizer(model_name):
    """
    Load a model and tokenizer from HuggingFace
    """
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer


### Entailment Model ###
class BaseEntailment:
    """Base class for entailment models."""

    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """Entailment model using Deberta-v2-xlarge-mnli."""

    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(self.device)

    def check_implication(self, text1, text2, *args, **kwargs):
        """
        Check implication between two texts
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(
            F.softmax(logits, dim=1)
        )  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get("DEBERTA_FULL_LOG", False):
            logging.info("Deberta Input: %s -> %s", text1, text2)
            logging.info("Deberta Prediction: %s", prediction)

        return prediction


### Branching Model ###
def get_topk_next_tokens(
    model: AutoModelForCausalLM,
    inputs: Dict[str, torch.Tensor],
    num_branches: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the top k most likely next tokens and their probabilities.
    Also returns the log probabilities.
    """
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        # Apply temperature scaling
        next_token_logits = next_token_logits / temperature

    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    probabilities = torch.softmax(next_token_logits, dim=-1)
    topk_values, topk_indices = torch.topk(probabilities, num_branches)
    topk_logprobs = torch.gather(log_probs, -1, topk_indices)

    return topk_values, topk_indices, topk_logprobs


def generate_branching_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int,
    num_branches: int = 10,
) -> List[Tuple[str, float, float]]:
    """
    Generate multiple responses with one low-temperature and rest high-temperature outputs.
    Returns tuples of (text, confidence_score, log_probability)
    """
    responses = []

    # First generate one low-temperature response
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("\nGenerating low-temperature response (temperature=0.3)")
    # Get initial tokens with low temperature
    topk_values, topk_indices, topk_logprobs = get_topk_next_tokens(
        model, inputs, num_branches=1, temperature=0.3
    )

    # Generate low-temperature response
    branch_inputs = {
        "input_ids": torch.cat([inputs["input_ids"], topk_indices[:, 0:1]], dim=1),
        "attention_mask": (
            torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((1, 1), device=inputs["attention_mask"].device),
                ],
                dim=1,
            )
            if "attention_mask" in inputs
            else None
        ),
    }

    generated_text, confidence_score, log_prob = generate_single_branch(
        model, tokenizer, max_length, branch_inputs, temperature=0.3
    )
    responses.append((generated_text, confidence_score, log_prob))

    # Then generate high-temperature responses
    print("\nGenerating high-temperature responses (temperature=1.2)")
    # Get initial tokens with high temperature
    topk_values, topk_indices, topk_logprobs = get_topk_next_tokens(
        model, inputs, num_branches=num_branches - 1, temperature=1.2
    )

    # Generate remaining responses with high temperature
    for k in range(num_branches - 1):
        print(f"\nStarting high-temp branch {k+1}")
        first_token = topk_indices[:, k : k + 1]
        first_token_text = tokenizer.decode(first_token[0])
        print(f"First token: '{first_token_text}'")

        if any(
            stop in first_token_text
            for stop in [".", "\n", "Explanation:", "Answer:", r" \ ", "\\"]
        ):
            print(f"Skipping branch {k+1} because it starts with a stop token")
            continue

        branch_inputs = {
            "input_ids": torch.cat(
                [inputs["input_ids"], topk_indices[:, k : k + 1]], dim=1
            ),
            "attention_mask": (
                torch.cat(
                    [
                        inputs["attention_mask"],
                        torch.ones((1, 1), device=inputs["attention_mask"].device),
                    ],
                    dim=1,
                )
                if "attention_mask" in inputs
                else None
            ),
        }

        generated_text, confidence_score, log_prob = generate_single_branch(
            model, tokenizer, max_length, branch_inputs, temperature=1.2
        )
        responses.append((generated_text, confidence_score, log_prob))

    print("\nAll branches complete\n")
    return responses


def generate_single_branch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int,
    inputs: Dict[str, torch.Tensor],
    temperature: float = 1.0,
) -> Tuple[str, float, float]:
    """
    Generate a single response branch with specified temperature.
    """
    response_tokens = [inputs["input_ids"][0, -1].item()]
    prob_diffs = []
    sequence_logprob = 0.0

    print(f"Starting with initial token: '{tokenizer.decode([response_tokens[0]])}'")

    for step in range(max_length):
        topk_values, topk_indices, topk_logprobs = get_topk_next_tokens(
            model, inputs, num_branches=10, temperature=temperature
        )

        next_token = topk_indices[0, 0].item()
        next_token_text = tokenizer.decode([next_token])
        sequence_logprob += topk_logprobs[0, 0].item()

        current_text = tokenizer.decode(response_tokens + [next_token])
        print(f"Step {step}: Token {next_token} -> '{next_token_text}'")
        print(f"Current text: '{current_text}'")

        if any(
            stop in next_token_text for stop in [".", "\n", "Explanation:", "Answer:"]
        ):
            if next_token_text.strip() == "." and not any(
                stop in current_text for stop in [".", "\n", "Explanation:", "Answer:"]
            ):
                response_tokens.append(next_token)
                sequence_logprob += topk_logprobs[0, 0].item()
            break

        prob_diff = (topk_values[0, 0] - topk_values[0, 1]).item()
        response_tokens.append(next_token)
        prob_diffs.append(prob_diff)

        next_token_tensor = torch.tensor(
            [[next_token]], device=inputs["input_ids"].device
        )
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_tensor], dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((1, 1), device=inputs["attention_mask"].device),
                ],
                dim=1,
            )

    generated_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    avg_prob_diff = sum(prob_diffs) / len(prob_diffs) if prob_diffs else 0
    normalized_logprob = (
        sequence_logprob / len(response_tokens) if response_tokens else 0
    )

    return generated_text.strip(), avg_prob_diff, normalized_logprob
