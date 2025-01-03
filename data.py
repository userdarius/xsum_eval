""" Load dataset """

import datasets


def get_dataset(name):
    """Load and return a dataset by name."""
    print(f"Loading {name} dataset")

    df = datasets.load_dataset(name)
    return {"document": df["document"], "id": df["id"], "summary": df["summary"]}
