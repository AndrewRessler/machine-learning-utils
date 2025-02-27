import json
from typing import Dict

def save_best_hyperparams(params: Dict, filepath: str = "best_hyperparams.json") -> None:
    """
    Saves the best hyperparameters to a JSON file.

    Parameters:
    - params: Dictionary containing best hyperparameters.
    - filepath: File path to save the JSON data.
    """
    try:
        with open(filepath, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Best hyperparameters saved to {filepath}")
    except Exception as e:
        print(f"Error saving hyperparameters: {e}")


def load_best_hyperparams(filepath: str = "best_hyperparams.json") -> Dict:
    """
    Loads the best hyperparameters from a JSON file.

    Parameters:
    - filepath: File path to load JSON data from.

    Returns:
    - Dictionary of best hyperparameters (empty if file not found).
    """
    try:
        with open(filepath, "r") as f:
            params = json.load(f)
        print(f"Loaded best hyperparameters from {filepath}")
        return params
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Returning empty hyperparameters.")
        return {}
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        return {}
