import argparse
from pathlib import Path
import yaml

from .Dataset import Dataset

from .Reinforce import Reinforce
import torch
torch.distributions.Distribution.set_default_validate_args(False)

def check_paths(cfg: dict) -> None:
    for key in ["taxonomy_path", "course_path", "cv_path", "job_path", "mastery_levels_path"]:
        p = Path(cfg[key])
        if not p.exists():
            raise FileNotFoundError(f"Missing file for '{key}': {p}")

def create_and_print_dataset(config):
    """Create the dataset and print its summary information."""
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main():
    """Main entry point for the recommendation system pipeline.

    This function:
    1. Parses command line arguments to get the configuration file path
    2. Loads the configuration from YAML file
    3. Handles weight optimization if needed
    4. Runs the specified recommendation model for the configured number of iterations

    Command line arguments:
        --config: Path to the configuration file (default: "UIR/config/run.yaml")
    """
    parser = argparse.ArgumentParser(description="Run recommender models.")
    parser.add_argument(
        "--config",
        help="Path to the configuration file",
        default="UIR/config/run.yaml",
    )
    args = parser.parse_args()
    path = args.config
    with open(path, "r", encoding="utf-8") as f:
        initial_config = yaml.load(f, Loader=yaml.FullLoader)
    
    if not isinstance(initial_config, dict):
        raise ValueError(f"Invalid YAML config (expected a dict): {path}")

    config = initial_config
    check_paths(config)

    dataset = create_and_print_dataset(config)
    exit(0)
        
    #print(f"feature: {config['feature']}")
    #print("-------------------------------------------")

    #recommender = Reinforce(
    #    dataset=dataset,
    #    config = config,
    #)

    #recommender.reinforce_recommendation()


if __name__ == "__main__":
    main()
