import os
import argparse
import signal

import numpy as np
import yaml
import sys

from matplotlib import pyplot as plt

from Dataset import Dataset
from Reinforce import Reinforce


def make_handler(log_path, path_name):
    def handle_sigint(sig, frame):
        print("\n[INFO] Manual Interruption (Ctrl+C).")
        plot_from_log(log_path, path_name)
        print("[INFO] Graphic generated. exit from the program.")
        sys.exit(0)
    return handle_sigint


def plot_from_log(log_path, path_name):
    log_path = os.path.join("UIR", log_path)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"[ERROR] path not found : {log_path}")

    steps, val1, val2 = [], [], []
    with open(log_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                s, v1, v2 = parts
                steps.append(int(s))
                val1.append(float(v1))
                val2.append(float(v2))

    # crea cartella di output se non esiste
    plot_dir = os.path.join("UIR", "plot")
    os.makedirs(plot_dir, exist_ok=True)

    # nome file del plot
    base_name = path_name
    plot_path = os.path.join(plot_dir, f"{base_name}.png")

    # plot
    plt.figure()
    plt.plot(steps, val1)
    plt.xlabel("Steps")
    plt.ylabel("avg")
    plt.title(f"average_nb_applicable_jobs - {base_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Grafico salvato in: {plot_path}")
def create_and_print_dataset(config):
    """Create and initialize the dataset for the recommendation system.
    
    This function creates a Dataset instance using the provided configuration
    and prints its summary information.
    
    Args:
        config (dict): Configuration dictionary containing dataset parameters
        
    Returns:
        Dataset: Initialized dataset object containing learners, jobs, and courses
    """
    dataset = Dataset(config)
    print(dataset)
    return dataset


def main(k=0, seed=42):
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
        "--config", help="Path to the configuration file", default="UIR/config/run.yaml"
    )

    args = parser.parse_args()

    # First load initial config
    with open(args.config, "r") as f:
        initial_config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize beta1 and beta2 as None
    beta1 = None
    beta2 = None

    # Run weight optimization if using weighted reward and weights are not in config
    if initial_config.get("feature") == "Weighted-Usefulness-as-Rwd":
        model_weights = initial_config.get("model_weights", {})
        if initial_config["model"] not in model_weights:
            print(f"\nOptimizing weights for {initial_config['model'].upper()}...")
            from weight_optimization import optimize_weights
            optimize_weights(args.config)
            
            # Reload config after weight optimization
            with open(args.config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = initial_config
            weights = model_weights[initial_config["model"]]
            print(f"\nUsing existing weights for {initial_config['model'].upper()}: beta1={weights['beta1']}, beta2={weights['beta2']}")

        # Get beta values for current model
        model_weights = config.get("model_weights", {})
        current_weights = model_weights.get(config["model"], {})
        beta1 = current_weights.get("beta1")
        beta2 = current_weights.get("beta2")
    else:
        config = initial_config
    if k != 0:
        config['k'] = int(k)
        config['seed'] = int(seed)
    for run in range(config["nb_runs"]):
        
        
        dataset = create_and_print_dataset(config)
        
        # Use the Reinforce class for all models
        if config["baseline"]: 
            print("feature: baseline")
            print("-------------------------------------------")
        else: 
            print(f"feature: {config['feature']}")
            print("-------------------------------------------")
            
        recommender = Reinforce(
            dataset=dataset,
            model=config["model"],
            k=config["k"],
            threshold=config["threshold"],
            run=run,
            save_name=config["name_exp"],
            total_steps=config["total_steps"],
            eval_freq=config["eval_freq"],
            feature=config["feature"],
            baseline=config["baseline"],
            method=config['method'],
            beta1=beta1,
            beta2=beta2
        )
        plot_filename = f"{config['name_exp']}_k{config['k']}"
        log_path = f"results_k{config['k']}_seed{config['seed']}/{recommender.all_results_filename}"

        signal.signal(signal.SIGINT, make_handler(log_path=log_path, path_name=plot_filename))

        recommender.reinforce_recommendation()

        plot_from_log(log_path, plot_filename)
        print(f"Model plot saved in: UIR/plot/{plot_filename}")

        


if __name__ == "__main__":
    main()
    '''k_s = np.arange(1, 6)
    seeds = np.arange(42, 53)
    for seed in seeds:
        for k in k_s:
            if int(k) == 1:
                if int(seed) == 42:
                    continue  # skip the first run (k=1, seed=42) as it's already done
            print(f"Processing experiment with sequence {k} and seed {seed}")
            main(k=k, seed=seed)'''
