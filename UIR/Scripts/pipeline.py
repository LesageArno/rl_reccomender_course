import os
import argparse
import signal
import sys
from matplotlib import pyplot as plt
import yaml

from .Dataset import Dataset
from .Reinforce import Reinforce


def make_handler(log_path, path_name):
    """Return a SIGINT handler that triggers a plot from the given log and exits."""

    def handle_sigint(sig, frame):
        print("\n[INFO] Manual Interruption (Ctrl+C).")
        plot_from_log(log_path, path_name)
        print("[INFO] Graphic generated. Exiting the program.")
        sys.exit(0)

    return handle_sigint


def plot_from_log(log_path, path_name):
    """Read log data and generate a plot of the training performance."""
    log_path = os.path.join("UIR", log_path)
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"[ERROR] path not found : {log_path}")

    steps, val1, val2 = [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                s, v1, v2 = parts
                steps.append(int(s))
                val1.append(float(v1))
                val2.append(float(v2))

    # create output folder if it does not exist
    plot_dir = os.path.join("UIR", "plot")
    os.makedirs(plot_dir, exist_ok=True)

    # define plot file name
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

    print(f"[INFO] Plot saved to: {plot_path}")


def create_and_print_dataset(config):
    """Create the dataset and print its summary information."""
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
        "--config",
        help="Path to the configuration file",
        default="UIR/config/run.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        initial_config = yaml.load(f, Loader=yaml.FullLoader)

    beta1 = None
    beta2 = None

    # handle weighted reward configuration
    if initial_config.get("feature") == "Weighted-Usefulness-as-Rwd":
        model_weights = initial_config.get("model_weights", {})
        if initial_config["model"] not in model_weights:
            print(f"\nOptimizing weights for {initial_config['model'].upper()}...")
            from weight_optimization import optimize_weights
            optimize_weights(args.config)
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = initial_config
            weights = model_weights[initial_config["model"]]
            print(
                f"\nUsing existing weights for {initial_config['model'].upper()}: "
                f"beta1={weights['beta1']}, beta2={weights['beta2']}"
            )

        model_weights = config.get("model_weights", {})
        current_weights = model_weights.get(config["model"], {})
        beta1 = current_weights.get("beta1")
        beta2 = current_weights.get("beta2")
    else:
        config = initial_config

    # allow overriding k and seed
    if k != 0:
        config["k"] = int(k)
        config["seed"] = int(seed)

    for run in range(config["nb_runs"]):
        dataset = create_and_print_dataset(config)

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
            method=config["method"],
            beta1=beta1,
            beta2=beta2,
            params=config.get("hypers", None),
        )

        plot_filename = f"{config['name_exp']}_k{config['k']}"
        log_path = f"results_k{config['k']}_seed{config['seed']}/{recommender.all_results_filename}"

        signal.signal(signal.SIGINT, make_handler(log_path=log_path, path_name=plot_filename))

        recommender.reinforce_recommendation()

        plot_from_log(log_path, plot_filename)
        print(f"Model plot saved in: UIR/plot/{plot_filename}")


if __name__ == "__main__":
    main()
