import utils
import argparse

parser = argparse.ArgumentParser(
    description="Plot training metrics from pickle files."
)
parser.add_argument(
    "--files",
    nargs='+',
    default=["qnn.pkl", "cnn.pkl"],
    help="List of pickle files containing training metrics. Example: --files qnn.pkl cnn.pkl"
)
args = parser.parse_args()
utils.plot_training_metrics_detailed(args.files)