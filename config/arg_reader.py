import argparse

from config.mode import Mode

def read_arguments() -> Mode:
    parser = argparse.ArgumentParser(description="FasterRCNN runner")

    parser.add_argument(
        "--mode",
        "-m",
        type=Mode,
        choices=list(Mode),
        required=True,
        help="Mode to run: 'train' or 'infer'"
    )

    args = parser.parse_args()

    return args.mode