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
        help="Mode to run: 'train','infer' or 'experiment'"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        help="Task name (optional)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode (disables logging, saves locally, etc.)"
    )

    args = parser.parse_args()

    return args.mode, args.task, args.dev