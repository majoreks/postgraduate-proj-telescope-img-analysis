import argparse
from typing import Optional, Tuple

from config.mode import Mode

def read_arguments() -> Tuple[Mode, str, bool, Optional[str], str, Optional[str]]:
    parser = argparse.ArgumentParser(description="FasterRCNN runner")

    parser.add_argument(
        "--mode",
        "-m",
        type=Mode,
        choices=list(Mode),
        required=True,
        help="Mode to run: 'train' or 'infer'"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        help="Task name for logging training (training only) OR Output directory for inference (inference only)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode (disables logging, saves locally, etc.)"
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        help="Mandatory path to model weights for inference (inference only)"
    )
    parser.add_argument(
        "--resnet-type",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="ResNet backbone type to use ('resnet18','resnet34','resnet50','resnet101')"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["v1", "v2"],
        help="FasterRCNN version to use ('v1','v2') (inference and training only)"
    )

    args = parser.parse_args()

    if args.mode == Mode.INFER and args.weights_path is None:
        raise Exception("weights_path is obligatory for infer (inference) mode")
    if args.mode == Mode.INFER and args.resnet_type is None:
        raise Exception("resnet_type is obligatory for infer (inference) mode")
    if args.mode == Mode.INFER and args.model_type is None:
        raise Exception("model_type is obligatory for infer (inference) mode")

    return args.mode, args.task, args.dev, args.weights_path, args.resnet_type, args.model_type
