"""Sniffer CLI: collect data, train ONNX models, predict live, inspect models."""

import argparse

DATA_DIR = "/opt/projects/bme688/DATA"
OUT_DIR = "/opt/projects/bme688"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sniff",
        description="BME688 odor classification toolkit (ONNX models).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="Record labeled sensor sweeps into a dataset CSV.")
    _ = p.add_argument("label", help="Odor label for the collected sweeps (e.g. tea).")
    _ = p.add_argument("--sweeps", type=int, default=20, help="Sweeps to record (default: 20).")
    _ = p.add_argument("--config", default=f"{DATA_DIR}/default.bmeconfig",
                   help="Path to the .bmeconfig heater profile.")
    _ = p.add_argument("--data-dir", default=DATA_DIR, help="Directory holding dataset CSVs.")

    p = sub.add_parser("train", help="Train a classifier and export it as ONNX.")
    _ = p.add_argument("model", choices=["rf", "nn"],
                   help="rf = Random Forest, nn = Neural Net (MLP).")
    _ = p.add_argument("--data-dir", default=DATA_DIR, help="Directory holding dataset CSVs.")
    _ = p.add_argument("--out-dir", default=OUT_DIR, help="Output directory for model and plots.")

    p = sub.add_parser("predict", help="Run live predictions from the sensor.")
    _ = p.add_argument("label", help="Actual label of the presented sample (for logging).")
    _ = p.add_argument("--model", default=f"{OUT_DIR}/odor_model_nn.onnx",
                   help="Path to the ONNX model.")
    _ = p.add_argument("--sweeps", type=int, default=25, help="Max sweeps before clean exit (default: 25).")
    _ = p.add_argument("--config", default=f"{DATA_DIR}/default.bmeconfig",
                   help="Path to the .bmeconfig heater profile.")
    _ = p.add_argument("--log", default=f"{DATA_DIR}/predictions_log.csv",
                   help="Prediction log CSV path.")

    p = sub.add_parser("inspect", help="Inspect ONNX models and replay the dataset offline.")
    _ = p.add_argument("--out-dir", default=OUT_DIR, help="Directory holding the ONNX models.")
    _ = p.add_argument("--data-dir", default=DATA_DIR, help="Directory holding dataset CSVs.")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "collect":
        from . import collect_ml6
        collect_ml6.main(args)
    elif args.command == "train":
        if args.model == "rf":
            from . import classify3_odors as mod
        else:
            from . import classify_odors3_nn as mod
        mod.main(args)
    elif args.command == "predict":
        from . import predict_live
        predict_live.main(args)
    elif args.command == "inspect":
        from . import inspect_and_test3
        inspect_and_test3.main(args)


if __name__ == "__main__":
    main()
