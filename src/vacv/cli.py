from __future__ import annotations

import argparse
from pathlib import Path

from .utils import read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Nuclear IT Hack: reproducible SMILES-only SI regression")
    sub = parser.add_subparsers(dest="command", required=True)
    eda = sub.add_parser("eda")
    eda.add_argument("--config", default="configs/baseline.json")
    eda.add_argument("--train", default="data/raw/train.csv")
    eda.add_argument("--test", default="data/raw/test.csv")
    eda.add_argument("--output", default="reports/eda")
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/baseline.json")
    train.add_argument("--train", default="data/raw/train.csv")
    train.add_argument("--output", default="artifacts/baseline")
    train.add_argument("--seed", type=int)
    train.add_argument("--models", nargs="+")
    predict = sub.add_parser("predict")
    predict.add_argument("--model", default="artifacts/baseline/model.joblib")
    predict.add_argument("--input", default="data/raw/test.csv")
    predict.add_argument("--template", default="data/raw/sample_submit.csv")
    predict.add_argument("--output", default="artifacts/submission.csv")
    reproduce = sub.add_parser("reproduce")
    reproduce.add_argument("--config", default="configs/baseline.json")
    reproduce.add_argument("--data-dir", default="data/raw")
    reproduce.add_argument("--output", default="artifacts/reproduced")
    collect = sub.add_parser("collect")
    collect.add_argument("--output", default="data/external")
    collect.add_argument("--offline", action="store_true", help="Use cached HTTP responses only")
    args = parser.parse_args(argv)
    if args.command == "collect":
        from .collect import collect_chembl
        collect_chembl(args.output, offline=args.offline)
    elif args.command == "predict":
        from .pipeline import predict_pipeline
        predict_pipeline(args.model, args.input, args.template, args.output)
    else:
        config = read_config(args.config)
        if args.command == "eda":
            from .eda import eda_pipeline
            eda_pipeline(args.train, args.test, config, args.output)
        elif args.command == "train":
            from .pipeline import train_pipeline
            if args.seed is not None:
                config["seed"] = args.seed
            if args.models:
                config["models"] = args.models
            train_pipeline(args.train, config, args.output)
        elif args.command == "reproduce":
            from .eda import eda_pipeline
            from .pipeline import predict_pipeline, train_pipeline
            data, output = Path(args.data_dir), Path(args.output)
            eda_pipeline(data / "train.csv", data / "test.csv", config, output / "eda")
            train_pipeline(data / "train.csv", config, output / "model")
            predict_pipeline(output / "model/model.joblib", data / "test.csv", data / "sample_submit.csv", output / "submission.csv")


if __name__ == "__main__":
    main()
