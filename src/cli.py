import argparse
import json
from pathlib import Path

from src.config import DEFAULT_MODEL_PATH
from src.data.loader import load_records
from src.data.validator import validate_records
from src.model import LogRiskModel
from src.training import train


def main():
    parser = argparse.ArgumentParser(description="Log risk classifier")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate"); validate.add_argument("--data", required=True)
    train_parser = commands.add_parser("train"); train_parser.add_argument("--data", required=True); train_parser.add_argument("--artifacts", default="artifacts")
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("--metrics", default="artifacts/metrics.json")
    predict = commands.add_parser("predict")
    for name in ("service", "environment", "log-level", "message", "dependency"):
        predict.add_argument(f"--{name}", required=name != "dependency")
    predict.add_argument("--status-code", type=int); predict.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    predict_json = commands.add_parser("predict-json"); predict_json.add_argument("--input", required=True); predict_json.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()
    if args.command == "validate":
        records, stats = load_records(args.data); report = validate_records(records, stats)
        print("Dataset validation\n------------------")
        print(f"Rows: {len(records)}\n\nClasses:")
        for name, count in report.class_counts.items(): print(f"{name or '<missing>':9} {count}")
        print("\nMissing values:"); [print(f"{key:15} {value}") for key, value in report.null_counts.items()]
        print(f"\nDuplicate rows: {report.duplicate_rows}")
        for warning in report.warnings: print(f"Warning: {warning}")
        for error in report.errors: print(f"Error: {error}")
        print(f"\nValidation: {'PASSED' if report.valid else 'FAILED'}")
        return 0 if report.valid else 1
    if args.command == "train":
        result, report, stats = train(args.data, args.artifacts)
        print(json.dumps({"load": stats, "metrics": result}, indent=2))
        return 0
    if args.command == "evaluate":
        print(Path(args.metrics).read_text(encoding="utf-8"))
        return 0
    model = LogRiskModel.load(args.model)
    if args.command == "predict":
        rows = [{"service": args.service, "environment": args.environment, "log_level": args.log_level, "message": args.message, "status_code": "" if args.status_code is None else str(args.status_code), "dependency": args.dependency or "", "exception_type": ""}]
    else:
        rows = json.loads(Path(args.input).read_text(encoding="utf-8")); rows = rows if isinstance(rows, list) else [rows]
    for row, prediction in zip(rows, model.predict(rows)):
        print(json.dumps({"input": row, **prediction.__dict__}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
