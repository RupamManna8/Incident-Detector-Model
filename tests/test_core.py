from src.config import CLASSES
from src.data.loader import load_records
from src.data.validator import validate_records


def test_supplied_dataset_validates_and_repairs_legacy_rows():
    records, stats = load_records("dataset.csv")
    report = validate_records(records, stats)
    assert report.valid
    assert stats["repaired_rows"] == 232
    assert tuple(report.class_counts) == CLASSES


def test_invalid_status_is_reported():
    report = validate_records([{"service": "x", "environment": "prod", "log_level": "ERROR", "message": "x", "status_code": "nope", "dependency": "", "risk_level": "LOW"}])
    assert not report.valid
    assert any("status_code" in error for error in report.errors)
