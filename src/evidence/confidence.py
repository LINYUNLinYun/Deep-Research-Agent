"""Idempotent report confidence calibration from attributable evidence."""
def calibrate(report, summary: dict) -> None:
    if not summary.get("total_claims"):
        return
    if report.confidence_basis is None:
        report.confidence_basis = report.confidence
    report.confidence = round(max(0.0, min(1.0,
        report.confidence_basis * (0.5 + 0.5 * summary.get("support_rate", 0.0)))), 2)
