"""Optional reporting hook. No reporting is performed by the public benchmark."""

def report_evaluation(task_config: dict) -> None:
    """Compatibility hook for applications that provide their own reporting."""
    return None
