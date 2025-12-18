import pandas as pd


def log(message: str) -> None:
    """Simple structured logger (stdout)."""
    print(f"[INFO] {message}")


def read_csv_safe(path, **kwargs) -> pd.DataFrame:
    """Read CSV with basic safety checks."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)
