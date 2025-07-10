import pandas as pd


def format_price_dynamically(price: float) -> str:
    """
    Dynamically formats the price based on its value to ensure appropriate precision.
    """
    if price is None or not isinstance(price, (int, float)) or pd.isna(price):
        return "N/A"

    if price >= 100:
        return f"{price:.2f}"
    elif price >= 1:
        return f"{price:.3f}"
    elif price >= 0.01:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"


def format_value(value, precision=2):
    """Generic formatter for non-price floats or other types."""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, (int, str)):
        return str(value)
    return "N/A"