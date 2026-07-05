"""Run inference with a trained Stage 1 model."""

from src.config import load_config
from src.config_schema import AppConfig


def predict(cfg: AppConfig | None = None) -> None:
    """Run Stage 1 inference for a new batch of patients.

    Args:
        cfg: validated project config. Loaded from disk if ``None``.

    Raises:
        NotImplementedError: this function is a placeholder for future work.
    """
    cfg = cfg or load_config()
    raise NotImplementedError("Prediction not yet implemented.")


if __name__ == "__main__":
    predict()
