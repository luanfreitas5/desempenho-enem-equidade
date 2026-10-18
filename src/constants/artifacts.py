"""Nomes de arquivos gerados pelas etapas do pipeline."""

from typing import Final

GOLD_FILE: Final[str] = "enem_gold.parquet"
GOLD_MANIFEST_FILE: Final[str] = "enem_gold.manifest.json"
TEST_PREDICTIONS_FILE: Final[str] = "test_predictions.parquet"

LINEAR_MODEL_NAME: Final[str] = "linear_baseline"
GRADIENT_BOOSTING_MODEL_NAME: Final[str] = "gradient_boosting"
MODEL_NAMES: Final[tuple[str, ...]] = (LINEAR_MODEL_NAME, GRADIENT_BOOSTING_MODEL_NAME)
MODEL_FILES: Final[dict[str, str]] = {
    LINEAR_MODEL_NAME: "linear_baseline.joblib",
    GRADIENT_BOOSTING_MODEL_NAME: "gradient_boosting.joblib",
}
PREDICTION_PREFIX: Final[str] = "pred_"

SILVER_MANIFEST_FILE: Final[str] = "manifest.json"
