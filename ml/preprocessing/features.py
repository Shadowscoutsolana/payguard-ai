"""
PaySim preprocessing and the shared feature definition used by BOTH training and live scoring.

Dataset: PaySim (Lopez-Rojas et al.), Kaggle "ealaxi/paysim1", CC BY-SA 4.0.
It is SIMULATED mobile-money data, not PayPal data.

Design decisions (measured on the full 6,362,620-row file):
- Only TRANSFER and CASH_OUT rows are used: they contain all 8,213 fraud cases.
- Balance columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) are EXCLUDED:
  97.8% of fraud rows have amount == oldbalanceOrg vs 0.000% of legitimate rows, so they leak
  the label, and a live PayPal payment does not expose the buyer's balance anyway.
- step (simulation hour) is used only for the time-based split, not as a feature: there is no
  verified mapping from PaySim simulation time to real clock time.
- Sender history is not learnable (only 0.15% of PaySim senders appear more than once).

Features (identical names and order at training and inference):
  log_amount             log1p(amount)
  is_transfer            1 for TRANSFER, 0 for CASH_OUT (a PayPal P2P payment maps to TRANSFER)
  recipient_prior_count  log1p(number of payments the recipient received in STRICTLY earlier hours)
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["log_amount", "is_transfer", "recipient_prior_count"]
LABEL_COLUMN = "isFraud"
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets"
DEFAULT_PAYSIM_FILE = DATASET_DIR / "PS_20174392719_1491204439457_log.csv"

# Time-based split on PaySim's simulation hour (1..743): train on the past, test on the future.
TRAIN_MAX_STEP = 400
VALID_MAX_STEP = 500


def load_paysim(path: Path = DEFAULT_PAYSIM_FILE) -> pd.DataFrame:
    """Loads PaySim and returns TRANSFER/CASH_OUT rows with step, features and label."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and place it in ml/datasets/."
        )
    df = pd.read_csv(path, usecols=["step", "type", "amount", "nameDest", LABEL_COLUMN],
                     dtype={"type": "category"})

    # Recipient's prior receipts across ALL transaction types, counted only from strictly
    # earlier hours, so no information from the same or later hours leaks into a row.
    per_hour = (df.groupby(["nameDest", "step"]).size().rename("n").reset_index()
                .sort_values(["nameDest", "step"]))
    per_hour["prior"] = per_hour.groupby("nameDest")["n"].cumsum() - per_hour["n"]
    df = df.merge(per_hour[["nameDest", "step", "prior"]], on=["nameDest", "step"], how="left")

    df = df[df["type"].isin(FRAUD_TYPES)].copy()
    df["log_amount"] = np.log1p(df["amount"].astype(float))
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["recipient_prior_count"] = np.log1p(df["prior"].astype(float))
    return df[["step"] + FEATURE_COLUMNS + [LABEL_COLUMN]].reset_index(drop=True)


def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train: steps 1-400 · Validation (early stopping): 401-500 · Test: 501-743."""
    train = df[df["step"] <= TRAIN_MAX_STEP]
    valid = df[(df["step"] > TRAIN_MAX_STEP) & (df["step"] <= VALID_MAX_STEP)]
    test = df[df["step"] > VALID_MAX_STEP]
    return train, valid, test


def build_live_features(amount: float, recipient_prior_receipts: int, is_transfer: int = 1) -> pd.DataFrame:
    """
    The same three features for a live payment, as a named single-row DataFrame
    (named columns avoid the 'X does not have valid feature names' warning).
    """
    return pd.DataFrame([{
        "log_amount": math.log1p(max(float(amount), 0.0)),
        "is_transfer": int(is_transfer),
        "recipient_prior_count": math.log1p(max(int(recipient_prior_receipts), 0)),
    }], columns=FEATURE_COLUMNS)
