"""
Temporal Features — Time-of-Day & Seasonality Encoding
========================================================
Encodes cyclical time information for intraday trading.
"""

import numpy as np
import pandas as pd
from loguru import logger


class TemporalFeatures:
    """Computes time-based features for intraday data."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.debug("  Temporal: index is not DatetimeIndex, skipping")
            return df

        idx = df.index
        hours = idx.hour + idx.minute / 60.0
        dow = idx.dayofweek
        dom = idx.day
        minutes_of_day = idx.hour * 60 + idx.minute

        # Cyclical encodings
        df["Hour_Sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["Hour_Cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["DayOfWeek_Sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["DayOfWeek_Cos"] = np.cos(2 * np.pi * dow / 7.0)
        df["DayOfWeek"] = dow.astype(float)
        df["DayOfMonth_Sin"] = np.sin(2 * np.pi * dom / 31.0)
        df["DayOfMonth_Cos"] = np.cos(2 * np.pi * dom / 31.0)

        # Session flags
        df["Is_Market_Open"] = (((idx.hour == 9) & (idx.minute >= 30)) | ((idx.hour == 10) & (idx.minute < 0))).astype(float)
        df["Is_Opening_Range"] = (((idx.hour == 9) & (idx.minute >= 30)) | ((idx.hour == 10) & (idx.minute < 15))).astype(float)
        df["Is_Lunch_Hour"] = (idx.hour == 12).astype(float)
        df["Is_Power_Hour"] = (idx.hour == 15).astype(float)
        df["Is_Closing"] = ((idx.hour == 15) & (idx.minute >= 45)).astype(float)
        df["Is_PreMarket"] = (((idx.hour >= 4) & (idx.hour < 9)) | ((idx.hour == 9) & (idx.minute < 30))).astype(float)
        df["Is_PostMarket"] = ((idx.hour >= 16) & (idx.hour < 20)).astype(float)

        # Minutes since open / until close (normalized)
        mso = np.clip(minutes_of_day - 570, 0, 390).astype(float)
        muc = np.clip(960 - minutes_of_day, 0, 390).astype(float)
        df["Minutes_Since_Open"] = mso / 390.0
        df["Minutes_Until_Close"] = muc / 390.0

        # Trading session (1=Asian, 2=London, 3=NY)
        session = np.zeros(len(idx), dtype=float)
        for i, h in enumerate(idx.hour):
            if h >= 19 or h < 3:
                session[i] = 1.0
            elif 3 <= h < 9:
                session[i] = 2.0
            elif 9 <= h < 16:
                session[i] = 3.0
        df["Trading_Session"] = session

        # Session overlap, day flags
        df["Is_Session_Overlap"] = (((idx.hour == 9) & (idx.minute >= 30)) | (idx.hour == 10)).astype(float)
        df["Is_Monday"] = (dow == 0).astype(float)
        df["Is_Friday"] = (dow == 4).astype(float)
        df["Is_Month_End"] = (dom >= 26).astype(float)

        return df

    def get_feature_names(self) -> list:
        return [
            "Hour_Sin", "Hour_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos", "DayOfWeek",
            "DayOfMonth_Sin", "DayOfMonth_Cos", "Is_Market_Open", "Is_Opening_Range",
            "Is_Lunch_Hour", "Is_Power_Hour", "Is_Closing", "Is_PreMarket", "Is_PostMarket",
            "Minutes_Since_Open", "Minutes_Until_Close", "Trading_Session",
            "Is_Session_Overlap", "Is_Monday", "Is_Friday", "Is_Month_End",
        ]
