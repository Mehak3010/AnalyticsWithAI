from typing import Any

import pandas as pd

from .config import Config


class Ingestion:
    def __init__(self, config: Config):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        path = self.config.dataset.path
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        # Parse dates
        df[self.config.dataset.date_column] = pd.to_datetime(df[self.config.dataset.date_column])

        # Optional monitoring window
        date_col = self.config.dataset.date_column
        if self.config.dataset.start_date:
            df = df[df[date_col] >= pd.Timestamp(self.config.dataset.start_date)]
        if self.config.dataset.end_date:
            df = df[df[date_col] < pd.Timestamp(self.config.dataset.end_date) + pd.Timedelta(days=1)]

        # Sort by date
        df = df.sort_values(date_col)

        return df

    def validate_and_clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        report = {}

        # 1. Missing dates (reindex to full calendar)
        # This will be done per series in the metric engine, but let's check overall range
        date_col = self.config.dataset.date_column
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        report["date_range"] = (min_date, max_date)

        # 2. Duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        report["duplicates_removed"] = initial_rows - len(df)

        # 3. Text-formatted numbers and negative values
        # (Assuming they are already numeric from read_csv, but let's ensure)
        for metric in self.config.metrics:
            if metric.column:
                df[metric.column] = pd.to_numeric(df[metric.column], errors="coerce")

            if metric.numerator and metric.numerator.column:
                df[metric.numerator.column] = pd.to_numeric(df[metric.numerator.column], errors="coerce")

            if metric.denominator and metric.denominator.column:
                df[metric.denominator.column] = pd.to_numeric(df[metric.denominator.column], errors="coerce")

        # 4. Data Quality Report
        report["null_rates"] = df.isnull().mean().to_dict()

        # Fail loudly if fewer than min_history points
        # (This check is more relevant after aggregation, but let's do a basic check here)
        unique_dates = df[date_col].nunique()
        if unique_dates < self.config.detection.min_history:
            raise ValueError(
                f"Insufficient data: {unique_dates} unique dates, minimum required: {self.config.detection.min_history}"
            )

        return df, report
