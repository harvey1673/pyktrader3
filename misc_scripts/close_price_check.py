"""Recent futures close-price validation used by the factor update job."""

import logging

import pandas as pd

from pycmqlib3.utility.email_tool import send_html_by_smtp
from pycmqlib3.utility.sec_bits import EMAIL_NOTIFY, EMAIL_QQ, LOCAL_PC_NAME, NOTIFIERS


def find_recent_close_price_nans(price_df, run_date, lookback_days=10):
    """Return every missing close in the recent inspection window.

    The inspection window is the last ``lookback_days`` rows on or before
    ``run_date``, including leading NaNs and entirely missing products.
    """
    if not isinstance(price_df, pd.DataFrame):
        raise TypeError("price_df must be a pandas DataFrame")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if price_df.empty:
        raise ValueError("price_df is empty")

    if isinstance(price_df.columns, pd.MultiIndex):
        if "field" in price_df.columns.names:
            field_level = price_df.columns.names.index("field")
        else:
            candidate_levels = [
                level
                for level in range(price_df.columns.nlevels)
                if "close" in set(price_df.columns.get_level_values(level))
            ]
            if len(candidate_levels) != 1:
                raise ValueError("Unable to identify the close-price column level")
            field_level = candidate_levels[0]
        if "close" not in set(price_df.columns.get_level_values(field_level)):
            raise ValueError("price_df has no close-price columns")
        close_df = price_df.xs("close", axis=1, level=field_level)
    else:
        close_columns = [column for column in price_df.columns if column == "close"]
        if not close_columns:
            raise ValueError("price_df has no close-price columns")
        close_df = price_df.loc[:, close_columns]

    close_df = close_df.copy()
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()
    close_df = close_df.loc[close_df.index.normalize() <= pd.Timestamp(run_date)]
    if close_df.empty:
        raise ValueError(f"price_df has no rows on or before {run_date}")

    recent_close = close_df.tail(lookback_days)
    missing_mask = recent_close.isna()
    missing_rows = []
    for date_value, row in missing_mask.iterrows():
        for product in row.index[row.to_numpy()]:
            missing_rows.append(
                {"date": pd.Timestamp(date_value), "product": str(product)}
            )
    return pd.DataFrame(missing_rows, columns=["date", "product"])


def check_recent_close_prices(
    price_df, run_date, lookback_days=10, email_notify=EMAIL_NOTIFY
):
    """Check recent closes and email an alarm when any values are NaN."""
    missing = find_recent_close_price_nans(
        price_df,
        run_date=run_date,
        lookback_days=lookback_days,
    )
    if missing.empty:
        logging.info(
            "recent close-price check passed for the last %s rows",
            lookback_days,
        )
        return missing

    missing = missing.copy()
    missing["date"] = pd.to_datetime(missing["date"]).dt.strftime("%Y-%m-%d")
    logging.error(
        "recent close-price check found %s NaN values:\n%s",
        len(missing),
        missing.to_string(index=False),
    )
    if email_notify:
        date_summary = (
            missing.groupby("date")["product"]
            .agg(lambda values: ", ".join(sorted(values)))
            .reset_index(name="products_with_nan_close")
        )
        subject = (
            f"{LOCAL_PC_NAME} futures close price NaN alarm"
            f"<{pd.Timestamp(run_date).strftime('%Y.%m.%d')}>"
        )
        html = (
            "<html><head></head><body>"
            f"<p>Found <strong>{len(missing)}</strong> missing close-price "
            f"observations within the latest {lookback_days} rows on or before "
            "the factor update date, including leading NaNs and products with "
            "no valid closes in the window.</p>"
            "<p>Missing products by date:</p>"
            f"{date_summary.to_html(index=False, escape=True)}"
            "<p>Detailed missing observations:</p>"
            f"{missing.to_html(index=False, escape=True)}"
            "</body></html>"
        )
        sent = send_html_by_smtp(EMAIL_QQ, NOTIFIERS, subject, html)
        if not sent:
            logging.error("failed to send the close-price NaN alarm email")
    return missing
