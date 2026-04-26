from pycmqlib3.data.ifind.ifind_data import IfindDatafeed
from pycmqlib3.core.trading_object import HistoryRequest
from pycmqlib3.core.trading_const import Exchange, Interval
import pandas as pd

ifind_api = IfindDatafeed()

def read_edb_data(
        indicators,
        start_date,
        end_date):
    df = ifind_api.query_edb_data(indicators, start_date, end_date)
    # pivot = df.pivot(index='time',
    #                  columns=['id', 'index_name'],
    #                  values='value')
    return df


def read_bar_data(
        instID: str,
        exch: str,
        interval: str,
        start_date: str,
        end_date: str,
    ):
    req = HistoryRequest(
        symbol=instID,
        exchange=Exchange(exch),
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        interval=Interval(interval),
    )
    bars = ifind_api.query_bar_history(req)
    return bars
