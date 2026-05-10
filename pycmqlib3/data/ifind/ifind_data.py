from datetime import timedelta, datetime
from typing import List, Optional, Sequence
from pytz import timezone
import pandas as pd

from iFinDPy import (
    THS_iFinDLogin,
    THS_HQ,
    THS_HF,
    THSData,
    THS_EDB
)

from pycmqlib3.utility.sec_bits import ifind_user
from pycmqlib3.core.trading_object import HistoryRequest, BarData
from pycmqlib3.core.trading_const import Exchange, Interval


CHINA_TZ = timezone("Asia/Shanghai")

EXCHANGE_MAP = {
    Exchange.SSE: "SH",
    Exchange.SZSE: "SZ",
    Exchange.CFFEX: "CFE",
    Exchange.SHFE: "SHF",
    Exchange.CZCE: "CZC",
    Exchange.DCE: "DCE",
}

INTERVAL_MAP = {
    Interval.MINUTE: "1",
    Interval.HOUR: "60"
}

SHIFT_MAP = {
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
}


def _normalize_indicators(indicators: str | Sequence[str]) -> str:
    """Convert indicator input to the comma-separated format expected by iFinD."""
    if isinstance(indicators, str):
        return indicators.strip()

    cleaned = [item.strip() for item in indicators if str(item).strip()]
    if not cleaned:
        raise ValueError("indicators cannot be empty")
    return ";".join(cleaned)


class IfindDatafeed(object):
    """同花顺iFinD数据服务接口"""

    def __init__(self) -> None:
        """"""
        self.username: str = ifind_user["user"]
        self.password: str = ifind_user["pwd"]
        self.inited = False

    def init(self) -> bool:
        """初始化"""
        if self.inited:
            return True

        code: int = THS_iFinDLogin(self.username, self.password)
        if code not in [-201, 0]:  # -201表示已登录，0表示登录成功
            print(f"iFinD登录失败，错误代码：{code}")
            return False
        return True

    def query_bar_history(self, req: HistoryRequest) -> Optional[List[BarData]]:
        """查询K线数据"""
        # 检查是否登录
        if not self.inited:
            self.init()

        # 生成iFinD合约代码
        ifind_exchange: str = EXCHANGE_MAP[Exchange(req.exchange)]
        ifind_symbol: str = f"{req.symbol}.{ifind_exchange}"

        # 计算时间戳平移值
        shift: timedelta = SHIFT_MAP.get(req.interval, None)

        # 查询数据内容
        indicators: str = "open;high;low;close;volume;amount;openInterest"

        # 日线数据
        if req.interval == Interval.DAILY:
            params: str = "Fill:Original"
            result: THSData = THS_HQ(
                ifind_symbol,
                indicators,
                params,
                req.start.strftime("%Y-%m-%d %H:%M:%S"),
                req.end.strftime("%Y-%m-%d %H:%M:%S"),
            )
        # 日内数据
        else:
            # 生成iFinD数据周期
            ifind_interval: str = INTERVAL_MAP[req.interval]
            params: str = f"Fill:Original,Interval:{ifind_interval}"

            result: THSData = THS_HF(
                ifind_symbol,
                indicators,
                params,
                req.start.strftime("%Y-%m-%d %H:%M:%S"),
                req.end.strftime("%Y-%m-%d %H:%M:%S"),
            )

        # 如果报错则直接返回空值
        if result.errorcode:
            return []

        # 解析成K线数据
        # bars: List[BarData] = []

        # for tp in result.data.itertuples():
        #     # 生成时间戳
        #     if ":" in tp.time:
        #         dt = datetime.strptime(tp.time, "%Y-%m-%d %H:%M")
        #     else:
        #         dt = datetime.strptime(tp.time, "%Y-%m-%d")

        #     # 检查时间戳平移
        #     if shift:
        #         dt -= shift

        #     # 获取持仓量
        #     if tp.openInterest:
        #         open_interest = tp.openInterest
        #     else:
        #         open_interest = 0

        #     # 生成K线对象
        #     bar = BarData(
        #         symbol=req.symbol,
        #         exchange=req.exchange,
        #         datetime=CHINA_TZ.localize(dt),
        #         interval=req.interval,
        #         open=tp.open,
        #         high=tp.high,
        #         low=tp.low,
        #         close=tp.close,
        #         volume=tp.volume,
        #         #turnover=tp.amount,
        #         openInterest=open_interest,
        #         gateway_name="IFIND"
        #     )
        #     bars.append(bar)

        return result.data


    def query_edb_data(self, indicators: str, start_date, end_date):
        """查询EDB数据"""
        # 检查是否登录
        if not self.inited:
            self.init()

        indicator_str = _normalize_indicators(indicators)
        result = THS_EDB(
            indicator_str,
            "",
            pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            pd.to_datetime(end_date).strftime("%Y-%m-%d"),
            format="format:dataframe",
        )

        # 如果报错则直接返回空值
        if result.errorcode != 0:
            print(f"iFinD查询EDB数据失败，错误代码：{result.errorcode} {result.errmsg}")
            return None

        data = result.data.copy()
        if "time" in data.columns:
            data["time"] = pd.to_datetime(data["time"], errors="coerce")
        return data
