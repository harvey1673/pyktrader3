import json
import re
from functools import wraps, lru_cache
from datetime import datetime, timedelta
from pycmqlib3.core.trading_const import Exchange

TEMP_DIR = ""

def save_json(filename: str, data: dict) -> None:
    """
    Save data into json file in temp path.
    """
    filepath = TEMP_DIR + filename
    with open(filepath, mode="w+", encoding="UTF-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

def load_json(filename: str, auto_save: bool = True) -> dict:
    """
    Load data from json file in temp path.
    """
    filepath = TEMP_DIR + filename

    try:
        with open(filepath, mode="r", encoding="UTF-8") as f:
            data = json.load(f)
        return data
    except:
        if auto_save:
            save_json(filename, {})
        return {}

@lru_cache()
def get_underlying_symbol(symbol: str):
    """
    取得合约的短号.  rb2005 => rb
    :param symbol:
    :return: 短号
    """
    # 套利合约
    if symbol.find(' ') != -1:
        # 排除SP SPC SPD
        s = symbol.split(' ')
        if len(s) < 2:
            return symbol
        symbol = s[1]

        # 只提取leg1合约
        if symbol.find('&') != -1:
            s = symbol.split('&')
            if len(s) < 2:
                return symbol
            symbol = s[0]

    p = re.compile(r"([A-Z]+)[0-9]+", re.I)
    underlying_symbol = p.match(symbol)

    if underlying_symbol is None:
        return symbol

    return underlying_symbol.group(1)

@lru_cache()
def get_full_symbol(symbol: str):
    """
    获取全路径得合约名称, MA005 => MA2005, j2005 => j2005
    """
    if symbol.endswith('SPD'):
        return symbol

    underlying_symbol = get_underlying_symbol(symbol)
    if underlying_symbol == symbol:
        return symbol

    symbol_month = symbol.replace(underlying_symbol, '')
    if len(symbol_month) == 3:
        # 支持2020年合约
        return '{0}2{1}'.format(underlying_symbol, symbol_month)
    else:
        return symbol

def get_trading_date(dt: datetime = None):
    """
    根据输入的时间，返回交易日的日期
    :param dt:
    :return:
    """
    if dt is None:
        dt = datetime.now()

    if dt.isoweekday() in [6, 7]:
        # 星期六,星期天=>星期一
        return (dt + timedelta(days=8 - dt.isoweekday())).strftime('%Y-%m-%d')

    if dt.hour >= 20:
        if dt.isoweekday() == 5:
            # 星期五=》星期一
            return (dt + timedelta(days=3)).strftime('%Y-%m-%d')
        else:
            # 第二天
            return (dt + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        return dt.strftime('%Y-%m-%d')

def get_real_symbol_by_exchange(full_symbol, vn_exchange):
    """根据交易所，返回真实合约"""
    if vn_exchange == Exchange.CFFEX:
        return full_symbol.upper()

    if vn_exchange in [Exchange.DCE, Exchange.SHFE, Exchange.INE]:
        return full_symbol.lower()

    if vn_exchange == Exchange.CZCE:
        underlying_symbol = get_underlying_symbol(full_symbol).upper()
        yearmonth_len = len(full_symbol) - len(underlying_symbol) - 1
        return underlying_symbol.upper() + full_symbol[-yearmonth_len:]

    return full_symbol