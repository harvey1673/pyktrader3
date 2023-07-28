import datetime
import re, requests
from io import StringIO, BytesIO
import pandas as pd
from pycmqlib3.utility.misc import inst2product
from pycmqlib3.utility.dbaccess import save_data
from pycmqlib3.utility.misc import CHN_Holidays
import akshare as ak
CFFEX_OPTION_URL_300 = "http://www.cffex.com.cn/quote_IO.txt"
# 深圳证券交易所
SZ_OPTION_URL_300 = "http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=ysplbrb&TABKEY=tab1&random=0.10432465776720479"
# 上海证券交易所
SH_OPTION_URL_50 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510050"
SH_OPTION_URL_KING_50 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510050_{}"
SH_OPTION_URL_300 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510300"
SH_OPTION_URL_KING_300 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510300_{}"
SH_OPTION_URL_500 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510500"
SH_OPTION_URL_KING_500 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510500_{}"
SH_OPTION_URL_KC_50 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/588000"
SH_OPTION_URL_KC_KING_50 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/588000_{}"
SH_OPTION_URL_KC_50_YFD = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/588080"
SH_OPTION_URL_KING_50_YFD = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/588080_{}"
SH_OPTION_PAYLOAD = {
    "select": "select: code,name,last,change,chg_rate,amp_rate,volume,amount,prev_close"
}
SH_OPTION_PAYLOAD_OTHER = {
    "select": "contractid,last,chg_rate,presetpx,exepx"
}
# 大连商品交易所
DCE_OPTION_URL = "http://www.dce.com.cn/publicweb/quotesdata/dayQuotesCh.html"
DCE_DAILY_OPTION_URL = "http://www.dce.com.cn/publicweb/quotesdata/exportDayQuotesChData.html"
# 上海期货交易所
SHFE_OPTION_URL = "http://www.shfe.com.cn/data/dailydata/option/kx/kx{}.dat"
# 郑州商品交易所
CZCE_DAILY_OPTION_URL_3 = "http://www.czce.com.cn/cn/DFSStaticFiles/Option/{}/{}/OptionDataDaily.txt"
# PAYLOAD
SHFE_HEADERS = {"User-Agent": "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)"}
DATE_PATTERN = re.compile(r"^([0-9]{4})[-/]?([0-9]{2})[-/]?([0-9]{2})")
OPTION_COLUMNS = ['date', 'instID', 'product', 'underlying', 'strike', 'option_type',
                  'open', 'high', 'low', 'close', 'pre_settle', 'settle', 'px_chg',
                  'delta', 'volume', 'oi', 'oi_chg', 'turnover', 'exec_volume']


def fix_nb_func(num, out_type='int', na_value='0'):
    s = str(num)
    s = s.replace(",", "")
    if len(s) == 0:
        s = na_value
    if out_type == 'int':
        res = int(s)
    else:
        res = float(s)
    return res


def fix_table_data(opt_df):
    opt_df['product'] = opt_df['product'].str.strip()
    opt_df['underlying'] = opt_df['underlying'].str.strip()
    opt_df['instID'] = opt_df['instID'].str.strip()
    opt_df['volume'] = opt_df['volume'].apply(lambda x: fix_nb_func(x, 'int'))
    opt_df['oi'] = opt_df['oi'].apply(lambda x: fix_nb_func(x, 'int'))
    opt_df['oi_chg'] = opt_df['oi_chg'].apply(lambda x: fix_nb_func(x, 'int'))
    opt_df['date'] = pd.to_datetime(opt_df['date'])
    return opt_df


def get_dce_option_data(tday=datetime.date.today()):
    url = DCE_DAILY_OPTION_URL
    payload = {
        "dayQuotes.variety": "all",
        "dayQuotes.trade_type": "1",
        "year": str(tday.year),
        "month": str(tday.month - 1),
        "day": str(tday.day),
        "exportFlag": "excel",
    }
    nb_tries = 3
    while nb_tries > 0:
        try:
            res = requests.post(url, data=payload, timeout=60)
            table_df = pd.read_excel(BytesIO(res.content), header=0)
        except Exception as e:
            print('timeout due to exception %s' % s)
            nb_tries -= 1
    if nb_tries == 0:
        return pd.DataFrame(), pd.DataFrame()
    atmvol_df = table_df.iloc[
                 table_df[table_df.iloc[:, 0].str.contains("合约")].iloc[-1].name:,
                 [0, 1],
                 ]
    atmvol_df.reset_index(inplace=True, drop=True)
    atmvol_df.iloc[0] = atmvol_df.iat[0, 0].split("\t")
    atmvol_df.columns = atmvol_df.iloc[0]
    atmvol_df = atmvol_df.iloc[1:, :]
    atmvol_df.columns = ['contract', 'atm']
    table_df = table_df.dropna(subset=['Delta'])
    table_df.columns = ['product_name', 'instID', 'open', 'high', 'low', 'pre_settle', 'settle',
                        'px_chg', 'px_chg1', 'delta', 'volume', 'oi', 'oi_chg', 'turnover', 'exec_volume']
    table_df['underlying'] = table_df['instID'].apply(lambda x: x.split('-')[0])
    table_df['option_type'] = table_df['instID'].apply(lambda x: x.split('-')[1])
    table_df['strike'] = table_df['instID'].apply(lambda x: x.split('-')[2])
    table_df['product'] = table_df['underlying'].apply(lambda x: inst2product(x))
    table_df['date'] = tday
    atmvol_df['date'] = tday
    atmvol_df.columns = ['date', 'undelying', 'atmvol']
    table_df = table_df[OPTION_COLUMNS]
    table_df = fix_table_data(table_df)
    return table_df, atmvol_df


def get_czce_option_data(tday=datetime.date.today()):
    url = CZCE_DAILY_OPTION_URL_3.format(tday.strftime("%Y"), tday.strftime("%Y%m%d"))
    try:
        r = requests.get(url)
        f = StringIO(r.text)
        table_df = pd.read_table(f, encoding="utf-8", skiprows=1, sep="|")
    except:
        return


def get_shfe_option_data(tday=datetime.date.today()):
    url = SHFE_OPTION_URL.format(day.strftime("%Y%m%d"))
    r = requests.get(url, headers=SHFE_HEADERS)
    json_data = r.json()
    table_df = pd.DataFrame(
        [
            row
            for row in json_data["o_curinstrument"]
            if row["INSTRUMENTID"] not in ["小计", "合计"]
               and row["INSTRUMENTID"] != ""
        ]
    )
    table_df.columns = [
        'option_product', 'product_id', 'product_name', 'instID',
        'pre_settle', 'open', 'high', 'low', 'close', 'settle',
        'px_chg', 'px_chg1', 'volume', 'oi', 'oi_chg', 'nb_orders',
        'exec_volume', 'turnover', 'delta',
        'underlying', 'strike', 'option_type', 'product',
    ]
    table_df['option_type'] = table_df['option_type'].apply(lambda x: 'C' if int(x)==1 else 'P')
    vol_df = pd.DataFrame(json_data["o_cursigma"])
    vol_df.columns = ['option_product', 'product_id', 'product_name', 'underlying',
                      'volume', 'oi', 'oi_chg', 'exec_volume', 'turnover', 'atmvol', 'product']
    table_df['date'] = tday
    vol_df['date'] = tday
    table_df = table_df[OPTION_COLUMNS]
    vol_df = vol_df[['date', 'underlying', 'atmvol']]
    vol_df = vol_df[vol_df['atmvol'].apply(lambda x: len(str(x)) > 0)]
    table_df = fix_table_data(table_df)
    return table_df, vol_df


def option_gfex_daily(symbol: str = "工业硅", trade_date: str = "20230418"):
    """
    广州期货交易所-日频率-量价数据
    广州期货交易所: 工业硅(上市时间: 20221222)
    http://www.gfex.com.cn/gfex/rihq/hqsj_tjsj.shtml
    :param trade_date: 交易日
    :type trade_date: str
    :param symbol: choice of {"工业硅"}
    :type symbol: str
    :return: 日频行情数据
    :rtype: pandas.DataFrame
    """
    calendar = get_calendar()
    day = convert_date(trade_date) if trade_date is not None else datetime.date.today()
    if day.strftime("%Y%m%d") not in calendar:
        warnings.warn("%s非交易日" % day.strftime("%Y%m%d"))
        return
    symbol_map = {"工业硅": 1}
    url = "http://www.gfex.com.cn/u/interfacesWebTiDayQuotes/loadList"
    payload = {"trade_date": day.strftime("%Y%m%d"), "trade_type": symbol_map[symbol]}
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Content-Length": "32",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Host": "www.gfex.com.cn",
        "Origin": "http://www.gfex.com.cn",
        "Pragma": "no-cache",
        "Proxy-Connection": "keep-alive",
        "Referer": "http://www.gfex.com.cn/gfex/rihq/hqsj_tjsj.shtml",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "content-type": "application/x-www-form-urlencoded",
    }
    r = requests.post(url, data=payload, headers=headers)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json["data"])
    temp_df.rename(
        columns={
            "variety": "商品名称",
            "diffI": "持仓量变化",
            "high": "最高价",
            "turnover": "成交额",
            "impliedVolatility": "隐含波动率",
            "diff": "涨跌",
            "delta": "Delta",
            "close": "收盘价",
            "diff1": "涨跌1",
            "lastClear": "前结算价",
            "open": "开盘价",
            "matchQtySum": "行权量",
            "delivMonth": "合约名称",
            "low": "最低价",
            "clearPrice": "结算价",
            "varietyOrder": "品种代码",
            "openInterest": "持仓量",
            "volumn": "成交量",
        },
        inplace=True,
    )
    temp_df = temp_df[
        [
            "商品名称",
            "合约名称",
            "开盘价",
            "最高价",
            "最低价",
            "收盘价",
            "前结算价",
            "结算价",
            "涨跌",
            "涨跌1",
            "Delta",
            "成交量",
            "持仓量",
            "持仓量变化",
            "成交额",
            "行权量",
            "隐含波动率",
        ]
    ]
    return temp_df


def option_gfex_vol_daily(symbol: str = "工业硅", trade_date: str = "20230418"):
    """
    广州期货交易所-日频率-合约隐含波动率
    广州期货交易所: 工业硅(上市时间: 20221222)
    http://www.gfex.com.cn/gfex/rihq/hqsj_tjsj.shtml
    :param symbol: choice of {"工业硅"}
    :type symbol: str
    :param trade_date: 交易日
    :type trade_date: str
    :return: 日频行情数据
    :rtype: pandas.DataFrame
    """
    calendar = get_calendar()
    day = convert_date(trade_date) if trade_date is not None else datetime.date.today()
    if day.strftime("%Y%m%d") not in calendar:
        warnings.warn("%s非交易日" % day.strftime("%Y%m%d"))
        return
    symbol_map = {"工业硅": 1}
    symbol_map[symbol]  # 占位
    url = "http://www.gfex.com.cn/u/interfacesWebTiDayQuotes/loadListOptVolatility"
    payload = {"trade_date": day.strftime("%Y%m%d")}
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Content-Length": "32",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Host": "www.gfex.com.cn",
        "Origin": "http://www.gfex.com.cn",
        "Pragma": "no-cache",
        "Proxy-Connection": "keep-alive",
        "Referer": "http://www.gfex.com.cn/gfex/rihq/hqsj_tjsj.shtml",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "content-type": "application/x-www-form-urlencoded",
    }
    r = requests.post(url, data=payload, headers=headers)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json["data"])
    temp_df.rename(
        columns={
            "seriesId": "合约系列",
            "varietyId": "-",
            "hisVolatility": "隐含波动率",
        },
        inplace=True,
    )
    temp_df = temp_df[
        [
            "合约系列",
            "隐含波动率",
        ]
    ]
    return temp_df


def load_opt_by_exch(exch='DCE', start_date='20210101', end_date='20230705'):
    load_dates = [d for d in pd.date_range(start=start_date, end=end_date, freq='B') if d not in CHN_Holidays]
    failed_dates = []
    if exch == 'DCE':
        get_exch_option_func = get_dce_option_data
    elif exch == 'SHFE':
        get_exch_option_func = get_shfe_option_data
    else:
        return []
    for d in load_dates:
        try:
            table_df, vol_df = get_exch_option_func(tday=d)
            if len(table_df) == 0:
                failed_dates.append(d)
            else:
                save_data('opt_daily', table_df)
                save_data('opt_vol', vol_df)
                print('date=%s is done' % d)
        except Exception as e:
            print("exception: %s on %s, continue..." % (e, d))
            failed_dates.append(d)
    return failed_dates

