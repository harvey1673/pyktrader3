# encoding: UTF-8

# 从tdx下载股票数据.
# 收盘后的数据基本正确, 但盘中实时拿数据时:
# 1. 1Min的Bar可能不是最新的, 会缺几分钟.
# 2. 当周期>1Min时, 最后一根Bar可能不是完整的, 强制修改后
#    - 5min修改后freq基本正确
#    - 1day在VNPY合成时不关心已经收到多少Bar, 所以影响也不大
#    - 但其它分钟周期因为不好精确到每个品种, 修改后的freq可能有错
#   https://rainx.gitbooks.io/pytdx/content/pytdx_hq.html
# 华富资产

import sys
import os
import pickle
import bz2
import traceback
import pandas as pd
import random
from time import sleep

from datetime import datetime, timedelta
from logging import ERROR
from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams

from pycmqlib3.core.trading_const import Exchange
from pycmqlib3.core.trading_object import BarData
from pycmqlib3.data.tdx.tdx_common import (
    PERIOD_MAPPING,
    get_tdx_market_code,
    get_cache_config,
    get_cache_json,
    save_cache_config,
    get_stock_type,
    TDX_STOCK_CONFIG,
    TDX_PROXY_CONFIG)

# 每个周期包含多少分钟
NUM_MINUTE_MAPPING = {}
NUM_MINUTE_MAPPING['1min'] = 1
NUM_MINUTE_MAPPING['5min'] = 5
NUM_MINUTE_MAPPING['15min'] = 15
NUM_MINUTE_MAPPING['30min'] = 30
NUM_MINUTE_MAPPING['1hour'] = 60
NUM_MINUTE_MAPPING['1day'] = 60 * 5.5  # 股票，收盘时间是15：00，开盘是9：30

# 常量
QSIZE = 800

# 通达信 <=> 交易所代码 映射
TDX_VN_STOCK_MARKET_MAP = {
    TDXParams.MARKET_SH: Exchange.SSE,  # 1: 上交所
    TDXParams.MARKET_SZ: Exchange.SZSE  # 0： 深交所
}
VN_TDX_STOCK_MARKET_MAP = {v: k for k, v in TDX_VN_STOCK_MARKET_MAP.items()}

# 通达信 <=> rq交易所代码 映射
TDX_RQ_STOCK_MARKET_MAP = {
    TDXParams.MARKET_SH: 'XSHG',  # 1: 上交所
    TDXParams.MARKET_SZ: 'XSHE'  # 0： 深交所
}
RQ_TDX_STOCK_MARKET_MAP = {v: k for k, v in TDX_RQ_STOCK_MARKET_MAP.items()}


# 本地缓存文件

class TdxStockData(object):
    exclude_ips = []

    def __init__(self, strategy=None, proxy_ip="", proxy_port=0):
        """
        构造函数
        :param strategy: 上层策略，主要用与使用write_log（）
        """
        self.strategy = strategy

        self.proxy_ip = proxy_ip
        self.proxy_port = proxy_port

        if self.proxy_port == 0 and len(self.proxy_ip) == 0:
            proxy_config = get_cache_json(TDX_PROXY_CONFIG)
            proxy_ip = proxy_config.get('proxy_ip', '')
            proxy_port = proxy_config.get('proxy_port', 0)
            if len(proxy_ip) > 0 and proxy_port > 0:
                self.proxy_ip = proxy_ip
                self.proxy_port = proxy_port
                self.write_log(f'使用vnpy/data/tdx/{TDX_PROXY_CONFIG}的proxy:{proxy_ip}:{proxy_port}')

        self.api = None
        self.connection_status = False  # 连接状态

        self.best_ip = None
        self.symbol_market_dict = {}  # tdx合约与tdx市场的字典

        self.config = get_cache_config(TDX_STOCK_CONFIG)
        self.symbol_dict = self.config.get('symbol_dict', {})
        self.cache_time = self.config.get('cache_time', datetime.now() - timedelta(days=7))
        self.best_ip = self.config.get('best_ip', {})
        self.exclude_ips = self.config.get('exclude_ips', [])

        if len(self.symbol_dict) == 0 or self.cache_time < datetime.now() - timedelta(days=1):
            self.cache_config()

    def write_log(self, content):
        """记录日志"""
        if self.strategy:
            self.strategy.write_log(content)
        else:
            print(content)

    def write_error(self, content):
        """记录错误"""
        if self.strategy:
            self.strategy.write_log(content, level=ERROR)
        else:
            print(content, file=sys.stderr)

    def select_best_ip(self, ip_list, proxy_ip="", proxy_port=0, exclude_ips=[]):
        """
        选取最快的IP
        :param ip_list:
        :param proxy_ip: 代理
        :param proxy_port: 代理端口
        :param exclude_ips: 排除清单
        :return:
        """
        from pytdx.util.best_ip import ping
        data = [ping(ip=x['ip'], port=x['port'], type_='stock', proxy_ip=proxy_ip, proxy_port=proxy_port) for x in
                ip_list if x['ip'] not in exclude_ips]
        results = []
        for i in range(len(data)):
            # 删除ping不通的数据
            if data[i] < timedelta(0, 9, 0):
                results.append((data[i], ip_list[i]))
            else:
                if ip_list[i].get('ip') not in self.exclude_ips:
                    self.exclude_ips.append(ip_list[i].get('ip'))

        # 按照ping值从小大大排序
        results = [x[1] for x in sorted(results, key=lambda x: x[0])]

        return results[0]

    def connect(self, is_reconnect: bool = False):
        """
        连接API
        :param:is_reconnect， 是否重新连接
        :return:
        """
        # 创建api连接对象实例
        try:
            if self.api is None or not self.connection_status:
                self.write_log(u'开始连接通达信股票行情服务器')
                self.api = TdxHq_API(heartbeat=True, auto_retry=True, raise_exception=True)

                # 选取最佳服务器
                if is_reconnect or self.best_ip is None:
                    self.best_ip = self.config.get('best_ip', {})
                    if is_reconnect:
                        selected_ip = self.best_ip.get('ip')
                        if selected_ip not in self.exclude_ips:
                            self.exclude_ips.append(selected_ip)
                        self.best_ip = {}
                    else:
                        # 超时的话，重新选择
                        last_datetime_str = self.best_ip.get('datetime', None)
                        if last_datetime_str:
                            try:
                                last_datetime = datetime.strptime(last_datetime_str, '%Y-%m-%d %H:%M:%S')
                                ip = self.best_ip.get('ip')
                                is_bad_ip = ip and ip in self.best_ip.get('exclude_ips', [])
                                if (datetime.now() - last_datetime).total_seconds() > 60 * 60 * 2 or is_bad_ip:
                                    self.best_ip = {}
                                    if not is_bad_ip:
                                        self.exclude_ips = []
                            except Exception as ex:  # noqa
                                self.best_ip = {}
                        else:
                            self.best_ip = {}

                if len(self.best_ip) == 0:
                    from pytdx.util.best_ip import stock_ip
                    self.best_ip = self.select_best_ip(ip_list=stock_ip,
                                                       proxy_ip=self.proxy_ip,
                                                       proxy_port=self.proxy_port,
                                                       exclude_ips=self.exclude_ips)
                    # 保存最新的选择，排除
                    self.config.update({'best_ip': self.best_ip,
                                        'select_dt': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'exclude_ips': self.exclude_ips})
                    save_cache_config(self.config, TDX_STOCK_CONFIG)

                # 如果配置proxy5，使用vnpy项目下的pytdx
                if len(self.proxy_ip) > 0 and self.proxy_port > 0:
                    self.api.connect(ip=self.best_ip['ip'], port=self.best_ip['port'],
                                     proxy_ip=self.proxy_ip, proxy_port=self.proxy_port)
                else:
                    # 使用pip install pytdx
                    self.api.connect(ip=self.best_ip['ip'], port=self.best_ip['port'])

                self.write_log(f'创建tdx连接, : {self.best_ip}')
                self.connection_status = True

        except Exception as ex:
            self.write_log(u'连接服务器{}tdx异常:{},{}'.format(self.best_ip, str(ex), traceback.format_exc()))
            cur_ip = self.best_ip.get('ip', None)
            if cur_ip is not None and cur_ip not in self.exclude_ips:
                self.write_log(f'排除{cur_ip}')
                self.exclude_ips.append(cur_ip)
                self.best_ip = {}

            return

    def disconnect(self):
        """断开连接"""
        if self.api is not None:
            self.api = None

    def cache_config(self):
        """缓存所有股票的清单"""
        for market_id in range(2):
            print('get market_id:{}'.format(market_id))
            security_list = self.get_security_list(market_id)
            if len(security_list) == 0:
                continue
            for security in security_list:
                tdx_symbol = security.get('code', None)
                exchange = Exchange.SZSE.value if market_id == 0 else Exchange.SSE.value
                stock_type = get_stock_type(tdx_symbol)
                security.update({'market_id': market_id})
                security.update({'stock_type': stock_type})
                security.update({'exchange': exchange})

                if tdx_symbol:
                    self.symbol_dict.update({f'{tdx_symbol}_{market_id}': security})

        self.config.update({'symbol_dict': self.symbol_dict, 'cache_time': datetime.now()})
        save_cache_config(data=self.config, config_file_name=TDX_STOCK_CONFIG)

    def get_security_list(self, market_id: int = 0):
        """
        获取市场代码
        :param: market_id: 1，上交所 ， 0， 深交所
        :return:
        """
        if self.api is None:
            self.connect()

        start = 0
        results = []
        # 接口有数据量连续，循环获取，直至取不到结果为止
        while True:
            try:
                result = self.api.get_security_list(market_id, start)
            except Exception:
                break
            if len(result) > 0:
                start += len(result)
            else:
                break
            results.extend(result)

        return results

    def get_name(self, symbol, market_id=None):
        """
        获取名称
        :param symbol:  代码  或者 代码.交易所
        :param market_id: 如果存在代码.交易所时，不使用该值
        :return:
        """
        if '.' in symbol:
            symbol, exchange = symbol.split('.')
            if exchange == Exchange.SSE.value:
                market_id = 1
            elif exchange == Exchange.SZSE.value:
                market_id = 0

        symbol_info = self.symbol_dict.get(f'{symbol}_{market_id}')
        if symbol_info:
            return symbol_info.get('name', symbol)

        return symbol

    # ----------------------------------------------------------------------
    def get_bars(self,
                 symbol: str,
                 period: str,
                 callback=None,
                 bar_freq: int = 1,
                 start_dt: datetime = None,
                 return_bar: bool = True):
        """
        返回k线数据
        symbol：股票 000001.XG
        period: 周期: 1min,5min,15min,30min,1hour,1day,
        """
        if not self.api:
            self.connect()
        ret_bars = []
        if self.api is None:
            return False, []

        # symbol => tdx_code, market_id
        if '.' in symbol:
            tdx_code, market_str = symbol.split('.')
            # 1, 上交所 ， 0， 深交所
            market_id = 1 if market_str.upper() in ['XSHG', Exchange.SSE.value] else 0
            self.symbol_market_dict.update({tdx_code: market_id})  # tdx合约与tdx市场的字典
        else:
            market_id = get_tdx_market_code(symbol)
            tdx_code = symbol
            self.symbol_market_dict.update({symbol: market_id})  # tdx合约与tdx市场的字典
        name = self.get_name(tdx_code, market_id)

        # period => tdx_period
        if period not in PERIOD_MAPPING.keys():
            self.write_error(u'{} 周期{}不在下载清单中: {}'
                             .format(datetime.now(), period, list(PERIOD_MAPPING.keys())))
            # print(u'{} 周期{}不在下载清单中: {}'.format(datetime.now(), period, list(PERIOD_MAPPING.keys())))
            return False, ret_bars
        tdx_period = PERIOD_MAPPING.get(period)

        # start_dt => qry_start_dt  & qry_end_dt
        if start_dt is None:
            self.write_log(u'没有设置开始时间，缺省为10天前')
            qry_start_date = datetime.now() - timedelta(days=10)
            start_dt = qry_start_date
        else:
            qry_start_date = start_dt
        qry_end_date = datetime.now()
        if qry_start_date > qry_end_date:
            qry_start_date = qry_end_date

        self.write_log('{}开始下载tdx股票: {},代码:{} {}数据, {} to {}.'
                       .format(datetime.now(), name, tdx_code, tdx_period, qry_start_date, qry_end_date))
        stock_type = get_stock_type(tdx_code, market_id)
        if stock_type == 'index_cn':
            get_bar_func = self.api.get_index_bars
        else:
            get_bar_func = self.api.get_security_bars

        try:
            _start_date = qry_end_date
            _bars = []
            _pos = 0
            while _start_date > qry_start_date:
                _res = get_bar_func(
                    category=PERIOD_MAPPING[period],
                    market=market_id,
                    code=tdx_code,
                    start=_pos,
                    count=QSIZE)
                if _res is not None:
                    _bars = _res + _bars
                _pos += QSIZE
                if _res is not None and len(_res) > 0:
                    _start_date = _res[0]['datetime']
                    _start_date = datetime.strptime(_start_date, '%Y-%m-%d %H:%M')
                    self.write_log(u'分段取数据开始:{}'.format(_start_date))
                else:
                    break
            if len(_bars) == 0:
                self.write_error('{} Handling {}, len1={}..., continue'.format(
                    str(datetime.now()), tdx_code, len(_bars)))
                sleep(3 * random.random())
                return False, ret_bars

            current_datetime = datetime.now()
            data = self.api.to_df(_bars)
            data = data.assign(datetime=pd.to_datetime(data['datetime']))
            data = data.assign(ticker=symbol)
            data['symbol'] = symbol
            data = data.drop(
                ['year', 'month', 'day', 'hour', 'minute', 'price', 'ticker'],
                errors='ignore',
                axis=1)
            data = data.rename(
                index=str,
                columns={'vol': 'volume'})

            if len(data) == 0:
                print('{} Handling {}, len2={}..., continue'.format(
                    str(datetime.now()), tdx_code, len(data)))
                return False, ret_bars

            # 通达信是以bar的结束时间标记的，vnpy是以bar开始时间标记的,所以要扣减bar本身的分钟数
            data['datetime'] = data['datetime'].apply(
                lambda x: x - timedelta(minutes=NUM_MINUTE_MAPPING.get(period, 1)))
            data['trading_date'] = data['datetime'].apply(lambda x: (x.strftime('%Y-%m-%d')))
            data['date'] = data['datetime'].apply(lambda x: (x.strftime('%Y-%m-%d')))
            data['time'] = data['datetime'].apply(lambda x: (x.strftime('%H:%M:%S')))
            data = data.set_index('datetime', drop=False)

            if return_bar:
                self.write_log('dataframe => [BarData]')
                exchange = TDX_VN_STOCK_MARKET_MAP.get(market_id, Exchange.LOCAL)
                for index, row in data.iterrows():
                    try:
                        add_bar = BarData(
                            gateway_name='tdx',
                            symbol=symbol,
                            exchange=exchange,
                            datetime=index
                        )
                        add_bar.date = row['date']
                        add_bar.time = row['time']
                        add_bar.trading_day = row['trading_date']
                        add_bar.open_price = float(row['open'])
                        add_bar.high_price = float(row['high'])
                        add_bar.low_price = float(row['low'])
                        add_bar.close_price = float(row['close'])
                        add_bar.volume = float(row['volume'])
                    except Exception as ex:
                        self.write_error('error when convert bar:{},ex:{},t:{}'
                                         .format(row, str(ex), traceback.format_exc()))
                        # print('error when convert bar:{},ex:{},t:{}'.format(row, str(ex), traceback.format_exc()))
                        return False, ret_bars

                    if start_dt is not None and index < start_dt:
                        continue
                    ret_bars.append(add_bar)

                    if callback is not None:
                        freq = bar_freq
                        bar_is_completed = True
                        if period != '1min' and index == data['datetime'][-1]:
                            # 最后一个bar，可能是不完整的，强制修改
                            # - 5min修改后freq基本正确
                            # - 1day在VNPY合成时不关心已经收到多少Bar, 所以影响也不大
                            # - 但其它分钟周期因为不好精确到每个品种, 修改后的freq可能有错
                            if index > current_datetime:
                                bar_is_completed = False
                                # 根据秒数算的话，要+1，例如13:31,freq=31，第31根bar
                                freq = NUM_MINUTE_MAPPING[period] - int((index - current_datetime).total_seconds() / 60)
                        callback(add_bar, bar_is_completed, freq)

            else:
                self.write_log('dataframe => [ dict ]')
                ret_bars = list(data.T.to_dict().values())
            return True, ret_bars
        except Exception as ex:
            self.write_error('exception in get:{},{},{}'.format(tdx_code, str(ex), traceback.format_exc()))
            # print('exception in get:{},{},{}'.format(tdx_symbol,str(ex), traceback.format_exc()))
            self.write_log(u'重置连接')
            self.api = None
            self.connect(is_reconnect=True)
            return False, ret_bars

    def get_last_bars(self, symbol: str, period: str = '1min', n: int = 2, return_bar: bool = True):
        """
        获取最后n根bar
        :param symbol:
        :param period:
        :param n:取bar数量
        :param return_bar:
        :return:
        """
        if not self.api:
            self.connect()
        ret_bars = []
        if self.api is None:
            return False, []

        # symbol => tdx_code, market_id
        if '.' in symbol:
            tdx_code, market_str = symbol.split('.')
            # 1, 上交所 ， 0， 深交所
            market_id = 1 if market_str.upper() in ['XSHG', Exchange.SSE.value] else 0
            self.symbol_market_dict.update({tdx_code: market_id})  # tdx合约与tdx市场的字典
        else:
            market_id = get_tdx_market_code(symbol)
            tdx_code = symbol
            self.symbol_market_dict.update({symbol: market_id})  # tdx合约与tdx市场的字典
        # period => tdx_period
        if period not in PERIOD_MAPPING.keys():
            self.write_error(u'{} 周期{}不在下载清单中: {}'
                             .format(datetime.now(), period, list(PERIOD_MAPPING.keys())))
            return False, ret_bars
        tdx_period = PERIOD_MAPPING.get(period)
        stock_type = get_stock_type(tdx_code)
        if stock_type == 'index_cn':
            get_bar_func = self.api.get_index_bars
        else:
            get_bar_func = self.api.get_security_bars
        try:
            datas = get_bar_func(
                category=PERIOD_MAPPING[period],
                market=market_id,
                code=tdx_code,
                start=0,
                count=n)
            if not datas or len(datas) == 0:
                return False, ret_bars

            if not return_bar:
                return True, datas

            exchange = TDX_VN_STOCK_MARKET_MAP.get(market_id, Exchange.LOCAL)
            delta_minutes = NUM_MINUTE_MAPPING.get(period, 1)
            for data in datas:
                bar_dt = datetime.strptime(data.get('datetime'), '%Y-%m-%d %H:%M')
                bar_dt = bar_dt - timedelta(minutes=delta_minutes)
                add_bar = BarData(
                    gateway_name='tdx',
                    symbol=symbol,
                    exchange=exchange,
                    datetime=bar_dt
                )
                add_bar.date = bar_dt.strftime('%Y-%m-%d')
                add_bar.time = bar_dt.strftime('%H:%M:%S')
                add_bar.trading_day = add_bar.date
                add_bar.open_price = float(data['open'])
                add_bar.high_price = float(data['high'])
                add_bar.low_price = float(data['low'])
                add_bar.close_price = float(data['close'])
                add_bar.volume = float(data['vol'])
                ret_bars.append(add_bar)
            return True, ret_bars
        except Exception as ex:
            self.write_error(f'获取{symbol}数据失败:{str(ex)}')
            return False, ret_bars

        # ----------------------------------------------------------------------

    def save_cache(self,
                   cache_folder: str,
                   cache_symbol: str,
                   cache_date: str,
                   data_list: list):
        """保存文件到缓存"""

        os.makedirs(cache_folder, exist_ok=True)

        if not os.path.exists(cache_folder):
            self.write_error('缓存目录不存在:{},不能保存'.format(cache_folder))
            return
        cache_folder_year_month = os.path.join(cache_folder, cache_date[:6])
        os.makedirs(cache_folder_year_month, exist_ok=True)

        save_file = os.path.join(cache_folder_year_month, '{}_{}.pkb2'.format(cache_symbol, cache_date))
        with bz2.BZ2File(save_file, 'wb') as f:
            pickle.dump(data_list, f)
            self.write_log(u'缓存成功:{}'.format(save_file))

    def load_cache(self,
                   cache_folder: str,
                   cache_symbol: str,
                   cache_date: str):
        """加载缓存数据"""
        if not os.path.exists(cache_folder):
            # self.write_error('缓存目录:{}不存在,不能读取'.format(cache_folder))
            return None
        cache_folder_year_month = os.path.join(cache_folder, cache_date[:6])
        if not os.path.exists(cache_folder_year_month):
            # self.write_error('缓存目录:{}不存在,不能读取'.format(cache_folder_year_month))
            return None

        cache_file = os.path.join(cache_folder_year_month, '{}_{}.pkb2'.format(cache_symbol, cache_date))
        if not os.path.isfile(cache_file):
            # self.write_error('缓存文件:{}不存在,不能读取'.format(cache_file))
            return None
        with bz2.BZ2File(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data

        return None

    def get_history_transaction_data(self,
                                     symbol: str,
                                     trading_date,
                                     cache_folder: str = None):
        """
        获取当某一交易日的历史成交记录
        :param symbol: 查询合约  xxxxxx.交易所
        :param trading_date: 可以是日期参数，或者字符串参数,支持 2019-01-01 或  20190101格式
        :param cache_folder:
        :return:
        """
        if not self.api:
            self.connect()

        ret_datas = []

        # trading_date ，转为为查询数字类型
        if isinstance(trading_date, datetime):
            trading_date = trading_date.strftime('%Y%m%d')
        if isinstance(trading_date, str):
            trading_date = int(trading_date.replace('-', ''))

        cache_symbol = symbol
        cache_date = str(trading_date)

        max_data_size = sys.maxsize
        # symbol.exchange => tdx_code market_code
        if '.' in symbol:
            tdx_code, market_str = symbol.split('.')
            market_code = 1 if market_str.upper() in ['XSHG', Exchange.SSE.value] else 0
            self.symbol_market_dict.update({tdx_code: market_code})  # tdx合约与tdx市场的字典
        else:
            market_code = get_tdx_market_code(symbol)
            tdx_code = symbol
            self.symbol_market_dict.update({symbol: market_code})  # tdx合约与tdx市场的字典

        symbol_config = self.symbol_dict.get(f'{tdx_code}_{market_code}', {})
        decimal_point = symbol_config.get('decimal_point', 2)

        q_size = QSIZE * 5
        # 每秒 2个， 10小时
        max_data_size = 1000000
        # 优先从缓存加载
        if cache_folder:
            buffer_data = self.load_cache(cache_folder, cache_symbol, cache_date)
            if buffer_data:
                return True, buffer_data

        self.write_log(u'开始下载{} 历史{}分笔数据'.format(trading_date, symbol))

        is_today = False
        if trading_date == int(datetime.now().strftime('%Y%m%d')):
            is_today = True

        try:
            _datas = []
            _pos = 0

            while True:
                if is_today:
                    # 获取当前交易日得交易记录
                    _res = self.api.get_transaction_data(
                        market=self.symbol_market_dict[tdx_code],
                        code=tdx_code,
                        start=_pos,
                        count=q_size)
                else:
                    # 获取历史交易记录
                    _res = self.api.get_history_transaction_data(
                        market=self.symbol_market_dict[tdx_code],
                        date=trading_date,
                        code=tdx_code,
                        start=_pos,
                        count=q_size)
                last_dt = None
                if _res is not None:
                    _datas = _res + _datas
                _pos += min(q_size, len(_res))

                if _res is not None and len(_res) > 0:
                    self.write_log(u'分段取{}分笔数据:{} ~{}, {}条,累计:{}条'
                                   .format(trading_date, _res[0]['time'], _res[-1]['time'], len(_res), _pos))
                else:
                    break

                if len(_datas) >= max_data_size:
                    break

            if len(_datas) == 0:
                self.write_error(u'{}分笔成交数据获取为空'.format(trading_date))
                return False, _datas

            for d in _datas:
                dt = datetime.strptime(str(trading_date) + ' ' + d.get('time'), '%Y%m%d %H:%M')
                if last_dt is None or last_dt < dt:
                    last_dt = dt
                else:
                    if last_dt < dt + timedelta(seconds=59):
                        last_dt = last_dt + timedelta(seconds=1)
                d.update({'datetime': last_dt})
                d.update({'volume': d.pop('vol', 0)})
                if decimal_point > 2:
                    price = round(d.get('price') / (10 ** (decimal_point - 2)), decimal_point)
                    d.update({'price': price})

                d.update({'trading_date': last_dt.strftime('%Y-%m-%d')})

            _datas = sorted(_datas, key=lambda s: s['datetime'])

            # 缓存文件
            if cache_folder and not is_today:
                self.save_cache(cache_folder, cache_symbol, cache_date, _datas)

            return True, _datas

        except Exception as ex:
            self.write_error(
                'exception in get_transaction_data:{},{},{}'.format(symbol, str(ex), traceback.format_exc()))
            return False, ret_datas

    def get_stock_list(self, types=["stock_cn", "etf_cn", "bond_cn", "cb_cn"]):
        """股票所有的code&name列表"""
        if self.api is None:
            self.connect()

        data = pd.concat(
            [pd.concat(
                [self.api.to_df(self.api.get_security_list(j, i * 1000)).assign(sse='sz' if j == 0 else 'sh').set_index(
                    ['code', 'sse'], drop=False) for i in range(int(self.api.get_security_count(j) / 1000) + 1)],
                axis=0) for j
                in range(2)], axis=0)
        sz = data.query('sse=="sz"')
        sh = data.query('sse=="sh"')
        sz = sz.assign(sec=sz.code.apply(get_stock_type))
        sh = sh.assign(sec=sh.code.apply(get_stock_type))

        temp_df = pd.concat([sz, sh]).query('sec in ["{}"]'.format(types)).sort_index().assign(
            name=data['name'].apply(lambda x: str(x)[0:6]))

        hq_codelist = []

        for i in range(0, len(temp_df)):
            row = temp_df.iloc[i]
            hq_codelist.append(
                {
                    "code": row['code'],
                    "exchange": Exchange.SSE.value if row['sse'] == 'sh' else Exchange.SZSE.value,
                    "market_id": 1 if row['sse'] == 'sh' else 0,
                    "name": row['name']

                }
            )

        return hq_codelist

    def get_security_quotes(self, all_stock, code=None):
        """
        支持三种形式的参数
        get_security_quotes(market, code )
        get_security_quotes((market, code))
        get_security_quotes([(market1, code1), (market2, code2)] )
        :param all_stock （market, code) 的数组
        :param code{optional} code to query
        :return:
        """
        if self.api is None:
            self.connect()

        return self.api.get_security_quotes(all_stock, code)

    def get_stock_quotes_by_type(self, stock_type):
        """根据股票代码类型，获取其最新行情"""
        stock_list = [(stock.get('market_id'), stock.get('code')) for stock in self.symbol_dict.values() if
                      stock.get('stock_type') == stock_type]

        num_per_count = 60
        results = []
        for i in range(0, len(stock_list) + 1, num_per_count):
            cur_results = self.get_security_quotes(stock_list[i:i + num_per_count])
            results.extend(cur_results)

        return results
