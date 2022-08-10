# flake8: noqa
import os
import sys
import json

# os.environ["VNPY_TESTING"] = "1"

from pycmqlib3.data.tdx.tdx_common import FakeStrategy
from pycmqlib3.data.tdx.tdx_stock_data import *

t1 = FakeStrategy()
t2 = FakeStrategy()

# 创建API对象(使用本地socket5代理）
#api_01 = TdxStockData(strategy=t1, proxy_ip='localhost', proxy_port=1080)
# 不使用代理
api_01 = TdxStockData(strategy=t1)
#
# # 获取市场下股票
# for market_id in range(2):
#     print('get market_id:{}'.format(market_id))
#     security_list = api_01.get_security_list(market_id)
#     if len(security_list) == 0:
#         continue
#     for security in security_list:
#         if security['code'].startswith('88'):
#             str_security = json.dumps(security, indent=1, ensure_ascii=False)
#             print(str_security)
#         if security.get('code', '').startswith('12') or u'转债' in security.get('name', ''):
#             str_security = json.dumps(security, indent=1, ensure_ascii=False)
#             # print('market_id:{},{}'.format(market_id, str_security))

    # str_markets = json.dumps(security_list, indent=1, ensure_ascii=False)
    # print(u'{}'.format(str_markets))

# 获取历史分钟线
ret,result = api_01.get_bars('880351.SSE', period='1hour', callback=t1.display_bar)
if ret:
    for bar in result:
        print(bar)
# ret,result = api_01.get_last_bars(symbol='002024',return_bar=True)
# if ret:
#     print(result)


# api.get_bars(symbol, period='5min', callback=display_bar)
# api.get_bars(symbol, period='1day', callback=display_bar)
# api_02 = TdxData(t2)
# api_02.get_bars('601390', period='1day', callback=t1.display_bar)

# 获取历史分时数据
# ret,result = api_01.get_history_transaction_data('RB99', '20190909')
# for r in result[0:10] + result[-10:]:
#    print(r)

# # 获取历史分时数据
# ret, result = api_01.get_history_transaction_data('110031', '20200504')
# for r in result[0:10] + result[-10:]:
#     print(r)
