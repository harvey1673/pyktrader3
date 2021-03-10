#-*- coding:utf-8 -*-

TICK_PER_SAVE = 0
TICK_BUF_SHIFT_SIZE = 600
BAR_BUF_SHIFT_SIZE = 600
tick_data_list = ['timestamp', 'date', 'tick_id', 'price', 'high', 'low', 'volume', 'openInterest',
                  'bid_price1', 'bid_vol1', 'ask_price1', 'ask_vol1',
                  #'bidPrice2', 'bidVol2', 'askPrice2', 'askVol2',
                  #'bidPrice3', 'bidVol3', 'askPrice3', 'askVol3',
                  #'bidPrice4', 'bidVol4', 'askPrice4', 'askVol4',
                  #'bidPrice4', 'bidVol4', 'askPrice4', 'askVol4',
                  ]
min_data_list = ['datetime', 'date', 'min_id', 'bar_id', 'open', 'high','low', 'close', 'volume', 'openInterest', 'tick_min']
day_data_list = ['date', 'open', 'high','low', 'close', 'volume', 'openInterest']
dtype_map = {'date': 'datetime64[D]',
            'datetime': 'datetime64[ms]',
            'timestamp': 'datetime64[ms]',
            'open': 'float',
            'close': 'float',
            'high': 'float',
            'low': 'float',
            'price': 'float',
            'volume': 'int',
            'openInterest': 'int',
            'min_id': 'int',
            'bar_id': 'int',
            'tick_min': 'int',
            'tick_id': 'int',
            'bid_price1': 'float',
            'bid_vol1': 'int',
            'ask_price1': 'float',
            'ask_vol1': 'int',
            'bid_price2': 'float',
            'bid_vol2': 'int',
            'ask_price2': 'float',
            'ask_vol2': 'int',
            'bid_price3': 'float',
            'bid_vol3': 'int',
            'ask_price3': 'float',
            'ask_vol3': 'int',
            'bid_price4': 'float',
            'bid_vol4': 'int',
            'ask_price4': 'float',
            'ask_vol4': 'int',
            'bid_price5': 'float',
            'bid_vol5': 'int',
            'ask_price5': 'float',
            'ask_vol5': 'int',
             }
