import misc
import json
import data_handler as dh
import pandas as pd
import numpy as np
import strategy as strat
import datetime
import backtest
import sys

def bband_chan_sim( mdf, config):
    start_equity = config['capital']
    marginrate = config['marginrate']
    offset = config['offset']
    pos_class = config['pos_class']
    pos_args  = config['pos_args']
    param = config['param']
    #tick_base = config['tick_base']
    close_daily = config['close_daily']
    win = param[0]
    k = param[1]
    chan = param[2]
    min_win = config['min_win']
    if chan > 0:
        chan_func = config['chan_func']
    tcost = config['trans_cost']
    unit = config['unit']
    freq = config['freq']
    xdf = dh.conv_ohlc_freq(mdf, freq)
    xdf['boll_ma'] = dh.MA(xdf, win).shift(1)
    boll_std = dh.STDDEV(xdf, win).shift(1)
    xdf['upbnd'] = xdf['boll_ma'] + boll_std * k
    xdf['lowbnd'] = xdf['boll_ma'] - boll_std * k
    if chan > 0:
        xdf['chan_h'] = eval(chan_func['high']['func'])(xdf, chan, **chan_func['high']['args']).shift(1)
        xdf['chan_l'] = eval(chan_func['low']['func'])(xdf, chan, **chan_func['low']['args']).shift(1)
    else:
        xdf['chan_h'] = 0
        xdf['chan_l'] = 10000000
    xdf['high_band'] = xdf[['chan_h', 'upbnd']].max(axis=1)
    xdf['low_band'] = xdf[['chan_l', 'lowbnd']].min(axis=1)
    xdf['prev_close'] = xdf['close'].shift(1)
    xdf['close_ind'] = np.isnan(xdf['close'].shift(-1))
    if close_daily:
        daily_end = (xdf['date']!=xdf['date'].shift(-1))
        xdf['close_ind'] = xdf['close_ind'] | daily_end
    ll = xdf.shape[0]
    xdf['pos'] = pd.Series([0]*ll, index = xdf.index)
    xdf['cost'] = pd.Series([0]*ll, index = xdf.index)
    xdf['traded_price'] = xdf.open
    curr_pos = []
    closed_trades = []
    tradeid = 0
    holding_period = 0
    for idx, dd in enumerate(xdf.index):
        mslice = xdf.loc[dd]
        if len(curr_pos) == 0:
            pos = 0
        else:
            pos = curr_pos[0].pos
            holding_period += 1
            lookback = max(win - holding_period, min_win)
            ma_stop = np.mean(xdf.close[(idx-lookback):idx])
            stop_line = 0
            if pos > 0:
                stop_line = min(mslice.upbnd, stop_line)
            else:
                stop_line = max(mslice.lowbnd, stop_line)
            curr_pos[0].set_exit(stop_line)
        xdf.set_value(dd, 'pos', pos)
        if np.isnan(mslice.boll_ma) or np.isnan(mslice.chan_h):
            continue
        if mslice.close_ind or (pos !=0 and curr_pos[0].check_exit(mslice.open, 0)):
            if pos!=0:
                curr_pos[0].close(mslice.open - misc.sign(pos) * offset, dd)
                tradeid += 1
                curr_pos[0].exit_tradeid = tradeid
                closed_trades.append(curr_pos[0])
                curr_pos = []
                xdf.set_value(dd, 'cost', xdf.at[dd, 'cost'] - abs(pos) * ( mslice.open * tcost))
                xdf.set_value(dd, 'traded_price', mslice.open - misc.sign(pos) * offset)
                pos = 0
        else:
            if ((mslice.open >= mslice.high_band) or (mslice.open <= mslice.low_band)) and (pos==0):
                target_pos = ( mslice.open >= mslice.high_band) * unit - (mslice.open <= mslice.low_band) * unit
                new_pos = pos_class([mslice.contract], [1], target_pos, mslice.open, mslice.open, **pos_args)
                tradeid += 1
                new_pos.entry_tradeid = tradeid
                new_pos.open(mslice.open + misc.sign(target_pos)*offset, dd)
                curr_pos.append(new_pos)
                pos = target_pos
                xdf.set_value(dd, 'cost', xdf.at[dd, 'cost'] - abs(target_pos) * (mslice.open * tcost))
                xdf.set_value(dd, 'traded_price', mslice.open + misc.sign(target_pos)*offset)
        xdf.ix[dd, 'pos'] = pos
    return (xdf, closed_trades)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_func']  = 'bktest_boll_bandit.bband_chan_sim'
    sim_config['scen_keys'] = ['close_daily', 'param']
    sim_config['sim_name']   = 'boll_bandit_30min_HL'
    sim_config['products']   = ['rb', 'i', 'j', 'jm', 'ZC', 'ni', 'y', 'p', 'm', 'RM', 'cs', 'jd', 'a', 'l', 'pp', 'TA', 'MA', 'bu', 'cu', 'al', 'zn', 'ru']
    sim_config['start_date'] = '20150102'
    sim_config['end_date']   = '20160513'
    sim_config['need_daily'] = False
    sim_config['close_daily']  =  [ False, True]
    sim_config['param'] = [(20, 1, 5), (20, 1, 10), (20, 1, 20), (20, 1, 40), (20, 1.5, 5), (20, 1.5, 10), (20, 1.5, 20), (20, 1.5, 40), \
                           (40, 1, 10), (40, 1, 20), (40, 1, 40), (40, 1, 80), (40, 1.5, 10), (40, 1.5, 20), (40, 1.5, 40), (40, 1.5, 80), \
                           (80, 1, 10), (80, 1, 20), (80, 1, 40), (80, 1, 80), (80, 1.5, 10), (80, 1.5, 20), (80, 1.5, 40), (80, 1.5, 80)]
    sim_config['pos_class'] = 'strat.TradePos'
    sim_config['offset']    = 1
    chan_func = { 'high': {'func': 'dh.DONCH_H', 'args':{'field': 'high'}},
                  'low':  {'func': 'dh.DONCH_L', 'args':{'field': 'low'}}}
    config = {'capital': 10000,
              'freq': '30min',
              'min_win': 10,
              'trans_cost': 0.0,
              'unit': 1,
              'stoploss': 0.0,
              'pos_args': {},
              'exit_min': 2055,
              'chan_func': chan_func,
              }
    sim_config['config'] = config
    with open(filename, 'w') as outfile:
        json.dump(sim_config, outfile)
    return sim_config

if __name__=="__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("need to input a file name for config file")
    else:
        gen_config_file(args[0])
    pass
