import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from pandas import pd
from backtest import *


class XSMOMRetSim(StratSim):
    def __init__(self, config):
        self.data_store = config['data_store']
        super(XSMOMRetSim, self).__init__(config)

    def process_config(self, config):
        self.assets = config['assets']
        self.tick_base = config['tick_base']
        self.offset = config['offset']
        self.win_list = config['win_list']
        self.ma_win = config['ma_win']
        self.freq = config['freq']
        self.total_risk = config['total_risk']
        self.shift_mode = config.get('shift_mode', 1)
        self.rebal_freq = config.get('rebal_freq', 1)
        self.quantile_cutoff = config.get('quantile_cutoff', 0.2)
        self.exclude_minlist = config.get('exclude_minlist', [])
        self.mom_excl_range = config.get('mom_excl_range', 0.0)
        self.tcost = config['trans_cost']
        self.unit = config['unit']
        self.weights = config.get('weights', len(self.assets)*[1.0])

    def process_data(self, df):
        if self.freq in self.data_store:
            adf = self.data_store[self.freq]
        else:
            if ('m' in self.freq) and (int(self.freq[:-1]) > 1):
                adf = dh.conv_ohlc_freq1(df, self.freq)
            elif ('s' in self.freq):
                sp = day_split_dict[self.freq]
                adf = dh.day_split1(df, sp)
            else:
                print("uncovered data case")
                adf = df
                self.data_store[self.freq] = adf
            self.data_store[self.freq] = adf
        xdf = adf.copy()
        for asset in self.assets:
            if self.shift_mode == 1:
                xdf[(asset, 'lr')] = ((xdf[(asset, 'close')] - xdf[(asset, 'shift')]).astype('float') \
                                      / (xdf[(asset, 'close')].shift(1) - xdf[(asset, 'shift')]) - 1.0) * 100.0
            else:
                xdf[(asset, 'lr')] = (xdf[(asset, 'close')].astype('float') / xdf[(asset, 'close')].shift(1) - 1.0) * 100.0
            for ix, win in enumerate(self.win_list):
                xdf[(asset, 'lr%s' % (ix))] = xdf[(asset, 'lr')].rolling(win).sum().rolling(self.ma_win).mean()
            xdf[(asset, 'closeout')] = 0.0
            xdf[(asset, 'cost')] = 0.0
            xdf[(asset, 'pos')] = 0.0
            xdf[(asset, 'traded_price')] = xdf[(asset, 'open')]
        self.df = xdf.dropna()

    def run_vec_sim(self):
        xdf = self.df
        kcut = int(self.quantile_cutoff * len(self.assets) + 0.5)
        upper_rank = len(self.assets) - kcut
        lower_rank = 1 + kcut
        rank_dict = {}
        rng_dict = {}
        for idx in range(len(self.win_list)):
            col = 'lr%s' % (idx)
            tmp_df = xdf[[(asset, col) for asset in self.assets]]
            rank_dict[col] = tmp_df.rank(axis=1)
            rng_dict[col] = tmp_df.max(axis=1) - tmp_df.min(axis=1)
        xdf['start_min'] = xdf.index.to_series().apply(lambda x: misc.get_min_id(x))
        xdf['rebal_flag'] = 1
        flag = (xdf['start_min'].isin(self.exclude_minlist)) | (rng_dict['lr0'] < self.mom_excl_range)
        xdf.loc[flag, 'rebal_flag'] = 0
        xdf['rebal_seqno'] = xdf['rebal_flag'].cumsum()
        sum_rank = pd.DataFrame(columns = self.assets, index = xdf.index)
        for asset in self.assets:
            sum_rank[asset] = rank_dict['lr0'][(asset, 'lr0')]
            for col in ['lr%s' % (idx + 1) for idx in range(len(self.win_list) - 1)]:
                sum_rank[asset] = sum_rank[asset] + rank_dict[col][(asset, col)]
        sum_rank = sum_rank.rank(axis=1, method='first')
        long_pos = pd.DataFrame(0, columns = self.assets, index = xdf.index)
        short_pos = pd.DataFrame(0, columns = self.assets, index = xdf.index)
        for rebal_idx in range(self.rebal_freq):
            long_tmp = pd.DataFrame(columns = self.assets, index = xdf.index)
            short_tmp = pd.DataFrame(columns = self.assets, index = xdf.index)
            rebal_flag = xdf['rebal_seqno'].apply(lambda x: (x % self.rebal_freq) == rebal_idx)
            for asset in self.assets:
                long_tmp.loc[(sum_rank[asset] > upper_rank) & rebal_flag, asset] = 1.0
                long_tmp.loc[(sum_rank[asset] <= upper_rank) & rebal_flag, asset] = 0.0
                short_tmp.loc[(sum_rank[asset] < lower_rank) & rebal_flag, asset] = 1.0
                short_tmp.loc[(sum_rank[asset] >= lower_rank) & rebal_flag, asset] = 0.0
            long_tmp = long_tmp.fillna(method='ffill').fillna(0)
            short_tmp = short_tmp.fillna(method='ffill').fillna(0)
            #long_sum = long_tmp.sum(axis=1)
            #short_sum = short_tmp.sum(axis=1)
            long_pos = long_pos + long_tmp #.div(long_sum, axis=0)
            short_pos = short_pos + short_tmp #.div(short_sum, axis=0)
        net_pos = long_pos - short_pos
        pos_nchg = (net_pos == net_pos.shift(1))
        net_pos[pos_nchg] = np.nan
        extract_fields = ['open', 'close', 'traded_price', 'contract', 'cost', 'pos']
        df_list = []
        for asset, offset in zip(self.assets, self.offset):
            if self.shift_mode == 1:
                orig_close = xdf[(asset, 'close')] - xdf[(asset, 'shift')]
            elif self.shift_mode == 2:
                orig_close = xdf[(asset, 'close')] * np.exp(-xdf[(asset, 'shift')])
            else:
                orig_close = xdf[(asset, 'close')]
            net_pos[asset] = (net_pos[asset] * self.total_risk / orig_close.astype('float')).shift(1).fillna(method='ffill').fillna(0.0).astype('int')
            #long_pos[asset] = (long_pos[asset] * self.total_risk / xdf[(asset, 'close')].astype('float')).astype('int').shift(1).fillna(method='ffill')
            #short_pos[asset] = (short_pos[asset] * self.total_risk / xdf[(asset, 'close')].astype('float')).astype('int').shift(1).fillna(method='ffill')
            xdf[(asset, 'pos')] = net_pos[asset]
            xdf[(asset, 'traded_price')] = xdf[(asset, 'open')]
            xdf.ix[-1, (asset, 'pos')] = 0
            xdf[(asset, 'cost')] = abs(xdf[(asset, 'pos')] - xdf[(asset, 'pos')].shift(1)) * offset
            xdf[(asset, 'cost')] = xdf[(asset, 'cost')].fillna(0.0)
            fields = [(asset, field) for field in extract_fields]
            tdf = xdf[fields]
            tdf.columns = extract_fields
            tdf['date'] = xdf['date']
            tdf['min_id'] = xdf['min_id']
            df_list.append(tdf)
            #tdf.to_csv("test_%s_offset_%s.csv" % (asset, offset))
        #xdf.to_csv("test_result.csv")
        closed_trades = []
        #for asset, df, offset in zip(self.assets, df_list, self.offset):
        #    closed_trades = closed_trades + simdf_to_trades1(df, slippage = offset)
        return (df_list, closed_trades)

def gen_config_file(filename):
    sim_config = {}
    sim_config['sim_class'] = 'bktest.xstest_mom_ret.XSMOMRetSim'
    sim_config['sim_func'] = 'run_vec_sim'
    sim_config['sim_freq'] = 'm'
    sim_config['scen_keys'] = ['freq', 'rebal_freq', 'win_list', 'quantile_cutoff']
    sim_config['sim_name'] = 'xsmom_ret_200124'
    sim_config['products'] = [['rb', 'hc', 'i', 'j', 'jm']]
    sim_config['start_date'] = '20161201'
    sim_config['end_date'] = '20200123'
    sim_config['win_list'] = [[6], [8], [10], [12], [16], [20], [24], [30], [36]]
    sim_config['freq'] = ['5m', '10m', '15m', '30m', '60m', '14m', '29m', '58m']
    sim_config['rebal_freq'] = [2, 4, 6, 8, 12]
    sim_config['quantile_cutoff'] = [0.2, 0.4]
    sim_config['pos_class'] = 'trade_position.TradePos'
    sim_config['offset'] = 1
    config = {'capital': 10000000,
              'trans_cost': 0.0,
              'unit': 1,
              'total_risk': 1000000,
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