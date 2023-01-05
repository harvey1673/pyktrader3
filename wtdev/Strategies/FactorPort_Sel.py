from wtpy import BaseSelStrategy
from wtpy import SelContext
import numpy as np
import pandas as pd
import datetime
import json
from pycmqlib3.utility.misc import day_shift, inst2exch, inst2product
from pycmqlib3.utility.dbaccess import load_factor_data
from pycmqlib3.analytics.data_handler import ATR
from pycmqlib3.utility.process_wt_data import wt_time_to_min_id


class StraFactorPortSel(BaseSelStrategy):
    def __init__(self, config_file):
        with open(config_file, 'r') as fp:
            strat_conf = json.load(fp)
        strat_args = strat_conf.get('config', {'name': 'test'})
        BaseSelStrategy.__init__(self, strat_args['name'])
        assets = strat_args['assets']
        num_assets = len(assets)
        self.__vol_weight = [1.0] * num_assets
        self.__prod_list = [''] * num_assets
        self.__codes = [''] * num_assets
        self.__prev_codes = [''] * num_assets
        self.__atr = [np.nan] * num_assets
        self.__target_pos = [0] * num_assets
        config_keys = ['factor_repo', 'vol_win',
                       'fact_db_table',
                       'exec_bar_list',
                       'pos_scaler',
                       'freq',
                       'hist_fact_lookback',
                       'roll_label']
        d = self.__dict__
        for key in config_keys:
            d['__'+key] = strat_args[key]
        for idx, asset_dict in enumerate(assets):
            under = asset_dict["underliers"][0]
            prev_under = asset_dict["prev_underliers"][0]
            self.__prod_list[idx] = inst2product(under)
            exch = inst2exch(under)
            self.__codes[idx] = exch + '.' + under
            if len(prev_under) > 0:
                self.__prev_codes[idx] = exch + '.' + prev_under
        self.__factor_data = {}
        self.__factor_pos = {}
        self.__pos_sum = pd.DataFrame()

    def on_init(self, context: SelContext):
        for idx, code in enumerate(self.__codes):
            context.stra_prepare_bars(code, 'd1', self.__vol_win + 10, isMain=False)
            if idx == 0:
                context.stra_prepare_bars(code, 'm1', 10, isMain=True)
            else:
                context.stra_prepare_bars(code, 'm1', 10, isMain=False)

    def load_fact_data(self, context: SelContext):
        cur_date = context.stra_get_date()
        end_date = datetime.date(cur_date // 10000,
                                 (cur_date % 10000)//100,
                                 cur_date % 100)
        start_date = day_shift(end_date, '-%sb' % (str(self.__hist_fact_lookback)))
        fact_list = list(set([self.__factor_repo[fact]['name'] for fact in self.__factor_repo.keys()]))
        df = load_factor_data(self.__prod_list,
                              factor_list=fact_list,
                              roll_label=self.__roll_label,
                              start=start_date,
                              end=end_date,
                              freq=self.__freq,
                              db_table=self.__fact_db_table)
        for fact in self.__factor_repo:
            xdf = pd.pivot_table(df[df['fact_name'] == self.__factor_repo[fact]['name']],
                                 values='fact_val',
                                 index=['date', 'serial_key'],
                                 columns=['product_code'],
                                 aggfunc='last')
            for prod in self.__prod_list:
                if prod not in xdf.columns:
                    xdf[prod] = np.nan
            self.__fact_data[fact] = xdf[self.__prod_list]
        self.__pos_sum = pd.DataFrame()
        for fact in self.__factor_repo:
            rebal_freq = self.__factor_repo[fact]['rebal']
            weight = self.__factor_repo[fact]['weight']
            self.__factor_pos[fact] = pd.DataFrame(index=self.__fact_data[fact].index,
                                                   columns=self.__fact_data[fact].columns)
            if self.__factor_repo[fact]['type'] == 'pos':
                self.__factor_pos[fact] = self.__fact_data[fact].copy()
            elif self.__factor_repo[fact]['type'] == 'ts':
                rebal_ts = pd.Series(range(len(self.__fact_data[fact].index)),
                                     index=self.__fact_data[fact].index)
                for rebal_idx in range(rebal_freq):
                    flag = rebal_ts % rebal_freq == rebal_idx
                    long_pos = pd.Series(np.nan, index=self.__fact_data[fact].index)
                    short_pos = pd.Series(np.nan, index=self.__fact_data[fact].index)
                    for asset in self.__prod_list:
                        pflag = (self.__fact_data[fact][asset] >= 0.0)
                        nflag = (self.__fact_data[fact][asset] <= 0.0)
                        long_pos[flag & pflag] = self.__fact_data[fact][asset][flag & pflag]
                        long_pos[flag & (~pflag)] = 0.0
                        long_pos[flag] = long_pos[flag].fillna(method='ffill').fillna(0.0)
                        short_pos[flag & nflag] = self.__fact_data[fact][asset][flag & nflag]
                        short_pos[flag & (~nflag)] = 0.0
                        short_pos[flag] = short_pos[flag].fillna(method='ffill').fillna(0.0)
                        self.__factor_pos[fact].loc[flag, asset] = long_pos[flag] + short_pos[flag]
            elif self.__factor_repo[fact]['type'] == 'xs':
                lower_rank = int(len(self.__prod_list) * self.__factor_repo[fact]['threshold']) + 1
                upper_rank = len(self.__prod_list) - int(len(self.__prod_list) * self.__factor_repo[fact]['threshold'])
                rank_df = self.__fact_data[fact].rank(axis=1)
                rebal_ts = pd.Series(range(len(self.__fact_data[fact].index)),
                                     index=self.__fact_data[fact].index)
                for rebal_idx in range(rebal_freq):
                    flag = rebal_ts % rebal_freq == rebal_idx
                    long_pos = pd.Series(np.nan, index=self.__fact_data[fact].index)
                    short_pos = pd.Series(np.nan, index=self.__fact_data[fact].index)
                    for asset in self.__prod_list:
                        pflag = (rank_df[asset] > upper_rank)
                        nflag = (rank_df[asset] < lower_rank)
                        long_pos[flag & pflag] = 1.0
                        long_pos[flag & (~pflag)] = 0.0
                        long_pos[flag] = long_pos[flag].fillna(method='ffill').fillna(0.0)
                        short_pos[flag & nflag] = -1.0
                        short_pos[flag & (~nflag)] = 0.0
                        short_pos[flag] = short_pos[flag].fillna(method='ffill').fillna(0.0)
                        self.__factor_pos[fact].loc[flag, asset] = long_pos[flag] + short_pos[flag]
                self.__factor_pos[fact] = self.__factor_pos[fact].fillna(0.0)
            fact_pos = pd.Series(self.__factor_pos[fact].iloc[-rebal_freq:, :].sum()/rebal_freq * weight,
                                 name=fact)
            self.__pos_sum = self.__pos_sum.append(fact_pos)
        self.__pos_sum = self.__pos_sum[self.__prod_list].round(2)
        net_pos = self.__pos_sum.sum()
        for idx, prodcode in enumerate(self.__prod_list):
            if prodcode == 'CJ':
                self.__target_pos[idx] = int((net_pos[prodcode] * self.__vol_weight[idx]/4 +
                                              (0.5 if net_pos[prodcode] > 0 else -0.5)))*4
            elif prodcode == 'ZC':
                self.__target_pos[idx] = int((net_pos[prodcode] * self.__vol_weight[idx]/2 +
                                              (0.5 if net_pos[prodcode] > 0 else -0.5)))*2
            else:
                self.__target_pos[idx] = int(net_pos[prodcode] * self.__vol_weight[idx] +
                                             (0.5 if net_pos[prodcode] > 0 else -0.5))

    def on_session_begin(self, context:SelContext, curTDate:int):
        for idx, code in enumerate(self.__codes):
            pInfo = context.stra_get_comminfo(code)
            vol_scale = pInfo.volscale
            daily_bars = context.stra_get_bars(code, 'd1', self.__vol_win + 10, isMain=False)
            daily_bars = daily_bars.to_df()
            self.__atr[idx] = ATR(daily_bars, self.__vol_win).iloc[-1]
            self.__vol_weight[idx] = self.__pos_scaler/self.__atr[idx]/vol_scale

    def on_calculate(self, context: SelContext):
        cur_time = context.stra_get_time()
        cur_min = wt_time_to_min_id(cur_time)
        if cur_min in [301, 1501]:
            self.load_fact_data(context)
        if cur_min not in self.__exec_bar_list:
            return
        for idx, code in enumerate(self.__codes):
            sInfo = context.stra_get_sessioninfo(code)
            if not sInfo.isInTradingTime(cur_time):
                continue
            prev_code = self.__prev_codes[idx]
            if len(prev_code) > 0:
                prev_pos = context.stra_get_position(prev_code)
                if prev_pos != 0:
                    context.stra_set_position(prev_code, 0, 'ExitPrevPosition')
                    context.stra_log_text(f"close prev position for {prev_code}")
            cur_pos = context.stra_get_position(code)
            target_pos = self.__target_pos[idx]
            if cur_pos != target_pos:
                context.stra_set_position(code, target_pos, 'AdjustPosition')
                context.stra_log_text(f"adjust position for {code} from {cur_pos} to {target_pos}")
            continue

    def on_tick(self, context: SelContext, code: str, newTick: dict):
        return

    def on_bar(self, context: SelContext, code: str, period: str, newBar: dict):
        return