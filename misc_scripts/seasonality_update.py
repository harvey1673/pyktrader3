import json
import copy
import logging
import datetime
import numpy as np
import pandas as pd
from pycmqlib3.analytics.tstool import calendar_label, lunar_label, xs_demean
from pycmqlib3.utility.dbaccess import load_factor_data
from pycmqlib3.utility.misc import prod2exch
from misc_scripts.factor_data_update import update_factor_db

prod_list = [
    'rb', 'hc', 'i', 'j', 'jm', 'FG', 
    'l', 'pp', 'v', 'TA', 'sc', 'eg',  'MA', 
    'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd',
]

class SeasonalStats:
    def __init__(self, season_rng):
        self.means = dict([(i, 0) for i in range(season_rng[0], season_rng[1]+1)])
        self.stds = dict([(i, 0) for i in range(season_rng[0], season_rng[1]+1)])
        self.counts = dict([(i, 0) for i in range(season_rng[0], season_rng[1]+1)])

    def initialize(self, counts, means, stds):    
        self.counts = counts
        self.means = means
        self.stds = stds

    def update(self, season, value, shift=0):        
        for s in range(season, season-shift-1, -1):
            if s in self.means:
                n = self.counts[s]
                old_mean = self.means[s]
                new_mean = old_mean + (value - old_mean) / (n + 1)
                self.means[s] = new_mean
                self.stds[s] = np.sqrt(((self.stds[s]**2 * n) + (value - old_mean) * (value - new_mean)) / (n + 1))
                self.counts[s] += 1
    
    def get_stats(self, season):        
        return [self.counts[season], self.means[season], self.stds[season]]


def seasonal_cal_update(price_df, spot_df,
                        cal_key='cal_mth',
                        cal_func=calendar_label,
                        label_field='label_mth',
                        shift=0,
                        season_rng=[1, 12],
                        product_list=prod_list, xs_weight=0.7):    
    fact_config = {'roll_label': 'hot', 'freq': 'd1', 'serial_key': 0, 'serial_no': 0}
    hist_seasonal_df = load_factor_data(product_list,
                            factor_list=[f"seazn_{cal_key}_days", f"seazn_{cal_key}_mean", f"seazn_{cal_key}_std"],
                            roll_label='hot',
                            start=price_df.index[0],
                            end=price_df.index[-1],
                            freq='d1',
                            db_table='fut_fact_data')    
    if len(hist_seasonal_df) > 0:        
        hist_seasonal_df = pd.pivot_table(hist_seasonal_df, 
                                        index='date', 
                                        columns=['product_code', 'fact_name'],
                                        values='fact_val', 
                                        aggfunc='last')
        hist_seasonal_df.index = pd.to_datetime(hist_seasonal_df.index)
        start_date = hist_seasonal_df.index[-1]
    else:
        start_date = pd.Timestamp('2008-01-02')

    df_pxchg = price_df.loc[:, price_df.columns.get_level_values(1)=='close'].pct_change().droplevel([1], axis=1)
    df_pxchg = df_pxchg[df_pxchg.index>start_date]
    if len(df_pxchg) > 0:
        update_seazn_list = []
        for prod in product_list:
            fact_config['product_code'] = prod
            fact_config['exch'] = prod2exch(prod)
            seasonal_stats = SeasonalStats(season_rng)
            if (prod, f'seazn_{cal_key}_mean') in hist_seasonal_df.columns:
                stat_df = cal_func(hist_seasonal_df[[(prod, f'seazn_{cal_key}_mean'), 
                                                        (prod, f'seazn_{cal_key}_std'), 
                                                        (prod, f'seazn_{cal_key}_days')]].droplevel([0], axis=1))
                
                stat_df = stat_df.groupby(label_field).last()
                mean_dict = stat_df[f'seazn_{cal_key}_mean'].to_dict()
                std_dict = stat_df[f'seazn_{cal_key}_std'].to_dict()
                days_dict = stat_df[f'seazn_{cal_key}_days'].to_dict()
                seasonal_stats.initialize(days_dict, mean_dict, std_dict)

            px_df = cal_func(df_pxchg[[prod+'c1']].dropna())
            px_df[label_field] = px_df[label_field].clip(season_rng[0], season_rng[1])
            stats_list = []            
            for i, row in px_df.iterrows():
                seasonal_stats.update(int(row[label_field]), row[prod+'c1'], shift)
                stats_list.append(seasonal_stats.get_stats(row[label_field]))
            stats_df = pd.DataFrame(stats_list, index=px_df.index, 
                                    columns=[f'seazn_{cal_key}_days', f'seazn_{cal_key}_mean', f'seazn_{cal_key}_std'])
            logging.info(f'updating seasonal factor {cal_key} for {prod}')
            for field in stats_df.columns:
                update_factor_db(stats_df, field, fact_config)
            stats_df.columns = pd.MultiIndex.from_tuples([(prod, col) for col in stats_df.columns])
            update_seazn_list.append(stats_df)
        curr_seasonal_df = pd.concat(update_seazn_list, axis=1)
        hist_seasonal_df = pd.concat([hist_seasonal_df, curr_seasonal_df])
    signal_df = pd.DataFrame(index=price_df.index, columns=product_list)
    for product in product_list:
        # ts = hist_seasonal_df[(product, f'seazn_{cal_key}_days')]
        # ts = ts[ts<20]
        # if len(ts) == 0:
        #     continue
        # sdate = ts.index[-1]
        signal_ts = hist_seasonal_df[(product, f'seazn_{cal_key}_mean')]/hist_seasonal_df[(product, f'seazn_{cal_key}_std')] * 16
        #signal_ts = signal_ts[signal_ts.index>sdate]
        signal_df[product] = signal_ts
    signal_df  = signal_df * (1-xs_weight) + xs_demean(signal_df) * xs_weight
    return signal_df





            

            




                
                
            
                 
                 
                 






