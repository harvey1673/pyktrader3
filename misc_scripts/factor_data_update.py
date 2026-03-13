import sys
import json
import copy
import logging
import csv
import numpy as np
from sqlalchemy import create_engine
from pycmqlib3.utility.dbaccess import dbconfig, mysql_replace_into, connect, load_factor_data
from pycmqlib3.utility import dataseries
from pycmqlib3.utility.misc import inst2product, prod2exch, inst2contmth, day_shift, \
    sign, is_workday, CHN_Holidays, nearby
import pycmqlib3.analytics.data_handler as dh
from pycmqlib3.analytics.tstool import *
from pycmqlib3.utility import base
from pycmqlib3.strategy.strat_util import generate_strat_position
from pycmqlib3.strategy.signal_repo import signal_buffer_config

ferrous_products_mkts = ['rb', 'hc', 'i', 'j', 'jm']
ferrous_mixed_mkts = ['ru', 'FG', 'ZC', 'SM', "SF"]
base_metal_mkts = ['cu', 'al', 'zn', 'pb', 'ni', 'sn']
precious_metal_mkts = ['au', 'ag']
ind_metal_mkts = ferrous_products_mkts + ferrous_mixed_mkts + base_metal_mkts
petro_chem_mkts = ['l', 'pp', 'v', 'TA', 'MA', 'bu']  # , 'sc', 'fu', 'eg']
ind_all_mkts = ind_metal_mkts + petro_chem_mkts
ags_oil_mkts = ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs']  # , 'b']
ags_soft_mkts = ['CF', 'CY', 'SR', 'jd', 'AP', 'UR', 'CJ']  # , 'sp', 'CJ', 'UR']
ags_all_mkts = ags_oil_mkts + ags_soft_mkts
eq_fut_mkts = ['IF', 'IH', 'IC', "IM"]
bond_fut_mkts = ['T', 'TF', 'TS', 'TL']
fin_all_mkts = eq_fut_mkts + bond_fut_mkts
commod_all_mkts = ind_all_mkts + ags_all_mkts + precious_metal_mkts
all_markets = commod_all_mkts + fin_all_mkts

trade_cont_map = {}

sim_start_dict = {'c': datetime.date(2011, 1, 1), 'm': datetime.date(2011, 1, 1),
                  'y': datetime.date(2011, 1, 1), 'l': datetime.date(2011, 1, 1), 'rb': datetime.date(2011, 1, 1),
                  'p': datetime.date(2011, 1, 1), 'cu': datetime.date(2011, 1, 1), 'al': datetime.date(2011, 1, 1),
                  'zn': datetime.date(2011, 1, 1), 'au': datetime.date(2011, 1, 1), 'v': datetime.date(2011, 1, 1),
                  'a': datetime.date(2011, 1, 1), 'ru': datetime.date(2011, 1, 1), 'ag': datetime.date(2012, 6, 1),
                  'i': datetime.date(2014, 1, 1), 'j': datetime.date(2012, 6, 1), 'jm': datetime.date(2013, 7, 1),
                  'CF': datetime.date(2012, 5, 1), 'TA': datetime.date(2012, 4, 15),
                  'PM': datetime.date(2013, 10, 1), 'RM': datetime.date(2013, 1, 1), 'SR': datetime.date(2013, 1, 1),
                  'FG': datetime.date(2013, 1, 1), 'OI': datetime.date(2013, 5, 1), 'RI': datetime.date(2013, 1, 1),
                  'WH': datetime.date(2014, 5, 1), 'pp': datetime.date(2014, 5, 1),
                  'IF': datetime.date(2010, 5, 1), 'MA': datetime.date(2012, 1, 1), 'TF': datetime.date(2019, 6, 1),
                  'IH': datetime.date(2015, 5, 1), 'IC': datetime.date(2015, 5, 1), 'cs': datetime.date(2015, 2, 1),
                  'jd': datetime.date(2014, 5, 1), 'ni': datetime.date(2015, 9, 1), 'sn': datetime.date(2017, 5, 1),
                  'ZC': datetime.date(2013, 11, 1), 'hc': datetime.date(2016, 4, 1), 'SM': datetime.date(2017, 1, 1),
                  'SF': datetime.date(2017, 9, 1), 'CY': datetime.date(2017, 9, 1), 'AP': datetime.date(2018, 1, 1),
                  'TS': datetime.date(2018, 9, 1), 'fu': datetime.date(2018, 9, 1), 'sc': datetime.date(2018, 8, 1),
                  'b': datetime.date(2018, 1, 1), 'pb': datetime.date(2016, 7, 1), 'bu': datetime.date(2015, 9, 15),
                  'T': datetime.date(2019, 4, 1), 'ss': datetime.date(2020, 5, 1), 'sp': datetime.date(2019, 5, 1),
                  'CJ': datetime.date(2019, 8, 9), 'UR': datetime.date(2019, 8, 9), 'SA': datetime.date(2020, 1, 1),
                  'eb': datetime.date(2020, 2, 1), 'eg': datetime.date(2019, 5, 1), 'rr': datetime.date(2019, 9, 1),
                  'pg': datetime.date(2020, 9, 5), 'lu': datetime.date(2020, 10, 1), 'nr': datetime.date(2020,1,1),
                  'lh': datetime.date(2021,5,1), 'PF': datetime.date(2021,1,1), 'PK': datetime.date(2021,4,1),
                  }

field_list = ['open', 'high', 'low', 'close', 'volume', 'openInterest', 'contract', 'shift']

port_pos_config = {
    # 'PTSIM1_FACTPORT_hot': {
    #     'pos_loc': 'C:/dev/pyktrader3/process/pt_test1',
    #     'roll': 'hot',
    #     'shift_mode': 2,
    #     'strat_list': [
    #         ('PTSIM1_FACTPORT1.json', 9000, 'd1'),
    #         ('PTSIM1_HRCRB.json', 18000, 'd1'),
    #         ('PTSIM1_LL.json', 5400, 'd1'),
    #         ('PTSIM1_FUNMTL.json', 4500, 'd1'),
    #         ('PTSIM1_FUNFER.json', 4500, 'd1'),
    #         ('PTSIM1_FUNBASE.json', 4500, 'd1'),
    #     ], },
    'PTSIM1_FACTPORT1_hot': {
        'pos_loc': 'C:/dev/pyktrader3/process/paper_sim1',
        'strat_list': [
            ('PTSIM1_FACTPORT1.json', 38000), #32000
            ('PTSIM1_SEAZN.json', 27000), #34000
            ('PTSIM1_EXCHWNT.json', 30000), #27000
            ('PTSIM1_HRCRB.json', 20000), #27000
            ('PTSIM1_LL.json', 30000), # 27000
            ('PTSIM1_LL2MR.json', 27000), 
            ('PTSIM1_SPDTF.json', 27000),
            ('PTSIM1_MR1Y.json', 27000), 
            ('PTSIM1_CNMAC1.json', 10000), #13000
            ('PTSIM1_CNMAC2.json', 14000),
            ('PTSIM1_FUNFER.json', 30000), # 37000
            ('PTSIM1_FERSPD.json', 90000), # 110000
            ('PTSIM1_AUSPD.json', 80000),
            ('PTSIM1_FUNBASE.json', 40000),
            ('PTSIM1_FUNENE.json', 10000), # 7000
            ('PTSIM1_FUNMTL.json', 25000),
            ('PTSIM1_BND1.json', 50000), # 
            ('PTSIM1_MANUEL_TRADING.csv', 1)
        ], },
}

pos_chg_notification = ['PTSIM1_FACTPORT1_hot',]


def _prepare_factor_df(xdf, field, config, start_date=None, end_date=None):
    df = xdf.copy()
    for key in config:
        df[key] = config[key]
    df['fact_name'] = field
    df['fact_val'] = df[field]
    df = df.dropna().reset_index()
    if start_date:
        df = df[pd.to_datetime(df['date']) >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['date']) <= pd.to_datetime(end_date)]
    if len(df) == 0:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df[['product_code', 'roll_label', 'exch', 'fact_name', 'freq', 'date', 'serial_no', 'serial_key', 'fact_val']]
    return df


class FactorDBBatchWriter:
    def __init__(self, dbtable='fut_fact_data', flavor='mysql', flush_rows=25000):
        self.dbtable = dbtable
        self.flavor = flavor
        self.flush_rows = flush_rows
        self.buffer = []
        self.buffer_rows = 0
        if flavor == 'mysql':
            self.conn = create_engine(
                'mysql+mysqlconnector://{user}:{passwd}@{host}/{dbase}'.format(user=dbconfig['user'],
                                                                               passwd=dbconfig['password'],
                                                                               host=dbconfig['host'],
                                                                               dbase=dbconfig['database']),
                echo=False)
            self.func = mysql_replace_into
        else:
            self.conn = connect(**dbconfig)
            self.func = None

    def add(self, df):
        if len(df) == 0:
            return
        self.buffer.append(df)
        self.buffer_rows += len(df)
        if self.buffer_rows >= self.flush_rows:
            self.flush()

    def flush(self):
        if self.buffer_rows == 0:
            return
        out_df = pd.concat(self.buffer, axis=0, ignore_index=True)
        out_df.to_sql(self.dbtable,
                      con=self.conn,
                      if_exists='append',
                      index=False,
                      method=self.func,
                      chunksize=self.flush_rows)
        self.buffer = []
        self.buffer_rows = 0

    def close(self):
        self.flush()
        if self.flavor == 'mysql':
            self.conn.dispose()


def update_factor_db(xdf, field, config, dbtable='fut_fact_data', flavor='mysql',
                     start_date=None, end_date=None, db_writer=None):
    logging.info('updating factor_name=%s' % field)
    df = _prepare_factor_df(xdf, field, config, start_date=start_date, end_date=end_date)
    if len(df) == 0:
        return
    if db_writer is not None:
        db_writer.add(df)
        return
    #insert_df_to_sql(df, dbtable, is_replace=True)
    if flavor == 'mysql':
        conn = create_engine(
            'mysql+mysqlconnector://{user}:{passwd}@{host}/{dbase}'.format(user=dbconfig['user'],
                                                                           passwd=dbconfig['password'],
                                                                           host=dbconfig['host'],
                                                                           dbase=dbconfig['database']),
            echo=False)
        func = mysql_replace_into
    else:
        conn = connect(**dbconfig)
        func = None
    df.to_sql(dbtable, con=conn, if_exists='append', index=False, method=func, chunksize=5000)
    if flavor == 'mysql':
        conn.dispose()


def create_strat_json(product_list, freq, roll_rule, factor_repo,
                      filename="C:\\dev\\data\\MM_FACT_PORT.json",
                      name='default'):
    strat_data = {}
    strat_data["class"] = "pycmqlib3.strategy.strat_factor_port.FactorPortTrader"
    strat_config = {}
    strat_config['name'] = name
    if freq == 'd':
        strat_config['freq'] = 's1'
    else:
        strat_config['freq'] = freq
    strat_config['roll_label'] = 'CAL_' + roll_rule
    strat_config['factor_repo'] = factor_repo
    strat_config['vol_win'] = 20
    strat_config['fact_db_table'] = 'fut_fact_data'
    strat_config['exec_bar_list'] = [1510]
    strat_config['pos_scaler'] = 1000

    assets = []
    for asset in product_list:
        asset_data = {}
        asset_data['underliers'] = [trade_cont_map[asset][0]]
        asset_data['volumes'] = [1]
        asset_data['alloc_w'] = 1.0
        asset_data['prev_underliers'] = ''
        assets.append(asset_data)
    strat_config['assets'] = assets

    filtered_factors = {}
    for fact_name in factor_repo:
        if factor_repo[fact_name]['type'] in ['xs', 'ts']:
            filtered_factors[fact_name] = copy.copy(factor_repo[fact_name])
    strat_config['factor_repo'] = filtered_factors
    strat_data['config'] = strat_config
    with open(filename, 'w') as f:
        json.dump(strat_data, f, indent=4)


def update_port_position(run_date=datetime.date.today()):
    results = {
        'pos_update': {},
        'details': {}
    }
    pos_date = day_shift(run_date, '1b', CHN_Holidays)
    pre_date = day_shift(pos_date, '-1b', CHN_Holidays)
    pos_date_str = pos_date.strftime('%Y%m%d')
    pre_date_str = pre_date.strftime('%Y%m%d')
    for port_name in port_pos_config.keys():
        target_pos = {}
        pos_by_strat = {}
        pos_loc = port_pos_config[port_name]['pos_loc']
        curr_signal_file = f'{pos_loc}/curr_signal_{pre_date_str}.json'
        curr_signal = {}
        next_signal = {}
        try:
            with open(curr_signal_file, 'r') as fp:
                curr_signal = json.load(fp)
        except:
            pass

        port_file = port_name
        for strat_file, pos_scaler in port_pos_config[port_name]['strat_list']:
            config_file = f'{pos_loc}/settings/{strat_file}'
            if ".json" in strat_file:
                with open(config_file, 'r') as fp:
                    strat_conf = json.load(fp)
                strat_args = strat_conf['config']
                assets = strat_args['assets']
                vol_key = strat_conf.get("vol_key", "pct_vol")
                roll = strat_conf.get("roll_label", "hot")
                repo_type = strat_args.get('repo_type', 'asset')
                freq = strat_conf.get("freq", "d1")
                hist_fact_lookback = strat_conf.get("hist_fact_lookback", 20)
                factor_repo = strat_args['factor_repo']
                product_list = []
                for asset_dict in assets:
                    under = asset_dict["underliers"][0]
                    product = inst2product(under)
                    product_list.append(product)

                logging.info(f"updating position for {strat_file}...")
                res = generate_strat_position(run_date, product_list, factor_repo,
                                            repo_type=repo_type,
                                            roll_label=roll,
                                            pos_scaler=pos_scaler,
                                            freq=freq,
                                            hist_fact_lookback=hist_fact_lookback,
                                            vol_key=vol_key,
                                            curr_signal=curr_signal,
                                            signal_config=signal_buffer_config)
                strat_target = res['target_pos']
                next_signal = {**(res['curr_signal'].to_dict()), **next_signal}
                results['details'][f'{port_name}:{strat_file}'] = res['pos_sum'].T
            elif ".csv" in config_file:
                with open(config_file) as f:
                    strat_target= {k: int(v) * pos_scaler  for k, v in csv.reader(f)}
   
            pos_by_strat[strat_file] = strat_target
            for prod in strat_target:
                if prod not in target_pos:
                    target_pos[prod] = 0
                if ~np.isnan(strat_target[prod]):
                    target_pos[prod] += strat_target[prod]

        for prodcode in target_pos:
            if np.isnan(target_pos[prodcode]):
                target_pos[prodcode] = 0
                continue
            if prodcode in ['ps']:
                target_pos[prodcode] = int((target_pos[prodcode] / 10 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 10
            elif prodcode in ['lc']:
                target_pos[prodcode] = int((target_pos[prodcode] / 5 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 5
            elif prodcode in ['SH', 'PR']:
                target_pos[prodcode] = int((target_pos[prodcode] / 4 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 4
            elif prodcode in ['MA', 'PX', 'TA', 'PF', 'eb', 'eg', 'pg', 'l', 'v', 'pp']:
                target_pos[prodcode] = int((target_pos[prodcode] / 8 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 8
            else:
                target_pos[prodcode] = int(target_pos[prodcode] + (0.5 if target_pos[prodcode] > 0 else -0.5))

        posfile = '%s/%s_%s.json' % (pos_loc, port_file, pos_date_str)
        with open(posfile, 'w') as ofile:
            json.dump(target_pos, ofile, indent=4)

        next_signal_file = f'{pos_loc}/curr_signal_{pos_date_str}.json'
        with open(next_signal_file, 'w') as ofile:
            json.dump(next_signal, ofile, indent=4)

        stratfile = '%s/pos_by_strat_%s_%s.json' % (pos_loc, port_file, pos_date_str)
        with open(stratfile, 'w') as ofile:
            json.dump(pos_by_strat, ofile, indent=4)

        if port_file in pos_chg_notification:
            with open('%s/%s_%s.json' % (pos_loc, port_file, pre_date_str), 'r') as fp:
                curr_pos = json.load(fp)
            pos_df = pd.DataFrame({'cur': curr_pos, 'tgt': target_pos})
            pos_df['diff'] = pos_df['tgt'] - pos_df['cur']
            results['pos_update'][port_file] = pos_df
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        now = datetime.datetime.now()
        tday = now.date()
        if (not is_workday(tday, 'CHN')) or (now.time() < datetime.time(14, 59, 0)):
            tday = day_shift(tday, '-1b', CHN_Holidays)
    folder = "C:/dev/data/"
    name = "pf_position_update"
    base.config_logging(folder + name + ".log", level=logging.INFO,
                        format='%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                        to_console=True,
                        console_level=logging.INFO)
    logging.info("running portfolio position for %s" % str(tday))
    res = update_port_position(run_date=tday)
