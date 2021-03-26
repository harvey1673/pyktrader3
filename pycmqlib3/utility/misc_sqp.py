import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm
import sys
if sys.version_info[0] < 3:
    from io import StringIO
else:
    from io import StringIO
from . import misc, dbaccess
from pycmqlib3.analytics import ts_tool

BBG_MAP_FILE = "C:\\dev\\cmqlib3\\data_store\\sqp_product_mapping.csv"
MAP_DF = pd.read_csv(BBG_MAP_FILE)
BBG_MAP = MAP_DF[['bbg_code', 'product_code']].set_index('bbg_code').transpose().to_dict(orient='records')[0]
RIC_MAP = MAP_DF[['ric_code', 'product_code']].dropna().set_index('ric_code').transpose().to_dict(orient='records')[0]
LEVEL_DICT = {'h': 2, 'm': 1, 'l': 0}

margin_dict = { 'IF': 0.1, 'IC': 0.1, 'IH': 0.01, 'T': 0.02, 'TF':0.012, 'TS': 0.005,
                'cu': 0.1, 'al':0.1, 'zn': 0.1, 'pb': 0.1, 'sn': 0.1, 'ni': 0.1, 'ss': 0.1,
                'au': 0.08, 'ag': 0.12, 'rb': 0.08, 'hc': 0.08, 'ru': 0.08, 'bu': 0.1, 'fu': 0.1, 'sp': 0.07,
                'i' : 0.11, 'j': 0.09, 'jm': 0.09, 'pg': 0.11, 'eb': 0.12, 'eg': 0.11,
                'l' : 0.11, 'v': 0.09, 'pp': 0.11, 'a': 0.09, 'b': 0.08, 'rr': 0.06,
                'm':  0.08, 'y' : 0.08, 'p': 0.09, 'c':  0.07, 'cs': 0.07, 'jd': 0.07,
                'MA': 0.07, 'ZC': 0.05, 'TA': 0.06, 'FG': 0.06, 'SA': 0.06,
                'SM': 0.07, 'SF': 0.07, 'AP': 0.08, 'CJ': 0.07, 'RM': 0.06, 'OI': 0.06,
                'SR': 0.07, 'CF': 0.07, 'CY': 0.07, 'UR': 0.05,
                'sc': 0.1, 'lu': 0.1, 'nr': 0.08}

def bbgcode2product(bbg_code, mapfile = None, param = 1):
    if mapfile:
        map_df = pd.read_csv(mapfile)
        bbg_map = map_df[['bbg_code', 'product_code']].set_index('bbg_code').transpose().to_dict(orient='records')
    else:
        bbg_map = BBG_MAP
    bbg_product = bbg_code[:(len(bbg_code)-1 - param)]
    product_code = bbg_map[bbg_product]
    return product_code

def ric2product(ric_code, mapfile = None, param = 1):
    if mapfile:
        map_df = pd.read_csv(mapfile)
        ric_map = map_df[['ric_code', 'product_code']].dropna().set_index('ric_code').transpose().to_dict(orient='records')
    else:
        ric_map = RIC_MAP
    ric_product = ric_code[:(len(ric_code)-1-param)]
    product_code = ric_map[ric_product]
    return product_code

def xcode2inst(xcode, mapfile = None, type = "bbg", param = 1):
    if type == 'ric':
        product_code = ric2product(xcode, mapfile, param)
    else:
        product_code = bbgcode2product(xcode, mapfile, param)
    exch = misc.prod2exch(product_code)
    cont_year = int(xcode[-param:])
    cont_mth = misc.month_code_map[xcode[-(1+param)].lower()]
    if exch in ['CZCE']:
        cont_year = cont_year % 10
        instID = product_code + "%01d" % cont_year + "%02d" % cont_mth
    else:
        if param == 1:
            cont_year = cont_year + 20
            if cont_year > 28:
                cont_year = cont_year - 10
        cont_mth = cont_year * 100 + cont_mth
        instID = product_code + str(cont_mth)
    return instID

def load_trade_pos(file = "C:\\dev\\data\\sqp_data\\", type = 'bbg'):
    if type == 'ric':
        xfunc = ric2product
        field = 'ric_code'
    else:
        xfunc = bbgcode2product
        field = 'bloomberg_id'
    trade_df = pd.read_csv(file)
    trade_df.dropna(subset = [field], inplace = True)
    trade_df['instID'] = trade_df[field].apply(lambda x: xcode2inst(x, type = type))
    trade_df['product_code'] = trade_df[field].apply(lambda x: xfunc(x))
    trade_df['exch'] = trade_df['product_code'].apply(lambda x: misc.prod2exch(x))
    if 'execution_price' in trade_df.columns:
        trade_df['trade_price'] = trade_df['execution_price'].apply(lambda x: float(x))
        trade_df['trade_date'] = trade_df['date'].apply(lambda x: datetime.datetime.strptime(x, "%d-%m-%Y").date())
    trade_df['volume'] = trade_df['quantity'].apply(lambda x: float(x))
    flag = trade_df['direction'].str.startswith("S") | trade_df['direction'].str.startswith("s")
    trade_df.loc[flag, 'volume'] = - trade_df.loc[flag, 'volume']
    return trade_df

def load_epoch_trade_pos(file = "C:\\dev\\data\\sqp_data\\"):
    trade_df = pd.read_csv(file)
    trade_df.columns = [col.lower() for col in trade_df.columns]
    trade_df.dropna(subset=["ric"], inplace=True)
    trade_df.rename(columns = {'amount': 'position', 'symbol': 'instID', 'price': 'trade_price', 'side': 'direction'}, inplace=True)
    trade_df['product_code'] = trade_df['instID'].apply(lambda x: misc.inst2product(x))
    trade_df['exch'] = trade_df['product_code'].apply(lambda x: misc.prod2exch(x))
    flag = trade_df['direction'].str.startswith("S") | trade_df['direction'].str.startswith("s")
    trade_df.loc[flag, 'position'] = - trade_df.loc[flag, 'position']
    return trade_df

def load_agg_pos_from_file(tday = datetime.date.today(), file_loc = "C:\\dev\\pycmqlib3\\paper_test\\positions\\"):
    filename = file_loc + "aggregate_position_conf_" + datetime.datetime.strftime(tday,"%d%m%Y") + '.csv'
    pos_df = pd.read_csv(filename)
    pos_df['contract'] = pos_df['bloomberg_id'].apply(lambda x: xcode2inst(x, type = 'bbg'))
    pos_df['product'] = pos_df['bloomberg_id'].apply(lambda x: bbgcode2product(x))
    pos_df['exchange'] = pos_df['product'].apply(lambda x: misc.prod2exch(x))
    pos_df['position'] = pos_df['quantity'].apply(lambda x: float(x))
    flag = pos_df['direction'].str.startswith("S") | pos_df['direction'].str.startswith("s")
    pos_df.loc[flag, 'position'] = - pos_df.loc[flag, 'position']
    #pos_df['date'] = tday
    pos_df['lot_size'] = pos_df['product'].apply(lambda x: misc.product_lotsize[x])
    pos_df = pos_df[['product', 'contract', 'exchange', 'lot_size', 'position']]
    return pos_df

def load_epoch_agg_pos(tday = datetime.date.today(), file_loc = "C:\\dev\\pycmqlib3\\paper_test\\positions\\"):
    filename = file_loc + datetime.datetime.strftime(tday, "%Y%m%d") + '_position.csv'
    pos_df = pd.read_csv(filename)
    pos_df.columns = ['_'.join(col.lower().split(' ')) for col in pos_df.columns]
    pos_df.rename(columns = {'p': 'quantity'}, inplace=True)
    pos_df['contract'] = pos_df['instrument'].apply(lambda x: xcode2inst(x, type = 'ric', param=2))
    pos_df['product'] = pos_df['contract'].apply(lambda x: misc.inst2product(x))
    pos_df['exchange'] = pos_df['product'].apply(lambda x: misc.prod2exch(x))
    pos_df['position'] = pos_df['quantity'].apply(lambda x: float(x))
    pos_df['lot_size'] = pos_df['product'].apply(lambda x: misc.product_lotsize[x])
    pos_df = pos_df[['product', 'contract', 'exchange', 'lot_size', 'position']]
    return pos_df

def calc_pos_val(df, ref_date, lookback = 30, var_percentile = 95, use_contract = True):
    start_date = misc.day_shift(ref_date, '-%sb' % str(lookback))
    inst_list = []
    map_field = ''
    for (asset, cont) in zip(df['product'], df['contract']):
        nb = 1
        rr = '-35b'
        if asset in ['cu', 'al', 'zn', 'pb', 'sn']:
            nb = 1
            rr = '-35b'
        elif asset in ['IF', 'IH', 'IC']:
            rr = '-2b'
        elif asset in ['au', 'ag', 'bu']:
            rr = '-25b'
        elif asset in ['TF', 'T']:
            rr = '-20b'
        elif asset in ['ni']:
            rr = '-40b'
        args = {'n': nb, 'roll_rule': rr, 'freq': 'd', 'need_shift': 2}
        if use_contract:
            map_field = 'contract'
            inst_list.append([cont, 'fut_daily', 'instID', {}])
        else:
            map_field = 'product'
            inst_list.append([asset, 'fut_daily', 'instID', args])
    xdf = ts_tool.get_multiple_data(inst_list, start_date, ref_date)
    price_dict = xdf.loc[ref_date, :].to_dict()
    df['price'] = df[map_field].apply(lambda x: price_dict[x])
    df['pos_val'] = df['price'] * df['lot_size'] * df['position']
    var_ratio = norm.ppf(var_percentile/100.0)
    cov_mat = np.log(xdf).diff().fillna(0.0).cov()
    cov_mat = cov_mat.loc[df[map_field], df[map_field]]
    df['pos_bias'] = np.sqrt(np.diag(cov_mat)) * df['pos_val']
    partial = np.dot(df['pos_val'].T, cov_mat)
    vol_p = np.sqrt(np.dot(partial, df['pos_val']))
    df['var95_30d'] = -partial * var_ratio/vol_p * df['pos_val']
    return df

def update_pos_table(tday = datetime.date.today(), file_loc = "Z:\\Ferrous\\SquarePoint\\20190925\\", strategy = ["SQP", 'SQP']):
    if strategy[0] == 'SQP':
        df = load_agg_pos_from_file(tday, file_loc)
    elif strategy[0] == 'EPOCH':
        df = load_epoch_agg_pos(tday, file_loc)
    df = calc_pos_val(df, tday, lookback = 30, var_percentile = 95)
    df['strategy'] = strategy[0]
    df['sub_strategy'] = strategy[1]
    df['date'] = tday
    df = df[['date', 'strategy', 'sub_strategy', 'product', 'contract', 'exchange', 'lot_size', 'position', 'pos_val', 'pos_bias',
             'var95_30d']]
    cnx = dbaccess.connect(**dbaccess.pos_dbconfig)
    df.to_sql('daily_position', cnx, 'sqlite', if_exists='append', index=False)
    cnx.close()

def update_risk_in_pos_table(start_date, end_date, strategy = ["SQP", 'SQP']):
    num_days = (end_date - start_date).days + 1
    drange = [ start_date + datetime.timedelta(days = x) for x in range(num_days)]
    drange = [ d for d in drange if misc.is_workday(d, 'CHN')]
    cnx = dbaccess.connect(**dbaccess.pos_dbconfig)
    for d in drange:
        df = dbaccess.load_agg_pos_from_db(start = d, end = d, strategy = strategy, out_cols=['position'], db_table="daily_position")
        if len(df) > 0:
            df = calc_pos_val(df, d, lookback=30, var_percentile=95)
            df['strategy'] = strategy[0]
            df['sub_strategy'] = strategy[1]
            df['date'] = d
            df = df[['date', 'strategy', 'sub_strategy', 'product', 'contract', 'exchange', 'lot_size', 'position', 'pos_val', 'var95_30d']]
            df.to_sql('daily_position', cnx, 'sqlite', if_exists='append', index=False)
    cnx.close()

def read_daily_trade(tday = datetime.date.today(), email_domain = "@squarepoint-capital.com", \
                          save_file = "C:\\dev\\pycmqlib3\\paper_test\\SQP1", trade_level = 2, trade_file = None):
    df = pd.DataFrame()
    if trade_file:
        df = pd.read_csv(trade_file).dropna()
    else:
        import win32com.client
        outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
        inbox_folder = outlook.GetDefaultFolder(6)
        for idx in range(inbox_folder.Items.Count,1, -1):
            email_obj = inbox_folder.Items.Item(idx)
            if ("Daily Recommendations" in email_obj.Subject) and (email_domain in email_obj.SenderEmailAddress.lower()):
                etime = email_obj.ReceivedTime
                email_dt = datetime.datetime(etime.year, etime.month, etime.day, etime.hour, etime.minute, etime.second)
                if email_dt.date() == tday:
                    ebody = "\r\n".join(email_obj.Body.split('\r\n')[6:-2])
                    ebody = StringIO(ebody)
                    df = pd.read_csv(ebody, sep="\t").dropna()
                    break
    if len(df)>0:
        df.columns = [col.strip() for col in df.columns]
        df  = df.rename(columns={"ric":"ric_code", "name": "desc"})
        df['instID'] = df["exchange_ticker"].str.strip()
        df['conviction'] = df['conviction'].apply(lambda x: x.strip().lower()[0])

        df['conv_level'] = df['conviction'].apply(lambda x: LEVEL_DICT[x])
        df = df[df['conv_level'] >= trade_level]
        df = df[['instID', 'ric_code', 'position', 'conviction', 'desc']].dropna()
        if save_file:
            filename = save_file + "_bulktrade_%s.json" % datetime.datetime.strftime(tday, "%y%m%d")
            df[['instID', 'position']].set_index('instID').to_json(filename, orient='index')
    return df

def read_fill_from_email(tday, email_domain = "@cargill.com", \
                         save_file = "C:\\dev\\pycmqlib3\\paper_test\\positions\\"):
    import win32com.client
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox_folder = outlook.GetDefaultFolder(6)
    df = None
    for idx in range(inbox_folder.Items.Count,1, -1):
        email_obj = inbox_folder.Items.Item(idx)
        if ("Fill Confirmation" in email_obj.Subject) and (datetime.datetime.strftime(tday,"%d%m%Y") in email_obj.Subject):
            etime = email_obj.ReceivedTime
            email_dt = datetime.datetime(etime.year, etime.month, etime.day, etime.hour, etime.minute, etime.second)
            if email_dt.date() == tday:
                for att in email_obj.Attachments:
                    try:
                        att.SaveAsFile(save_file + att.FileName)
                        if "daily_trade_conf" in att.FileName:
                            df = load_trade_pos(save_file + att.FileName)
                    except:
                        print("Attachment Download Failed, filename = %s" % att.FileName)
                        pass
            return df
    return None

def daily_position_update(tday = datetime.date.today(), update_level = 0, \
                          scaler_old = 1.0, scaler_new = 1.0, \
                          save_file = "C:\\dev\\pycmqlib3\\paper_test\\SQP1", \
                          pos_file = "C:\\dev\\pycmqlib3\\paper_test\\positions\\", \
                          out_json = None,
                          trade_file = None,
                          calc_margin = True):
    trade_df = read_daily_trade(tday = tday, save_file = save_file, trade_file = trade_file)
    LEVEL_DICT = {'h':2, 'm': 1, 'l': 0}
    trade_df['conv_level'] = trade_df['conviction'].apply(lambda x: LEVEL_DICT[x])
    trade_df = trade_df[trade_df['conv_level'] >= update_level]
    trade_df = trade_df[['instID', 'position']]
    prev_date = misc.day_shift(tday, '-1b', misc.CHN_Holidays)
    if pos_file:
        prev_df = load_agg_pos_from_file(prev_date, file_loc = pos_file)
    else:
        prev_df = dbaccess.load_agg_pos_from_db(start = prev_date, end = prev_date, strategy=('SQP', 'SQP'))
    prev_df = prev_df[['contract', 'position']]
    prev_df = prev_df.rename(columns={'contract': 'instID'})
    df = prev_df.merge(trade_df, how='outer', left_on='instID', right_on='instID', suffixes=('_old','_trade')).fillna(0.0)
    df['position_new'] = df['position_old'] + df['position_trade']
    df['pos_new'] = (df['position_new'] * scaler_new).apply(lambda x: int(x + (x % (1 if x >= 0 else -1))))
    df['pos_old'] = (df['position_old'] * scaler_old).apply(lambda x: int(x + (x % (1 if x >= 0 else -1))))
    df['position'] = df['pos_new'] - df['pos_old']
    if out_json:
        df[['instID', 'position']].set_index('instID').to_json(out_json, orient='index')
    if calc_margin:
        inst_list = list(df['instID'])
        conn = dbaccess.connect(**dbaccess.dbconfig)
        inst_close_dict = {}
        inst_margin_dict = {}
        for inst in inst_list:
            adf = dbaccess.load_daily_data_to_df(conn, 'fut_daily', inst, misc.day_shift(tday, '-5b', misc.CHN_Holidays), tday)
            inst_close_dict[inst] = float(adf['close'][-1]) * misc.product_lotsize[misc.inst2product(inst)]
            inst_margin_dict[inst] = margin_dict[misc.inst2product(inst)]
        df['price'] = df['instID'].apply(lambda x: inst_close_dict[x])
        df['margin_ratio'] = df['instID'].apply(lambda x: inst_margin_dict[x])
        df['margin_old'] = df['price'] * df['margin_ratio'] * abs(df['pos_old'])
        df['margin_new'] = df['price'] * df['margin_ratio'] * abs(df['pos_new'])
    return df

def read_epoch_daily_trade(tday, label = 'am', email_domain = "@epochcapital.com.au", \
                         save_file = "C:\\dev\\pycmqlib3\\paper_test\\positions\\"):
    import win32com.client
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
    inbox_folder = outlook.GetDefaultFolder(6)
    df = None
    for idx in range(inbox_folder.Items.Count,1, -1):
        email_obj = inbox_folder.Items.Item(idx)
        if ("CHYCAR Orders" in email_obj.Subject) and (datetime.datetime.strftime(tday,"%Y%m%d") in email_obj.Subject) \
                and (email_domain in email_obj.SenderEmailAddress.lower()):
            etime = email_obj.ReceivedTime
            email_dt = datetime.datetime(etime.year, etime.month, etime.day, etime.hour, etime.minute, etime.second)
            if (email_dt.date() == tday) and (
                    ((label == 'am') and (etime.hour < 12)) or ((label == 'pm') and (etime.hour > 12) and etime.hour < 15) \
                    or ((label == 'nt') and (etime.hour > 20))):
                ebody = email_obj.Body.split('\r\n')
                start_idx = -1
                for idx, line in enumerate(ebody):
                    if line[:7] == 'Account':
                        start_idx = idx
                        break
                data_txt = '\r\n'.join(ebody[start_idx:])
                ebody = StringIO(data_txt)
                tdf = pd.read_csv(ebody, sep=",").dropna()
                filename = save_file + "epoch_trade_%s_%s.csv" % (datetime.datetime.strftime(tday,"%Y%m%d"), label)
                tdf.to_csv(filename)
                df = load_epoch_trade_pos(filename)
                df = df[['instID', 'direction', 'position']]
                return df
    return df

def epoch_pos_update(tday = datetime.date.today(), label = 'am', scaler_old = 1.0, scaler_new = 1.0, \
                          email_domain = "@epochcapital.com.au", \
                          save_file = "C:\\dev\\pycmqlib3\\paper_test\\SQP1", \
                          pos_file = "Z:\\dev\\pycmqlib3\\paper_test\\positions\\", \
                          out_json = None,
                          calc_margin = True):
    prev_date = misc.day_shift(tday, '-1b', misc.CHN_Holidays)
    if pos_file:
        prev_df = load_epoch_agg_pos(prev_date, file_loc = pos_file)
    else:
        prev_df = dbaccess.load_agg_pos_from_db(start = prev_date, end = prev_date, strategy=('EPOCH', 'CHYCAR'))
    prev_df = prev_df.rename(columns={'contract': 'instID'})
    prev_df = prev_df[['instID', 'position']]
    if label == 'pm':
        prev_df = prev_df.set_index('instID')
        trade_df = read_epoch_daily_trade(tday=tday, label='am', email_domain = email_domain, save_file = save_file)
        trade_df = trade_df[['instID', 'position']].set_index('instID')
        prev_df = prev_df.add(trade_df, fill_value = 0)
        prev_df = prev_df.reset_index()
    trade_df = read_epoch_daily_trade(tday=tday, label=label, email_domain = email_domain, save_file=save_file)
    trade_df = trade_df[['instID', 'position']]
    df = prev_df.merge(trade_df, how='outer', left_on='instID', right_on='instID', suffixes=('_old','_trade')).fillna(0.0)
    df['position_new'] = df['position_old'] + df['position_trade']
    df['pos_new'] = (df['position_new'] * scaler_new).apply(lambda x: int(x + (x % (1 if x >= 0 else -1))))
    df['pos_old'] = (df['position_old'] * scaler_old).apply(lambda x: int(x + (x % (1 if x >= 0 else -1))))
    df['position'] = df['pos_new'] - df['pos_old']
    if out_json:
        df[['instID', 'position']].set_index('instID').to_json(out_json, orient='index')
    if calc_margin:
        inst_list = list(df['instID'])
        conn = dbaccess.connect(**dbaccess.dbconfig)
        inst_close_dict = {}
        inst_margin_dict = {}
        for inst in inst_list:
            adf = dbaccess.load_daily_data_to_df(conn, 'fut_daily', inst, misc.day_shift(tday, '-5b', misc.CHN_Holidays), tday)
            inst_close_dict[inst] = float(adf['close'][-1]) * misc.product_lotsize[misc.inst2product(inst)]
            inst_margin_dict[inst] = margin_dict[misc.inst2product(inst)]
        df['price'] = df['instID'].apply(lambda x: inst_close_dict[x])
        df['margin_ratio'] = df['instID'].apply(lambda x: inst_margin_dict[x])
        df['margin_old'] = df['price'] * df['margin_ratio'] * abs(df['pos_old'])
        df['margin_new'] = df['price'] * df['margin_ratio'] * abs(df['pos_new'])
    return df

def get_agg_risk(start, end, risk_names = ['pos_val'], strategy = ('SQP', 'SQP')):
    df = dbaccess.load_agg_pos_from_db(start = start, end = end, strategy = strategy)
    df['volume'] = df['lot_size'] * df['position']
    df['sector'] = df['product'].apply(lambda x: misc.product_class_map[x][0])
    df['subsector'] = df['product'].apply(lambda x: misc.product_class_map[x][1])
    res = {}
    for risk_name in risk_names:
        res[risk_name] = {}
        for col in ['product', 'sector', 'subsector']:
            res[risk_name][col] = pd.pivot_table(df, index='date', columns=[col], values=risk_name, aggfunc='sum', fill_value=0).astype('double')
            res[risk_name][col].reset_index(inplace = True)
            if col in ['sector', 'subsector']:
                res[risk_name][col]['Total'] = res[risk_name][col].sum(axis=1)
    return res