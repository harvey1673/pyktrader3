import urllib.request, json, datetime
import pandas as pd
import dbaccess
import sec_bits

DAILY_SALES_DATA_URL = "https://www.banksteel.com/api/bigdata/home/v4/trade/order/half"
WEEKLY_SALES_DATA_URL = "https://www.banksteel.com/api/bigdata/home/v4/trade/deal/week"

PROXIES = {'http':'http://%s:%s@10.252.22.102:4200' % (sec_bits.PROXY_CREDENTIALS['user'], sec_bits.PROXY_CREDENTIALS['passwd']),
           'https':'https://%s:%s@10.252.22.102:4200' % (sec_bits.PROXY_CREDENTIALS['user'], sec_bits.PROXY_CREDENTIALS['passwd'])}


def get_json_from_url(url_str):
    proxy = urllib.request.ProxyHandler(PROXIES)
    auth = urllib.request.HTTPBasicAuthHandler()
    opener = urllib.request.build_opener(proxy, auth, urllib.request.HTTPHandler)
    urllib.request.install_opener(opener)
    with urllib.request.urlopen(url_str) as url:
        data = json.loads(url.read().decode())
    return data

def get_hourly_sales_data(save_db = False):
    data = get_json_from_url(DAILY_SALES_DATA_URL)
    tday = datetime.date.today()
    out_df = pd.DataFrame()
    for key, day_shift in zip(['today'], [0]):
        df = pd.DataFrame.from_dict(data[key], orient='columns')
        df.index.name = 'datetime'
        df.reset_index(inplace=True)
        curr_date = tday - datetime.timedelta(days = day_shift)
        df['datetime'] = df['datetime'].apply(lambda x: \
                            datetime.datetime.strptime(x, '%H:%M').replace(\
                                year = curr_date.year, month = curr_date.month, day = curr_date.day))
        out_df = out_df.append(df)
    out_df['category'] = 'total'
    out_df = out_df.sort_values(by = 'datetime')
    conn = dbaccess.connect(**dbaccess.misc_dbconfig)
    if save_db:
        out_df.to_sql("banksteel_hourly_sales", conn, if_exists='append', index = False)
    conn.close()
    return out_df

def get_daily_sales_data(save_db = False):
    data = get_json_from_url(WEEKLY_SALES_DATA_URL)
    tday = datetime.date.today()
    out_df = pd.DataFrame()
    for key, day_shift in zip(['currentWeek', 'lastWeek'], [0, 1]):
        df = pd.DataFrame.from_dict(data[key], orient='columns')
        df.index.name = 'date'
        df.reset_index(inplace=True)
        curr_date = tday - datetime.timedelta(days=day_shift*7)
        d_shift = (curr_date.weekday() + 1) % 7
        wk_start = curr_date - datetime.timedelta(days=d_shift)
        df['date'] = df['date'].apply(lambda x: wk_start + datetime.timedelta(days = int(x)-1))
        out_df = out_df.append(df)
    out_df['category'] = 'total'
    out_df = out_df.sort_values(by='date')
    conn = dbaccess.connect(**dbaccess.misc_dbconfig)
    if save_db:
        out_df.to_sql("banksteel_daily_sales", conn, if_exists='append', index=False)

    return out_df