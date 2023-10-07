import zipfile
from io import BytesIO, StringIO
import datetime
import pandas as pd
import json
import workdays
import requests
from pycmqlib3.utility.misc import rev_month_code_map, month_code_map
from pycmqlib3.utility.dbaccess import save_data
FUTURE_COLUMNS = ['instID', 'exch', 'date', 'open', 'high', 'low', 'close', 'settle', 'volume', 'openInterest']
SGX_future_products = ['FEF', 'M65F']
save_json_file = "C:/dev/data/sgx_date_map.json"


def futures_sgx_daily(page_range=[6760, 6770]):
    try:
        with open(save_json_file, 'r') as infile:
            date_map = json.load(infile)
    except:
        date_map = {}
    failed_dates = []
    for page in range(*page_range):
        try:
            url = (
                f"https://links.sgx.com/1.0.0/derivatives-daily/{page}/FUTURE.zip"
            )
            r = requests.get(url)
            with zipfile.ZipFile(BytesIO(r.content)) as file:
                with file.open(file.namelist()[0]) as my_file:
                    data = my_file.read().decode()
                    if file.namelist()[0].endswith("txt"):
                        data_df = pd.read_table(StringIO(data))
                    else:
                        data_df = pd.read_csv(StringIO(data))
            data_df.columns = [col.lower() for col in data_df.columns]
            run_date = data_df['date'][0]
            date_map[str(run_date)] = page
            print(f'page={page}={run_date}')
            if 'series' not in data_df.columns:
                data_df['com'] = data_df['com'].str.strip()
                data_df = data_df[data_df['com_yy'].astype(str).str.isnumeric()]
                data_df['series'] = data_df.apply(lambda x: "%s%s%s" % (
                    x['com'].strip(), rev_month_code_map[int(x['com_mm'])].upper(), str(x['com_yy'])[-2:]), axis=1)
            data_df = data_df.rename(columns={'com': 'product', 'com_mm': 'mth', 'com_yy': 'year',
                                              'oint': 'openInterest', 'series': 'instID'})
            data_df['date'] = data_df['date'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y%m%d"))
            data_df['product'] = data_df['product'].str.strip()
            data_df = data_df[data_df['product'].isin(SGX_future_products)]
            if str(run_date) <= "20180118":
                for col in ['open', 'high', 'low', 'close', 'settle']:
                    data_df[col] = data_df[col]/100
            data_df['instID'] = data_df.apply(lambda x: f"{x['product']}{x['year'] % 100:02}{x['mth']:02}", axis=1)
            data_df['exch'] = 'SGX'
            data_df = data_df[FUTURE_COLUMNS]
            save_data('fut_daily', data_df, flavor='mysql')
        except Exception as e:
            print(f"error on {page}: exception={e}")
            failed_dates.append(page)
            continue
    with open(save_json_file, 'w') as ofile:
        json.dump(date_map, ofile, indent=4)
    res = {'failed_dates': failed_dates, 'date_map': date_map}
    return res


def fetch_daily_eod(update_win=10):
    last_page = 6775
    try:
        with open(save_json_file, 'r') as infile:
            date_map = json.load(infile)
        last_date = max(date_map.keys())
        last_page = date_map[last_date]
        last_date = datetime.datetime.strptime(last_date, '%Y%m%d').date()
        tday = datetime.date.today()
        update_win = workdays.networkdays(last_date, tday) + 3
    except:
        pass
    res = futures_sgx_daily(page_range=[last_page+1, last_page+update_win])
    return res


def fetch_fef_3pm_close(cdate=datetime.date.today()):
    try:
        with open(save_json_file, 'r') as infile:
            date_map = json.load(infile)
        cdate_key = cdate.strftime("%Y%m%d")
        if cdate_key in date_map:
            return True
    except:
        pass
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
    }
    try:
        url = (
                f"https://api2.sgx.com/sites/default/files/reports/prices-reports/{cdate.year}/{cdate.month:02}" +
                f"/wcm%40sgx_en%40iron_fef%40{cdate.strftime('%d-%b-%Y')}%40Iron_Ore_FEF.csv"
        )
        r = requests.get(url, headers=headers)
        tmp_df = pd.read_csv(BytesIO(r.content))
        tmp_df.columns = ["cont", "close"]
        tmp_df['cont'] = tmp_df['cont'].apply(lambda s: s.split("_")[0])

        tmp_df['instID'] = tmp_df['cont'].apply(
            lambda s: s[:-3] + str(int(s[-2:]) * 100 + month_code_map[s[-3].lower()]))
        tmp_df['date'] = cdate
        tmp_df['exch'] = 'SGX'
        tmp_df['settle'] = tmp_df['close']
        data_df = tmp_df[['instID', 'exch', 'date', 'close', 'settle']]
        save_data('fut_daily', data_df, flavor='mysql')
        return True
    except:
        return False


if __name__ == "__main__":
    fetch_fef_3pm_close(cdate=datetime.date.today())
