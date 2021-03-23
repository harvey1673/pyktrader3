import sys
import datetime
from pycmqlib3.utility.misc import day_shift, CHN_Holidays
from aks_data_update import update_hist_fut_daily, update_spot_daily
from factor_data_update import run_update

if __name__=="__main__":
    args = sys.argv[1:]
    if len(args)>=1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        tday = datetime.date.today()
    sdate = day_shift(tday, '-2b', CHN_Holidays)
    print('updating historical future data...')
    update_hist_fut_daily(sdate, tday, exchanges = ['DCE', 'INE', 'SHFE', 'CZCE', 'CFFEX',], flavor = 'mysql', fut_table = 'fut_daily')
    print('updating factor data calculation...')
    run_update(tday, '-3y')
    print('updating historical spot data ...')
    update_spot_daily(sdate, tday)
