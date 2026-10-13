import sys
import argparse
import logging
import datetime
import json

# This script is also launched directly by the daily batch job.  Make the two
# local source roots available before importing modules that depend on WTPY.
for local_source in ("C:/dev/pyktrader3", "C:/dev/wtpy"):
    if local_source not in sys.path:
        sys.path.append(local_source)

from misc_scripts.factor_data_update import update_port_position
from misc_scripts.fun_factor_update import update_db_factor
from misc_scripts.auto_update_data_xl import update_data_from_xl
from misc_scripts.sgx_daily_update import fetch_daily_eod, fetch_fef_3pm_close
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, is_workday
from pycmqlib3.utility import base
from pycmqlib3.utility.sec_bits import EMAIL_NOTIFY

update_func_list = [
    'fetch_sgx_eod',
    'fun_data_xl_loading',
    'fun_factor_update',
    'fact_pos_file',
]


def update_port_pos(tday=datetime.date.today(), email_notify=EMAIL_NOTIFY):
    job_status = {}
    logging.info('updating factor strategy position...')
    pos_update = {}
    details = {}
    for update_field in update_func_list:
        if update_field == 'fun_data_xl_loading':
            update_data_from_xl(lookback=100)
        elif update_field == 'fun_factor_update':
            update_db_factor(run_date=tday)
        elif update_field == 'fetch_sgx_eod':
            fetch_daily_eod()
            fetch_fef_3pm_close(cdate=tday)
        else:
            res = update_port_position(run_date=tday, email_notify=email_notify)
            pos_update = res['pos_update']
            details = res['details']
        print(f"{update_field} is done")
        job_status[update_field] = True
        # except:
        #     job_status[update_field] = False

    job_status['time'] = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    filename = "C:\\dev\\data\\port_position_update.json"
    with open(filename, 'w') as ofile:
        json.dump(job_status, ofile, indent=4)
    return job_status, pos_update


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the portfolio position update.")
    parser.add_argument("run_date", nargs="?", help="Run date in YYYYMMDD format.")
    email_group = parser.add_mutually_exclusive_group()
    email_group.add_argument(
        "--email",
        dest="email_notify",
        action="store_true",
        help="Send the position comparison email.",
    )
    email_group.add_argument(
        "--no-email",
        dest="email_notify",
        action="store_false",
        help="Do not send the position comparison email.",
    )
    parser.set_defaults(email_notify=EMAIL_NOTIFY)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_cli_args()
    if args.run_date:
        tday = datetime.datetime.strptime(args.run_date, "%Y%m%d").date()
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
    logging.info("running for %s" % str(tday))
    job_status, pos_update = update_port_pos(
        tday=tday, email_notify=args.email_notify
    )
    print(job_status, pos_update)
