import sys
import logging
import datetime
import json
from misc_scripts.factor_data_update import update_port_position
from misc_scripts.fun_factor_update import update_fun_factor
from misc_scripts.auto_update_data_xl import update_data_from_xl
from pycmqlib3.utility.sec_bits import EMAIL_HOTMAIL, NOTIFIERS, LOCAL_PC_NAME, EMAIL_NOTIFY
from pycmqlib3.utility.email_tool import send_html_by_smtp
update_func_list = [
    'fun_data_xl_loading',
    'fun_factor_update',
    'fact_pos_file',
]


def update_port_pos(tday=datetime.date.today(), email_notify=EMAIL_NOTIFY):
    job_status = {}
    logging.info('updating factor strategy position...')
    pos_update = {}
    for update_field in update_func_list:
        if update_field == 'fun_data_xl_loading':
            update_data_from_xl()
        elif update_field == 'fun_factor_update':
            update_fun_factor(run_date=tday)
        else:
            pos_update = update_port_position(run_date=tday)
        print(f"{update_field} is done")
        job_status[update_field] = True
        # except:
        #     job_status[update_field] = False

    if email_notify:
        sub = '%s port pos update<%s>' % (LOCAL_PC_NAME, tday.strftime('%Y.%m.%d'))
        html = "<html><head></head><body><p><br>"
        for key in pos_update:
            html += "Position change for %s:<br>%s" % (key, pos_update[key].to_html())
        html += "</p></body></html>"
        send_html_by_smtp(EMAIL_HOTMAIL, NOTIFIERS, sub, html)
    job_status['time'] = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    filename = "C:\\dev\\data\\port_position_update.json"
    with open(filename, 'w') as ofile:
        json.dump(job_status, ofile, indent=4)
    return job_status, pos_update


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        tday = datetime.date.today()
    job_status, pos_update = update_port_pos(tday=datetime.date.today(), email_notify=True)
    print(job_status, pos_update)
