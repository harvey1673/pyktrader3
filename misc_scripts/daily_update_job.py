import sys
import datetime
import pandas as pd
import json
from pycmqlib3.utility.sec_bits import EMAIL_HOTMAIL, EMAIL_NOTIFY, NOTIFIERS
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, sign, is_workday, inst2product
import pycmqlib3.analytics.data_handler as dh
from misc_scripts.aks_data_update import update_hist_fut_daily, update_spot_daily, \
    update_exch_receipt_table, update_exch_inv_table, update_rank_table
from misc_scripts.factor_data_update import update_factor_data, generate_strat_position
from pycmqlib3.utility.email_tool import send_html_by_smtp

scenarios_elite = [
        ('tscarry', 'ryieldnmb', 2.0, 1, 120, 1, (None, {}, ''), [0.0, 0.0]),
        ('tscarry', 'basmomnma', 1.0, 100, 120, 1, (None, {}, ''), [0.0, 0.0]),
        ('tscarry', 'basmomnma', 1.0, 170, 120, 1, (None, {}, ''), [0.0, 0.0]),
        #('xscarry', 'ryieldsma', 0.6, 1, 30, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        #('xscarry', 'ryieldsma', 1.5, 1, 190, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        ('xscarry-rank_cutoff', 'ryieldnma',1.9, 1, 20, 1, (None, {}, ''), [0.0, 0.0], 0.2),
        ('xscarry-rank_cutoff', 'ryieldnma', 2.1, 1, 110, 1, (None, {}, ''), [0.0, 0.0], 0.2),
        #'xscarry', 'basmomsma', 0.6, 100, 10, 5, (None, {}, ''), [0.0, 0.0], 0.2),
        #'xscarry', 'basmomsma', 0.6, 220, 10, 5, (None, {}, ''), [0.0, 0.0], 0.2),
        ('xscarry-rank_cutoff', 'basmomnma', 2.5, 80, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
        ('xscarry-rank_cutoff', 'basmomnma', 2.5, 150, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
        #('xscarry', 'basmomnma', 1.5, 220, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
        ('tsmom', 'momnma', 0.2, 10, 60, 1, (None, {}, ''), [0.0]),
        ('tsmom', 'momnma', 0.07, 220, 60, 1, (None, {}, ''), [0.0]),
        ('tsmom', 'hlbrk', 1.1, 10, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        ('tsmom', 'hlbrk', 0.9, 30, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        ('tsmom', 'hlbrk', 0.9, 240, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        ('tsmom', 'macdnma', 0.22, 8, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 10.0]),
        ('tsmom', 'macdnma', 0.20, 16, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 5.0]),
        #('tsmom', 'macdnma', 0.3, 24, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 3.34]),
        #('xsmom', 'mom', 0.15, 160, 1, 5, (None, {}, ''), [0.0], 0.2),
        ('xsmom-rank_cutoff', 'hlbrk', 0.5, 20, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        ('xsmom-rank_cutoff', 'hlbrk', 0.75, 120, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        ('xsmom-rank_cutoff', 'hlbrk', 0.75, 240, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
        #('xsmom', 'mom', 1.0, 20, 1, 5, (None, {}, ''), [0.0], 0.2),
        #('xsmom', 'mom', 1.0, 210, 1, 5, (None, {}, ''), [0.0], 0.2),
        ('xsmom-rank_cutoff', 'momnma', 0.75, 130, 90, 5, (None, {}, ''), [0.0], 0.2),
        ('xsmom', 'momnma', 0.75, 240, 90, 5, (None, {}, ''), [0.0], 0.2),
        #('xsmom', 'momsma', 0.8, 140, 120, 5, (None, {}, ''), [0.0], 0.2),
        #('xsmom', 'momsma', 0.8, 240, 120, 5, (None, {}, ''), [0.0], 0.2),
]

scenarios_test = [
    ('tscarry', 'ryieldnmb', 1.0, 1, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'ryieldqtl', 0.8, 1, 20, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'ryieldqtl', 0.8, 1, 60, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'ryieldqtl', 0.8, 1, 244, 1, (None, {}, ''), [0.0, 0.0]),

    ('tscarry', 'basmomnma', 0.5, 20, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomnma', 0.42, 60, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomnma', 0.35, 120, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomnma', 0.35, 180, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomqtl', 2.0, 120, 20, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomqtl', 1.8, 240, 20, 1, (None, {}, ''), [0.0, 0.0]),

    ('xscarry-rank', 'ryieldnma', 1.4, 1, 20, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'ryieldnma', 1.4, 1, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'ryieldnma', 1.4, 1, 244, 1, (None, {}, ''), [0.0, 0.0], 0.2),

    ('xscarry-rank', 'basmomnma', 2.0, 20, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'basmomnma', 2.0, 100, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'basmomnma', 2.0, 170, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),

    ('tsmom', 'hlbrk', 0.5, 20, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 40, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 61, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 122, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 244, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'macdnma', 0.22, 8, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 10.0]),
    ('tsmom', 'macdnma', 0.20, 16, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 5.0]),
    ('tsmom', 'macdnma', 0.18, 24, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 3.34]),
    ('xsmom-rank', 'hlbrk', 0.375, 20, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 40, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 61, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 122, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 244, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'momnma', 0.6, 10, 10, 1, (None, {}, ''), [0.0], 0.2),
    ('xsmom-rank', 'momnma', 0.6, 130, 120, 1, (None, {}, ''), [0.0], 0.2),
    ('xsmom-rank', 'momnma', 0.6, 240, 60, 1, (None, {}, ''), [0.0], 0.2),
]

mixed_metal_mkts = ['rb', 'hc', 'i', 'j', 'jm', 'ru', 'FG', 'cu', 'al', 'zn', 'ni']

commod_mkts = ['rb', 'hc', 'i', 'j', 'jm', 'ru', 'FG', 'cu', 'al', 'zn', 'pb', 'ni', 'sn', \
               'l', 'pp', 'v', 'TA', 'sc', 'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd', \
               'AP', 'SM', 'SF', 'ss', 'CJ', 'UR', 'eb', 'eg', 'pg', 'T', 'PK', 'PF', 'lh', \
               'MA', 'SR', 'cs', 'TF', 'lu', 'fu']


port_pos_config = [
    ('PT_FACTPORT3', 'C:/dev/pyktrader3/process/pt_test3/', 4600, 'CAL_30b', 's1'),
    ('PT_FACTPORT1', 'C:/dev/pyktrader3/process/pt_test3/', 4600, 'CAL_30b', 's1'),
    ('PT_FACTPORT3', 'C:/dev/pyktrader3/process/pt_test3/', 4600, 'hot', 'd1'),
    ('PT_FACTPORT1', 'C:/dev/pyktrader3/process/pt_test3/', 4600, 'hot', 'd1'),
    ('PT_FACTPORT3', 'C:/dev/pyktrader3/process/pt_test3/', 4600, 'expiry', 'd1'),
    ('PT_FACTPORT1', 'C:/dev/pyktrader3/process/pt_test3/', 4600, 'expiry', 'd1'),
]

pos_chg_notification = ['PT_FACTPORT3_CAL_30b', 'PT_FACTPORT1_hot']

scenarios_all = [
    ('tscarry', 'ryieldnmb', 2.8, 1, 120, 1, (None, {}, ''), [0.0, 0.0]),

    ('tscarry', 'ryieldnmb', 1.0, 1, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'ryieldqtl', 0.8, 1, 20, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'ryieldqtl', 0.8, 1, 60, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'ryieldqtl', 0.8, 1, 244, 1, (None, {}, ''), [0.0, 0.0]),

    ('tscarry', 'basmomnma', 0.7, 100, 120, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomnma', 0.5, 170, 120, 1, (None, {}, ''), [0.0, 0.0]),
    # ('tscarry', 'basmomnma', 0.2, 230, 120, 1, (None, {}, ''), [0.0, 0.0]),

    ('tscarry', 'basmomnma', 0.5, 20, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomnma', 0.42, 60, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomnma', 0.35, 120, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomnma', 0.35, 180, 122, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomqtl', 2.0, 120, 20, 1, (None, {}, ''), [0.0, 0.0]),
    ('tscarry', 'basmomqtl', 1.8, 240, 20, 1, (None, {}, ''), [0.0, 0.0]),

    ('xscarry', 'ryieldsma', 0.6, 1, 30, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    # ('xscarry', 'ryieldsma', 0.15, 1, 110, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry', 'ryieldsma', 1.5, 1, 190, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry', 'ryieldnma', 1.5, 1, 20, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry', 'ryieldnma', 1.8, 1, 110, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    # ('xscarry', 'ryieldnma', 0.2, 1, 210, 1, (None, {}, ''), [0.0, 0.0], 0.2),

    ('xscarry-rank', 'ryieldnma', 1.4, 1, 20, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'ryieldnma', 1.4, 1, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'ryieldnma', 1.4, 1, 244, 1, (None, {}, ''), [0.0, 0.0], 0.2),

    ('xscarry', 'basmomsma', 0.6, 100, 10, 5, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry', 'basmomsma', 0.6, 220, 10, 5, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry', 'basmomnma', 1.5, 80, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry', 'basmomnma', 1.5, 150, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry', 'basmomnma', 1.5, 220, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),

    ('xscarry-rank', 'basmomnma', 2.0, 20, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'basmomnma', 2.0, 100, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xscarry-rank', 'basmomnma', 2.0, 170, 122, 1, (None, {}, ''), [0.0, 0.0], 0.2),

    ('tsmom', 'hlbrk', 2.0, 10, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 1.5, 30, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 1.2, 240, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'momnma', 0.2, 10, 60, 1, (None, {}, ''), [0.0]),
    ('tsmom', 'momnma', 0.07, 220, 60, 1, (None, {}, ''), [0.0]),
    # ('tsmom', 'momxma', 0.2, 40, 30, 5, (misc.sign, {}, 'sign'), [0.0]),
    # ('tsmom', 'momxma', 0.15, 40, 80, 5, (misc.sign, {}, 'sign'), [0.0]),
    # ('tsmom', 'mixmom', 0.375, 10, 1, 10, (misc.sign, {}, 'sign'), [0.0]),
    # ('tsmom', 'mixmom', 0.3, 30, 1, 10, (misc.sign, {}, 'sign'), [0.0]),
    # ('tsmom', 'mixmom', 0.3, 220, 1, 10, (misc.sign, {}, 'sign'), [0.0]),
    # ('tsmom', 'rsixea', 0.25, 30, 40, 5, (misc.sign, {}, 'sign'), [0.0]),
    # ('tsmom', 'rsixea', 0.25, 30, 110, 5, (misc.sign, {}, 'sign'), [0.0]),
    # ('xsmom', 'mom', 0.15, 160, 1, 5, (None, {}, ''), [0.0], 0.2),

    ('tsmom', 'hlbrk', 0.5, 20, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 40, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 61, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 122, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'hlbrk', 0.5, 244, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('tsmom', 'macdnma', 0.22, 8, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'),
     [1.5, 10.0]),
    ('tsmom', 'macdnma', 0.20, 16, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'),
     [1.5, 5.0]),
    ('tsmom', 'macdnma', 0.18, 24, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'),
     [1.5, 3.34]),

    ('xsmom', 'hlbrk', 1.2, 120, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom', 'hlbrk', 1.2, 240, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom', 'mom', 1.0, 20, 1, 5, (None, {}, ''), [0.0], 0.2),
    ('xsmom', 'mom', 1.0, 210, 1, 5, (None, {}, ''), [0.0], 0.2),
    ('xsmom', 'momnma', 1.0, 130, 90, 5, (None, {}, ''), [0.0], 0.2),
    ('xsmom', 'momnma', 1.0, 240, 90, 5, (None, {}, ''), [0.0], 0.2),
    ('xsmom', 'momsma', 0.8, 140, 120, 5, (None, {}, ''), [0.0], 0.2),
    ('xsmom', 'momsma', 0.8, 240, 120, 5, (None, {}, ''), [0.0], 0.2),
    # ('xsmom', 'rsiema', 0.1, 70, 60, 5, (None, {}, ''), [0.0], 0.2),
    # ('xsmom', 'rsiema', 0.1, 100, 80, 5, (None, {}, ''), [0.0], 0.2),
    # ('xsmom', 'rsiema', 0.1, 90, 10, 5, (None, {}, ''), [0.0], 0.2),
    # ('xsmom', 'macdnma', 0.1, 8, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 12.5], 0.2),
    # ('xsmom', 'macdnma', 0.1, 16, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 6.25], 0.2),
    # ('xsmom', 'macdnma', 0.1, 32, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 3.125], 0.2),
    # ('xsmom', 'macdnma', 0.1, 64, 100, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 1.56], 0.2),

    ('xsmom-rank', 'hlbrk', 0.375, 20, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 40, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 61, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 122, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'hlbrk', 0.375, 244, 1, 1, (None, {}, ''), [0.0, 0.0], 0.2),
    ('xsmom-rank', 'momnma', 0.6, 10, 10, 1, (None, {}, ''), [0.0], 0.2),
    ('xsmom-rank', 'momnma', 0.6, 130, 120, 1, (None, {}, ''), [0.0], 0.2),
    ('xsmom-rank', 'momnma', 0.6, 240, 60, 1, (None, {}, ''), [0.0], 0.2),
]


run_settings = [
    ('commod_cal', commod_mkts, scenarios_all, 'CAL_30b', 's1'),
    ('commod_hot', commod_mkts, scenarios_all, 'hot', 'd1'),
    ('commod_exp', commod_mkts, scenarios_all, 'expiry', 'd1'),
]


def save_status(filename, job_status):
    with open(filename, 'w') as ofile:
        json.dump(job_status, ofile, indent=4)


def run_update(tday=datetime.date.today()):
    edate = min(datetime.date.today(), tday)     
    if not is_workday(edate, 'CHN'):
        edate = day_shift(edate, '-1b', CHN_Holidays)
    sdate = day_shift(edate, '-3b', CHN_Holidays)
    filename = "C:\\dev\\data\\dailyjob_status_%s.json" % edate.strftime("%Y%m%d")
    try:
        with open(filename, 'r') as f:
            job_status = json.load(f)
    except:
        job_status = {}

    print('updating historical future data...')
    update_field = 'fut_daily'
    if update_field not in job_status:
        job_status[update_field] = {}
    for exch in ["DCE", "CFFEX", "CZCE", "SHFE", "INE", "GFEX"]:
        try:
            if not job_status[update_field].get(exch, False):
                missing = update_hist_fut_daily(sdate, edate, exchanges = [exch], flavor = 'mysql', fut_table = 'fut_daily')
                if len(missing) == 0:
                    job_status[update_field][exch] = True
                    print(f'{exch} is updated')
                else:
                    job_status[update_field][exch] = False
                    print(f'{exch} has some issue {missing}')                
        except:
            job_status[update_field][exch] = False
            print("exch = %s EOD price is FAILED to update" % (exch))    
        save_status(filename, job_status)
    #update_hist_fut_daily(sdate, tday, exchanges = ["DCE", "CFFEX", "CZCE", "SHFE", "INE", ], flavor = 'mysql', fut_table = 'fut_daily')
    print('updating factor data calculation...')
    start_date = day_shift(edate, '-30m')
    update_field = 'fact_repo'
    if update_field not in job_status:
        job_status[update_field] = {}
    for (fact_key, fact_mkts, scenarios, roll_label, freq) in run_settings:
        try:
            if not job_status[update_field].get(fact_key, False):
                _ = update_factor_data(fact_mkts, scenarios, start_date, edate, roll_rule=roll_label, freq=freq)
                job_status[update_field][fact_key] = True
        except:
            job_status[update_field][fact_key] = False
            print("fact_key = %s is FAILED to update" % (fact_key))
        save_status(filename, job_status)
    print('updating factor strategy position...')
    update_field = 'fact_pos_file'
    if update_field not in job_status:
        job_status[update_field] = {}
    pos_update = {}
    target_pos = {}
    pos_by_strat = {}
    for port_name in port_pos_config.keys():
        pos_loc = port_pos_config[port_name]['pos_loc']
        roll = port_pos_config[port_name]['roll']
        port_file = port_name + '_' + roll
        if job_status[update_field].get(port_file, False):
            continue
        try:
            for strat_file, pos_scaler, freq in port_pos_config[port_name]['strat_list']:
                config_file = f'{pos_loc}/settings/{strat_file}'
                with open(config_file, 'r') as fp:
                    strat_conf = json.load(fp)
                strat_args = strat_conf['config']
                assets = strat_args['assets']
                repo_type = strat_args['repo_type']
                factor_repo = strat_args['factor_repo']

                product_list = []
                for asset_dict in assets:
                    under = asset_dict["underliers"][0]
                    product = inst2product(under)
                    product_list.append(product)

                strat_target, strat_sum = generate_strat_position(edate, product_list, factor_repo,
                                                                  repo_type=repo_type,
                                                                  roll_label=roll,
                                                                  pos_scaler=pos_scaler,
                                                                  freq=freq,
                                                                  hist_fact_lookback=20)
                pos_by_strat[strat_file] = strat_target

                for prod in strat_target:
                    if prod not in target_pos:
                        target_pos[prod] = 0
                    target_pos[prod] += strat_target[prod]

            for prodcode in target_pos:
                if prodcode == 'CJ':
                    target_pos[prodcode] = int((target_pos[prodcode] / 4 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 4
                elif prodcode == 'ZC':
                    target_pos[prodcode] = int((target_pos[prodcode] / 2 + (0.5 if target_pos[prodcode] > 0 else -0.5))) * 2
                else:
                    target_pos[prodcode] = int(target_pos[prodcode] + (0.5 if target_pos[prodcode] > 0 else -0.5))

            pos_date = day_shift(edate, '1b', CHN_Holidays)
            pre_date = day_shift(pos_date, '-1b', CHN_Holidays)
            pos_date = pos_date.strftime('%Y%m%d')
            pre_date = pre_date.strftime('%Y%m%d')
            posfile = '%s/%s_%s.json' % (pos_loc, port_file, pos_date)
            with open(posfile, 'w') as ofile:
                json.dump(target_pos, ofile, indent=4)

            stratfile = '%s/pos_by_strat_%s_%s.json' % (pos_loc, port_file, pos_date)
            with open(stratfile, 'w') as ofile:
                json.dump(pos_by_strat, ofile, indent=4)

            job_status[update_field][port_file] = True

            if port_file in pos_chg_notification:
                with open('%s/%s_%s.json' % (pos_loc, port_file, pre_date), 'r') as fp:
                    curr_pos = json.load(fp)
                pos_df = pd.DataFrame({'cur': curr_pos, 'tgt': target_pos})
                pos_df['diff'] = pos_df['tgt'] - pos_df['cur']
                pos_update[port_file] = pos_df

        except:
            job_status[update_field][port_file] = False
        save_status(filename, job_status)

    sdate = day_shift(tday, '-1b', CHN_Holidays)
    for (update_field, update_func, ref_text) in [('exch_receipt', update_exch_receipt_table, 'exch receipt'),
                                                  ('exch_inv', update_exch_inv_table, 'exchange warrant'),
                                                  ('spot_daily', update_spot_daily, 'spot data'),
                                                  ('rank_table', update_rank_table, 'top future broker ranking table')]:
        try:
            print('updating historical %s ...' % ref_text)
            if not job_status.get(update_field, False):
                update_func(sdate, tday, flavor='mysql')
                job_status[update_field] = True
        except:
            job_status[update_field] = False
            print("update_field = %s is FAILED to update" % (update_field))
        save_status(filename, job_status)

    update_field = 'email_notify'
    if not job_status.get(update_field, False) and EMAIL_NOTIFY:
        sub = 'EOD pos and job status<%s>' % (edate.strftime('%Y.%m.%d'))
        html = "<html><head></head><body><p><br>"
        for key in pos_update:
            html += "Position change for %s:<br>%s" % (key, pos_update[key].to_html())
        html += "Job status: %s <br>" % (json.dumps(job_status))
        html += "</p></body></html>"
        send_html_by_smtp(EMAIL_HOTMAIL, NOTIFIERS, sub, html)
        job_status[update_field] = True
    save_status(filename, job_status)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        tday = datetime.date.today()    
    run_update(tday)
    
    

