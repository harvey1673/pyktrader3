import sys
import datetime
import json
from pycmqlib3.utility.misc import day_shift, CHN_Holidays, sign, is_workday, inst2product
import pycmqlib3.analytics.data_handler as dh
from aks_data_update import update_hist_fut_daily, update_spot_daily, \
                            update_exch_receipt_table, update_exch_inv_table, update_rank_table
from factor_data_update import update_factor_data, generate_daily_position

scenarios_mixed = [('tscarry', 'ryield', 3.0, 1, 1, 5, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tscarry', 'basmom', 1.0, 60, 1, 10, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tscarry', 'basmom', 1.0, 100, 1, 10, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tscarry', 'basmom', 1.0, 240, 1, 10, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('xscarry', 'ryieldsma', 1.0, 1, 1, 5, (None, {}, ''), [0.0, 0.0]), \
                 ('xscarry', 'ryieldsma', 1.0, 1, 50, 5, (None, {}, ''), [0.0, 0.0]), \
                 ('xscarry', 'basmom', 1.0, 110, 1, 5, (None, {}, ''), [0.0, 0.0]), \
                 ('xscarry', 'basmom', 1.0, 140, 1, 5, (None, {}, ''), [0.0, 0.0]), \
                 ('xscarry', 'basmomsma', 1.0, 90, 20, 5, (None, {}, ''), [0.0, 0.0]), \
                 ('xscarry', 'basmomsma', 1.0, 230, 20, 5, (None, {}, ''), [0.0, 0.0]), \
                 ('tsmom', 'momxma', 1.0, 20, 50, 5, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'momxma', 1.0, 30, 120, 5, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'momxma', 1.0, 40, 30, 5, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'mixmom', 1.0, 10, 1, 10, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'mixmom', 1.0, 20, 1, 10, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'rsixea', 1.0, 20, 30, 5, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'rsixea', 1.0, 40, 30, 5, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'rsixea', 1.0, 60, 30, 5, (sign, {}, 'sign'), [0.0, 0.0]), \
                 ('tsmom', 'macdnma', 1.0, 8, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 10.0]), \
                 ('tsmom', 'macdnma', 1.0, 16, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 5.0]), \
                 ('tsmom', 'macdnma', 1.0, 24, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 3.34]), \
                 ('xsmom', 'mom', 1.0, 130, 1, 5, (None, {}, ''), [0.0]), \
                 ('xsmom', 'mom', 1.0, 230, 1, 5, (None, {}, ''), [0.0]), \
                 ('xsmom', 'rsiema', 1.0, 60, 80, 5, (None, {}, ''), [0.0]), \
                 ('xsmom', 'rsiema', 1.0, 10, 80, 5, (None, {}, ''), [0.0]), \
                 ('xsmom', 'rsiema', 1.0, 40, 20, 5, (None, {}, ''), [0.0]), \
                 ('xsmom', 'macdnma', 1.0, 16, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 6.25], 0.2), \
                 ('xsmom', 'macdnma', 1.0, 40, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 2.5], 0.2), \
                 ('xsmom', 'macdnma', 1.0, 56, 280, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 2.5], 0.2), ] 

mixed_metal_mkts = ['rb', 'hc', 'i', 'j', 'jm', 'ru', 'FG', 'cu', 'al', 'zn', 'ni']

commod_mkts = ['rb', 'hc', 'i', 'j', 'jm', 'ru', 'FG', 'cu', 'al', 'zn', 'pb', 'ni', 'sn', \
               'l', 'pp', 'v', 'TA', 'sc', 'm', 'RM', 'y', 'p', 'OI', 'a', 'c', 'CF', 'jd', \
               'AP', 'SM', 'SF', 'ss', 'CJ', 'UR', 'eb', 'eg', 'pg', 'T', 'PK', 'PF', 'lh', \
               'MA', 'SR', 'cs', 'TF', 'lu', 'fu']

port_pos_config = [
    ('PT_FACTPORT2', 'C:/dev/pyktrader3/process/pt_test2/', 4600, 'CAL_30b', 's1'),
    ('PT_FACTPORT3', 'C:/dev/pyktrader3/process/pt_test3/', 4600, 'CAL_30b', 's1'),
]

scenarios_all = [
             ('tscarry', 'ryieldnmb', 2.8, 1, 120, 1, (None, {}, ''), [0.0, 0.0]),
             ('tscarry', 'basmomnma', 0.7, 100, 120, 1, (None, {}, ''), [0.0, 0.0]),
             ('tscarry', 'basmomnma', 0.5, 170, 120, 1, (None, {}, ''), [0.0, 0.0]),
             #('tscarry', 'basmomnma', 0.2, 230, 120, 1, (None, {}, ''), [0.0, 0.0]),
             ('xscarry', 'ryieldsma', 0.6, 1, 30, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             #('xscarry', 'ryieldsma', 0.15, 1, 110, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'ryieldsma', 1.5, 1, 190, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'ryieldnma',1.5, 1, 20, 1, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'ryieldnma', 1.8, 1, 110, 1, (None, {}, ''), [0.0, 0.0], 0.2),
             #('xscarry', 'ryieldnma', 0.2, 1, 210, 1, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'basmomsma', 0.6, 100, 10, 5, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'basmomsma', 0.6, 220, 10, 5, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'basmomnma', 1.5, 80, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'basmomnma', 1.5, 150, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xscarry', 'basmomnma', 1.5, 220, 120, 5, (None, {}, ''), [0.0, 0.0], 0.2),
             ('tsmom', 'momnma', 0.2, 10, 60, 1, (None, {}, ''), [0.0]),
             ('tsmom', 'momnma', 0.07, 220, 60, 1, (None, {}, ''), [0.0]),
             ('tsmom', 'hlbrk', 2.0, 10, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             ('tsmom', 'hlbrk', 1.5, 30, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             ('tsmom', 'hlbrk', 1.2, 240, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             #('tsmom', 'momxma', 0.2, 40, 30, 5, (misc.sign, {}, 'sign'), [0.0]),
             #('tsmom', 'momxma', 0.15, 40, 80, 5, (misc.sign, {}, 'sign'), [0.0]),
             #('tsmom', 'mixmom', 0.375, 10, 1, 10, (misc.sign, {}, 'sign'), [0.0]),
             #('tsmom', 'mixmom', 0.3, 30, 1, 10, (misc.sign, {}, 'sign'), [0.0]),
             #('tsmom', 'mixmom', 0.3, 220, 1, 10, (misc.sign, {}, 'sign'), [0.0]),
             #('tsmom', 'rsixea', 0.25, 30, 40, 5, (misc.sign, {}, 'sign'), [0.0]),
             #('tsmom', 'rsixea', 0.25, 30, 110, 5, (misc.sign, {}, 'sign'), [0.0]),
             ('tsmom', 'macdnma', 0.4, 8, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 10.0]),
             ('tsmom', 'macdnma', 0.3, 16, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 5.0]),
             ('tsmom', 'macdnma', 0.3, 24, 160, 5, (dh.response_curve, {"response": "reverting", "param": 2}, 'reverting'), [1.5, 3.34]),
             #('xsmom', 'mom', 0.15, 160, 1, 5, (None, {}, ''), [0.0], 0.2),
             ('xsmom', 'hlbrk', 1.5, 20, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xsmom', 'hlbrk', 1.2, 120, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xsmom', 'hlbrk', 1.2, 240, 1, 10, (None, {}, ''), [0.0, 0.0], 0.2),
             ('xsmom', 'mom', 1.0, 20, 1, 5, (None, {}, ''), [0.0], 0.2),
             ('xsmom', 'mom', 1.0, 210, 1, 5, (None, {}, ''), [0.0], 0.2),
             ('xsmom', 'momnma', 1.0, 130, 90, 5, (None, {}, ''), [0.0], 0.2),
             ('xsmom', 'momnma', 1.0, 240, 90, 5, (None, {}, ''), [0.0], 0.2),
             ('xsmom', 'momsma', 0.8, 140, 120, 5, (None, {}, ''), [0.0], 0.2),
             ('xsmom', 'momsma', 0.8, 240, 120, 5, (None, {}, ''), [0.0], 0.2),
             #('xsmom', 'rsiema', 0.1, 70, 60, 5, (None, {}, ''), [0.0], 0.2),
             #('xsmom', 'rsiema', 0.1, 100, 80, 5, (None, {}, ''), [0.0], 0.2),
             #('xsmom', 'rsiema', 0.1, 90, 10, 5, (None, {}, ''), [0.0], 0.2),
             #('xsmom', 'macdnma', 0.1, 8, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 12.5], 0.2),
             #('xsmom', 'macdnma', 0.1, 16, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 6.25], 0.2),
             #('xsmom', 'macdnma', 0.1, 32, 200, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 3.125], 0.2),
             #('xsmom', 'macdnma', 0.1, 64, 100, 5, (dh.response_curve, {"response": "absorbing", "param": 2}, "absorbing"), [1.5, 1.56], 0.2),
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
    for (fact_key, fact_mkts, scenarios) in [('commod_all', commod_mkts, scenarios_all), ]:
        try:
            if not job_status[update_field].get(fact_key, False):
                _ = update_factor_data(fact_mkts, scenarios, start_date, edate, roll_rule='30b')
                job_status[update_field][fact_key] = True
        except:
            job_status[update_field][fact_key] = False
            print("fact_key = %s is FAILED to update" % (fact_key))
        save_status(filename, job_status)
    print('updating factor strategy position...')
    update_field = 'fact_pos_file'
    if update_field not in job_status:
        job_status[update_field] = {}
    for (port_name, pos_loc, pos_scaler, roll, freq) in port_pos_config:
        if job_status[update_field].get(port_name, False):
            continue
        config_file = f'{pos_loc}settings/{port_name}.json'
        with open(config_file, 'r') as fp:
            strat_conf = json.load(fp)
        strat_args = strat_conf['config']
        assets = strat_args['assets']
        factor_repo = strat_args['factor_repo']
        product_list = []
        for asset_dict in assets:
            under = asset_dict["underliers"][0]
            product = inst2product(under)
            product_list.append(product)
        try:
            target_pos, pos_sum = generate_daily_position(edate, product_list, factor_repo,
                                                            roll_label=roll,
                                                            pos_scaler=pos_scaler,
                                                            freq=freq,
                                                            hist_fact_lookback=20)
            pos_date = day_shift(edate, '1b', CHN_Holidays).strftime('%Y%m%d')
            posfile = '%s%s_%s.json' % (pos_loc, port_name, pos_date)
            with open(posfile, 'w') as ofile:
                json.dump(target_pos, ofile, indent=4)
            pos_sum.index.name = 'factor'
            pos_sum.to_csv('%spos_by_strat_%s_%s.csv' % (pos_loc, port_name, pos_date))
            job_status[update_field][port_name] = True
        except:
            job_status[update_field][port_name] = False
        save_status(filename, job_status)

    sdate = day_shift(tday, '-1b', CHN_Holidays)
    for (update_field, update_func, ref_text) in [('exch_receipt', update_exch_receipt_table, 'exch receipt'), \
                                                ('exch_inv', update_exch_inv_table, 'exchange warrant'), \
                                                ('spot_daily', update_spot_daily, 'spot data'), \
                                                ('rank_table', update_rank_table, 'top future broker ranking table'),]:
        try:
            print('updating historical %s ...' % ref_text)
            if not job_status.get(update_field, False):
                update_func(sdate, tday, flavor = 'mysql')
                job_status[update_field] = True
        except:
            job_status[update_field] = False
            print("update_field = %s is FAILED to update" % (update_field))
        save_status(filename, job_status)


if __name__=="__main__":    
    args = sys.argv[1:]
    if len(args)>=1:
        tday = datetime.datetime.strptime(args[0], "%Y%m%d").date()
    else:
        tday = datetime.date.today()    
    run_update(tday)
    
    

