# -*- coding: utf-8 -*-
import datetime
import sys
import time
import logging
import json
from pycmqlib3.utility import misc, base, sec_bits
from pycmqlib3.utility import dbaccess
from pycmqlib3.core.agent_save import SaveAgent
from pycmqlib3.core.agent_email import EmailAgent
from pycmqlib3.core.agent import Agent
from pycmqlib3.core.gui_agent import MainApp, Gui

def get_run_date():
    tday = datetime.date.today()
    now = datetime.datetime.now()
    if misc.is_workday(tday, 'CHN') and now.hour < 18:
        run_date = tday
    else:
        run_date = misc.day_shift(tday, '1b', misc.CHN_Holidays)
    return run_date


def save(config_file, run_date=None):
    with open(config_file, 'r') as infile:
        config = json.load(infile)
    name = config.get('name', 'save_ctp')
    folder = config['folder']
    filter_flag = config.get('filter_flag', False)
    base.config_logging(folder + name + ".log", level=logging.DEBUG,
                   format = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                   to_console = True,
                   console_level = logging.INFO)
    if run_date:
        year = run_date//10000
        month = (run_date % 10000)//100
        day = run_date % 100
        scur_day = datetime.date(year, month, day)
    else:
        scur_day = get_run_date()
    dbaccess.update_contract_list_table(scur_day)
    print('scur_day = %s' % scur_day.strftime('%Y%m%d'))
    save_agent = SaveAgent(config = config, tday = scur_day)
    if 'instIDs' in config:
        curr_insts = config['instIDs']
    else:
        curr_insts = misc.filter_main_cont(scur_day.strftime('%Y%m%d'), filter_flag)
    for inst in curr_insts:
        save_agent.add_instrument(inst)
    try:
        save_agent.restart()
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        save_agent.exit()

def save_gui(config_file, run_date=None):
    with open(config_file, 'r') as infile:
        config = json.load(infile)
    name = config.get('name', 'save_ctp')
    folder = config['folder']
    filter_flag = config.get('filter_flag', False)
    base.config_logging(folder + name + ".log", level=logging.DEBUG,
                   format = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                   to_console = True,
                   console_level = logging.INFO)
    if run_date:
        year = run_date//10000
        month = (run_date % 10000)//100
        day = run_date % 100
        scur_day = datetime.date(year, month, day)
    else:
        scur_day = get_run_date()
    dbaccess.update_contract_list_table(scur_day)
    if 'instIDs' in config:
        curr_insts = config['instIDs']
    else:
        curr_insts = misc.filter_main_cont(scur_day, filter_flag)
    print("number of insts to save = %s" % len(curr_insts))
    myApp = MainApp(scur_day, config, master=None)
    for inst in curr_insts:
        myApp.agent.add_instrument(inst)
    myApp.restart()
    myGui = Gui(myApp)
    # myGui.iconbitmap(r'c:\Python27\DLLs\thumbs-up-emoticon.ico')
    myGui.mainloop()

def run_gui(config_file, run_date=None):
    with open(config_file, 'r') as infile:
        config = json.load(infile)
    name = config.get('name', 'test_agent')
    folder = config['folder']
    base.config_logging(folder + name + ".log", level=logging.DEBUG,
                format = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                to_console = True, console_level = logging.INFO,
                to_msg = False,
                msg_conf = {'msg_class': 'loghandler_skype.SkypeHandler', \
                            'user_conf': dict({'group_name': 'AlgoTrade'}, **sec_bits.skype_user)},
                msg_level = logging.INFO)
    if run_date:
        year = run_date//10000
        month = (run_date % 10000)//100
        day = run_date % 100
        print(year, month, day)
        scur_day = datetime.date(year, month, day)
    else:
        scur_day = get_run_date()
    print('scur_day = %s' % scur_day.strftime('%Y%m%d'))
    myApp = MainApp(scur_day, config, master = None)
    myApp.restart()
    myGui = Gui(myApp)
    # myGui.iconbitmap(r'c:\Python27\DLLs\thumbs-up-emoticon.ico')
    myGui.mainloop()

def run(config_file, run_date=None):
    with open(config_file, 'r') as infile:
        config = json.load(infile)
    name = config.get('name', 'test_agent')
    folder = config['folder']
    base.config_logging(folder + name + ".log", level=logging.DEBUG,
                   format = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                   to_console = True,
                   console_level = logging.INFO)
    if run_date:
        year = run_date//10000
        month = (run_date % 10000)//100
        day = run_date % 100
        scur_day = datetime.date(year, month, day)
    else:
        scur_day = get_run_date()
    print('scur_day = %s' % scur_day.strftime('%Y%m%d'))
    agent_class = config.get('agent_class', 'Agent')
    cls_str = agent_class.split('.')
    agent_cls = getattr(__import__(str(cls_str[0])), str(cls_str[1]))
    agent = agent_cls(config=config, tday=scur_day)
    try:
        agent.restart()
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        agent.exit()

if __name__ == '__main__':
    args = sys.argv[1:]
    app_name = args[0]    
    params = {'config_file': args[1]}
    if len(args) > 2:        
        params['run_date'] = int(args[2])
    getattr(sys.modules[__name__], app_name)(**params)
    kw = input('press any key to exit\n')

