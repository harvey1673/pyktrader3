# -*- coding: utf-8 -*-
import datetime
import sys
import time
import logging
import json
from pycmqlib3.utility import misc, base, sec_bits
from pycmqlib3.core.agent_save import SaveAgent
from pycmqlib3.core.agent_email import EmailAgent
from pycmqlib3.core.agent import Agent
from pycmqlib3.core.gui_agent import MainApp, Gui

def save(config_file, tday):
    with open(config_file, 'r') as infile:
        config = json.load(infile)
    name = config.get('name', 'save_ctp')
    folder = config['folder']
    filter_flag = config.get('filter_flag', False)
    base.config_logging(folder + name + ".log", level=logging.DEBUG,
                   format = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                   to_console = True,
                   console_level = logging.INFO)
    scur_day = datetime.datetime.strptime(tday, '%Y%m%d').date()
    save_agent = SaveAgent(config = config, tday = scur_day)
    if 'instIDs' in config:
        curr_insts = config['instIDs']
    else:
        curr_insts = misc.filter_main_cont(tday, filter_flag)
    for inst in curr_insts:
        save_agent.add_instrument(inst)
    try:
        save_agent.restart()
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        save_agent.exit()

def save_gui(config_file, tday):
    with open(config_file, 'r') as infile:
        config = json.load(infile)
    name = config.get('name', 'save_ctp')
    folder = config['folder']
    filter_flag = config.get('filter_flag', False)
    base.config_logging(folder + name + ".log", level=logging.DEBUG,
                   format = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                   to_console = True,
                   console_level = logging.INFO)
    scur_day = datetime.datetime.strptime(tday, '%Y%m%d').date()
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

def run_gui(config_file, tday):
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
    scur_day = datetime.datetime.strptime(tday, '%Y%m%d').date()
    myApp = MainApp(scur_day, config, master = None)
    myApp.restart()
    myGui = Gui(myApp)
    # myGui.iconbitmap(r'c:\Python27\DLLs\thumbs-up-emoticon.ico')
    myGui.mainloop()

def run(config_file, tday):
    with open(config_file, 'r') as infile:
        config = json.load(infile)
    name = config.get('name', 'test_agent')
    folder = config['folder']
    base.config_logging(folder + name + ".log", level=logging.DEBUG,
                   format = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s',
                   to_console = True,
                   console_level = logging.INFO)
    scur_day = datetime.datetime.strptime(tday, '%Y%m%d').date()
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
    params = (args[1], args[2], )
    getattr(sys.modules[__name__], app_name)(*params)
