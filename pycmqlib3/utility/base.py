#-*- coding:utf-8 -*-
import sys
import functools
import logging.handlers
import logging
import decorator
from inspect import (
            getargspec,
        )

MY_FORMAT = '%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s'
CONSOLE_FORMAT = '%(asctime)s%(message)s'

def config_logging(filename,level=logging.DEBUG,format=MY_FORMAT, \
                   to_console=True, console_level=logging.INFO, \
                   to_msg = False, msg_level = logging.INFO, \
                   msg_conf={'msg_class': 'loghandler_skype.SkypeHandler', \
                             'user_conf': {'nickName': 'harvey'}}):
    logging.basicConfig(\
        handlers = [ logging.FileHandler(filename, 'w', 'utf-8')], level=level, format=format)
    #my_logger = logging.getLogger('root')
    #my_handler = logging.handlers.RotatingFileHandler(filename, mode='a', maxBytes=100*1024, \
    #                                backupCount=2, encoding=None, delay=0)
    #formatter = logging.Formatter(format)
    #my_handler.setFormatter(formatter)
    #my_handler.setLevel(level)
    #my_logger.addHandler(my_handler)
    if to_console:
        add_log2console(console_level)
    if to_msg:
        add_log2msg(msg_conf = msg_conf, level = msg_level)

def add_log2console(level = logging.INFO):
    console = logging.StreamHandler()
    console.setLevel(level)
    # set a format which is simpler for console use
    #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    #formatter = logging.Formatter('%(name)s:%(funcName)s:%(lineno)d:%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter(CONSOLE_FORMAT)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

def add_log2msg(msg_conf = {'msg_class': 'loghandler_skype.SkypeHandler', 'user_conf': {'nickName': 'harvey'}}, \
                level = logging.INFO):
    handler_str = msg_conf['msg_class']
    handler_mod = handler_str.split('.')
    if len(handler_mod) > 1:
        handler_class = getattr(__import__(str(handler_mod[0])), str(handler_mod[1]))
    else:
        handler_class = eval(handler_str)
    msg_handler = handler_class(user_conf = msg_conf['user_conf'])
    msg_handler.setLevel(level)
    formatter = logging.Formatter(CONSOLE_FORMAT)
    msg_handler.setFormatter(formatter)
    logging.getLogger().addHandler(msg_handler)

def fcustom(func,**kwargs):
    pf = functools.partial(func,**kwargs)
    pf.paras = ','.join(['%s=%s' % item for item in list(pf.keywords.items())])
    pf.__name__ = '%s:%s' % (func.__name__,pf.paras)
    return pf

def type_name(cobj): 
    clazz_obj = cobj
    while(isinstance(clazz_obj,functools.partial)):
        clazz_obj = clazz_obj.func
    aname = str(type(clazz_obj))[8:-2]
    return aname.split('.')[-1]

def module_name(cobj): 
    clazz_obj = cobj
    while(isinstance(clazz_obj,functools.partial)):
        clazz_obj = clazz_obj.func
    aname = str(type(clazz_obj))[8:-2]
    return aname.split('.')[0]

def class_name(cobj): 
    clazz_obj = cobj
    while(isinstance(clazz_obj,functools.partial)):
        clazz_obj = clazz_obj.func
    aname = str(type(clazz_obj))[8:-2]
    return tuple(aname.split('.'))

class BaseObject(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

    def has_attr(self,attr_name):
        return attr_name in self.__dict__

    def get_attr(self,attr_name):
        return self.__dict__[attr_name]

    def set_attr(self,attr_name,value):
        self.__dict__[attr_name] = value

    def __repr__(self):
        return 'BaseObject'


class CommonObject(BaseObject):
    def __init__(self,id,**kwargs):
        BaseObject.__init__(self,**kwargs)
        self.id = id

    def __repr__(self):
        return 'CommonObject'