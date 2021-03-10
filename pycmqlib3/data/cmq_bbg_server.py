# -*- coding: utf-8 -*-
from cmq_rpc import RpcServer
import pandas as pd
import json
import time
from tia.bbg import LocalTerminal

SERVER_SETTING = {'rep': 'tcp://0.0.0.0:11010', 'pub': 'tcp://0.0.0.0:11020'}

class BBGServer(RpcServer):       
    def __init__(self, rep, pub):
        super(BBGServer, self).__init__(rep, pub)
        self.register(self.bbg_terminal)
        
    def bbg_terminal(self, func, *args, **kwargs):
        """
        Get data from Bloomberg terminal
        """
        try:
            func = getattr(LocalTerminal, func)
        except Exception:
            return json.dumps({'error': 'no such function'})
        
        try:
            resp = func(*args, **kwargs).as_frame()
            return resp.to_json()
        except:
            return json.dumps({'error': 'something wrong in calculation'})

def run_bbg_server(setting = SERVER_SETTING):
    server = BBGServer(setting['rep'], setting['pub'])
    try:
        server.start()
        while 1:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()

if __name__ == '__main__':
    pass

