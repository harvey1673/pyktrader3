import pandas as pd
from . cmq_rpc import RpcClient, RemoteException

CLIENT_SETTING = {'req': 'tcp://sgferd0217w:11100', 'sub': 'tcp://sgferd0217w:11110'}

class WindClient(RpcClient):
    def __init__(self, req, sub):
        super(WindClient, self).__init__(req, sub)

def wsi(*args, **kwargs):
    client = WindClient(CLIENT_SETTING['req'], CLIENT_SETTING['sub'])
    client.start()
    df = pd.read_json(client.wsi(*args, **kwargs))
    client.stop()
    return df

def wsd(*args, **kwargs):
    client = WindClient(CLIENT_SETTING['req'], CLIENT_SETTING['sub'])
    client.start()
    df = pd.read_json(client.wsd(*args, **kwargs))
    client.stop()
    return df

if __name__ == '__main__':
    pass