import pandas as pd
from cmq_rpc import RpcClient

CLIENT_SETTING = {'req': 'tcp://sgferd0257w:11010', 'sub': 'tcp://sgferd0257w:11020'}

class BBGClient(RpcClient):
    def __init__(self, req, sub):
        super(BBGClient, self).__init__(req, sub)

    def terminal(self, func, *args, **kwargs):
        """
        Client side function for getting data from bloomberg
        param func: function name as a string
        """
        bbg_terminal = getattr(self, 'bbg_terminal')
        try:
            df = pd.read_json(bbg_terminal(func, *args, **kwargs))
            return df
        except Exception as e:
            print(e)
            return None


def get_hist_data(index_list, start, end, setting = CLIENT_SETTING):
    client = BBGClient(setting['req'], setting['sub'])
    client.start()
    out = {}
    for idx in index_list:
        df = client.terminal("get_historical", idx, ['PX_LAST'], start=start, end=end)
        try:
            if (len(df) > 0):
                out[idx] = pd.DataFrame({'date': df.index, 'close': df['PX_LAST'], 'spotID': idx})
        except:
            continue
    client.stop()
    return out

if __name__ == '__main__':
    pass