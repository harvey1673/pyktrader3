import datetime
from pycmqlib3.analytics import data_handler

day_split_dict = {'s1': [300, 2115],
                  's2': [300, 1500, 2115],
                  's3': [300, 1500, 1900, 2115],
                  's4': [300, 1500, 1630, 1900, 2115],}

class BarManager(object):
    def __init__(self, freq='m1', bar_array = None, data_args= None):
        self.func_list = []
        self.updated_bar_id = 0
        self.updated_date = datetime.date.today()
        self.split_idx = 0
        self.freq = int(freq[1:])
        self.bar_array = None
        if bar_array:
            self.bar_array = bar_array
            self.refresh_data()
        if data_args:
            self.bar_array = data_handler.DynamicRecArray(**data_args)

    def __len__(self):
        return self.bar_array.length

    def test_print(self):
        print(self.updated_date, self.updated_bar_id, self.split_idx)

    @property
    def data(self):
        return self.bar_array.data

    def refresh_data(self, bar_update = None):
        if bar_update:
            if bar_update[0] > self.updated_date:
                self.updated_date = bar_update[0]
                self.updated_bar_id = 0
            else:
                self.updated_bar_id = self.bar_array.data['bar_id'][-1]
            self.split_idx = int(bar_update[1] / self.freq)
        else:
            self.updated_date = self.bar_array.data['date'][-1]
            self.updated_bar_id = self.bar_array.data['bar_id'][-1]
            self.split_idx = int(self.updated_bar_id / self.freq)

    def check_bar_update(self, bar_date, bar_id, forced = False):
        if forced or (self.updated_date < bar_date) or (self.split_idx < int(bar_id / self.freq)):
            return True
        else:
            return False

    def check_data_update(self, min_data):
        if (min_data['date'][-1] > self.updated_date) or (min_data['bar_id'][-1] > self.updated_bar_id):
            if (min_data['date'][-1] == self.updated_date) and \
                    (int(min_data['bar_id'][-1]/self.freq) == int(self.updated_bar_id/self.freq)):
                status = 1
            else:
                status = 2
        else:
            status = 0
        return status

    def update(self, min_data, curr_date, curr_bar, forced = False):
        if self.check_bar_update(curr_date, curr_bar, forced):
            data_status = self.check_data_update(min_data)
            if data_status == 1:
                self.bar_array.remove_lastn(1)
            if data_status > 0:
                idx = 0
                rlen = len(min_data)
                for i in range(rlen):
                    if (min_data['bar_id'][rlen - i - 1] <= self.bar_array.data['bar_id'][-1]) and \
                            (min_data['date'][rlen - i - 1] <= self.bar_array.data['date'][-1]):
                        idx = i
                        break
                if idx > 0:
                    new_data = {'datetime': min_data['datetime'][-idx], 'open': min_data['open'][-idx],
                                'high': max(min_data['high'][-idx:]), \
                                'low': min(min_data['low'][-idx:]), 'close': min_data['close'][-1], \
                                'volume': sum(min_data['volume'][-idx:]), 'openInterest': min_data['openInterest'][-1], \
                                'min_id': min_data['min_id'][-1], 'bar_id': min_data['bar_id'][-1],
                                'date': min_data['date'][-1]}
                    if new_data['high'] > new_data['low']:
                        self.bar_array.append_by_dict(new_data)
            self.refresh_data([curr_date, curr_bar])
            return True
        else:
            return False

    def set_bar_array(self, bar_array = None, data_args = None):
        if bar_array:
            self.bar_array = bar_array
        elif data_args:
            self.bar_array = data_handler.DynamicRecArray(**data_args)
        self.refresh_data()

    def set_freq(self, freq):
        self.freq = freq

    def add_func(self, fobj):
        for func in self.func_list:
            if func.name == fobj.name:
                return False
        self.func_list.append(fobj)
        return True

    def run_init_func(self, df):
        for fobj in self.func_list:
            ts = fobj.sfunc(df)
            if type(ts).__name__ == 'Series':
                df[ts.name] = ts
            elif type(ts).__name__ == 'DataFrame':
                for col_name in ts.columns:
                    df[col_name] = ts[col_name]

    def run_update_func(self):
        for fobj in self.func_list:
            fobj.rfunc(self.bar_array.data)


class DaySplitBarManager(BarManager):
    def __init__(self, freq='s2', bar_array=None,  data_args= None, bar_func = data_handler.bar_conv_func2):
        min_split = day_split_dict[freq]
        self.bar_func = bar_func
        self.bar_split = [self.bar_func(min_id) for min_id in min_split]
        self.updated_bar_id = 0
        self.updated_date = datetime.date(2018,1,1)
        self.split_idx = 0
        super(DaySplitBarManager, self).__init__(freq, bar_array, data_args)

    def refresh_data(self, bar_update = None):
        if bar_update:
            if bar_update[0] > self.updated_date:
                self.updated_date = bar_update[0]
                self.updated_bar_id = 0
            else:
                self.updated_bar_id = self.bar_array.data['bar_id'][-1]
            curr_bar = bar_update[1]
        else:
            self.updated_date = self.bar_array.data['date'][-1]
            self.updated_bar_id = self.bar_array.data['bar_id'][-1]
            curr_bar = self.updated_bar_id
        for idx in range(len(self.bar_split[:-1])):
            self.split_idx = idx
            if (curr_bar >= self.bar_split[idx]) and (curr_bar < self.bar_split[idx + 1]):
                break

    def check_bar_update(self, bar_date, bar_id, forced=False):
        if forced or (self.updated_date < bar_date) or (bar_id >= self.bar_split[self.split_idx + 1]):
            return True
        else:
            return False

    def check_data_update(self, min_data):
        last_date = self.bar_array.data['date'][-1]
        last_bar = self.bar_array.data['bar_id'][-1]
        if (min_data['date'][-1] > last_date) or (min_data['bar_id'][-1] > last_bar):
            if (min_data['date'][-1] == last_date) and (last_bar >= self.bar_split[self.split_idx]):
                status = 1
            else:
                status = 2
        else:
            status = 0
        return status