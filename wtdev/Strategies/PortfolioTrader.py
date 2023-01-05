from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import pandas as pd
import datetime
import json
from pycmqlib3.utility.misc import inst2exch, inst2product, inst2contmth
from pycmqlib3.utility.process_wt_data import wt_time_to_min_id


class StraPortTrader(BaseCtaStrategy):
    def __init__(self, name, pos_loc='C:/dev/pyktrader3/process/pt_test2',
                 reload_times=[2200, 2230, 905, 930, 1030, 1130, 1330, 1430],
                 exec_times=[2107, 2250, 907, 933, 1034, 1334]):
        BaseCtaStrategy.__init__(self, name)
        config_file = f'{pos_loc}/settings/{name}.json'
        with open(config_file, 'r') as fp:
            strat_conf = json.load(fp)
        strat_args = strat_conf['config']
        assets = strat_args['assets']
        num_assets = len(assets)
        self.__pos_loc = pos_loc
        self.__prod_list = [''] * num_assets
        self.__codes = [''] * num_assets
        self.__prev_codes = [''] * num_assets
        self.__target_pos = [0] * num_assets
        self.__reload_times = reload_times
        self.__exec_times = exec_times
        for idx, asset_dict in enumerate(assets):
            under = asset_dict["underliers"][0]
            if len(asset_dict["prev_underliers"])>0:
                prev_under = asset_dict["prev_underliers"][0]
            else:
                prev_under = ''
            prod = inst2product(under)
            self.__prod_list[idx] = prod
            exch = inst2exch(under)
            contmth = inst2contmth(under) % 10000
            self.__codes[idx] = '.'.join([exch, prod, str(contmth)])
            if len(prev_under) > 0:
                contmth = inst2contmth(prev_under) % 10000
                self.__prev_codes[idx] = '.'.join([exch, prod, str(contmth)])

    def on_init(self, context: CtaContext):
        for idx, code in enumerate(self.__codes):
            if idx == 0:
                context.stra_prepare_bars(code, 'm1', 10, isMain=True)
            else:
                context.stra_prepare_bars(code, 'm1', 10, isMain=False)

    def reload_position(self, context: CtaContext):
        cur_date = context.stra_get_tdate()
        pos_loc = self.__pos_loc
        strat_name = self.__name__
        pos_file = f'{pos_loc}/{strat_name}_{cur_date}.json'
        context.stra_log_text(f"reload curr position from {pos_file}")
        with open(pos_file, 'r') as fp:
            pos_dict = json.load(fp)
        for idx, prod in enumerate(self.__prod_list):
            self.__target_pos[idx] = pos_dict.get(prod, 0)
            context.stra_log_text("product=%s, contract=%s, target=%s" % (prod, self.__codes[idx], self.__target_pos[idx]))

    def on_session_begin(self, context:CtaContext, curTDate:int):
        self.reload_position(context)

    def on_calculate(self, context: CtaContext):
        cur_time = context.stra_get_time()        
        if cur_time in self.__reload_times:
            self.reload_position(context)
        if cur_time not in self.__exec_times:
            return
        context.stra_log_text("Executing the target position for %s" % cur_time)
        for idx, code in enumerate(self.__codes):
            sInfo = context.stra_get_sessinfo(code)
            if not sInfo.isInTradingTime(cur_time):
                continue
            prev_code = self.__prev_codes[idx]
            if len(prev_code) > 0:
                prev_pos = context.stra_get_position(prev_code)
                if prev_pos != 0:
                    context.stra_set_position(prev_code, 0, 'ExitPrevPosition')
                    context.stra_log_text(f"close prev position for {prev_code}")
            cur_pos = context.stra_get_position(code)
            target_pos = self.__target_pos[idx]
            if cur_pos != target_pos:
                context.stra_set_position(code, target_pos, 'AdjustPosition')
                context.stra_log_text(f"adjust position for {code} from {cur_pos} to {target_pos}")
            continue

    def on_tick(self, context: CtaContext, code: str, newTick: dict):
        return

    def on_bar(self, context: CtaContext, code: str, period: str, newBar: dict):
        return