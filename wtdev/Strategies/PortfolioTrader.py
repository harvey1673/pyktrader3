from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np
import pandas as pd
import datetime
import json
from pycmqlib3.utility.misc import inst2exch, inst2product
from pycmqlib3.utility.process_wt_data import wt_time_to_min_id


class StraPortTrader(BaseCtaStrategy):
    def __init__(self, name, pos_loc='C:/dev/pyktrader3/process/'):
        BaseCtaStrategy.__init__(self, name)
        config_file = f'{pos_loc}settings/{name}.json'
        with open(config_file, 'r') as fp:
            strat_conf = json.load(fp)
        strat_args = strat_conf['config']
        assets = strat_args['assets']
        num_assets = len(assets)
        self.__prod_list = [''] * num_assets
        self.__codes = [''] * num_assets
        self.__prev_codes = [''] * num_assets
        for idx, asset_dict in enumerate(assets):
            under = asset_dict["underliers"][0]
            prev_under = asset_dict["prev_underliers"][0]
            self.__prod_list[idx] = inst2product(under)
            exch = inst2exch(under)
            self.__codes[idx] = exch + '.' + under
            if len(prev_under) > 0:
                self.__prev_codes[idx] = exch + '.' + prev_under
        self.__pos_sum = pd.DataFrame()
        self.___pos = [0] * num_assets

    def on_init(self, context: CtaContext):
        for idx, code in enumerate(self.__codes):
            if idx == 0:
                context.stra_prepare_bars(code, 'm1', 10, isMain=True)
            else:
                context.stra_prepare_bars(code, 'm1', 10, isMain=False)

    def reload_position(self, context: CtaContext):
        cur_date = context.stra_get_date()

    def on_session_begin(self, context:CtaContext, curTDate:int):
        self.reload_position(context)

    def on_calculate(self, context: CtaContext):
        cur_time = context.stra_get_time()
        cur_min = wt_time_to_min_id(cur_time)
        if cur_min in [301, 1501]:
            self.load_fact_data(context)
        if cur_min not in self.__exec_bar_list:
            return
        for idx, code in enumerate(self.__codes):
            sInfo = context.stra_get_sessioninfo(code)
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