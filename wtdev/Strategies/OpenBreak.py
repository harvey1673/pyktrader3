import pandas as pd

from wtpy import BaseCtaStrategy
from wtpy import CtaContext
from pycmqlib3.analytics.data_handler import ATR, day_split1
from pycmqlib3.utility.process_wt_data import wt_time_to_min_id
import numpy as np


class StraOpenBreak(BaseCtaStrategy):

    def __init__(self, name: str, code: str,
                 split_mode: str = 's1',
                 run_period: str = 'm1',
                 look_back: int = 1,
                 ratio: float = 1.0,
                 min_range: float = 0.4,
                 atr_win: int = 10,
                 close_tday: bool = False,
                 risk_factor: float = 1.0,
                 vol_adj: bool = False,
                 cleartimes: list = [[1455, 1515]],
                 isForStk:bool = False):
        BaseCtaStrategy.__init__(self, name)
        self.__trade_unit = 1
        self.__risk_factor = risk_factor
        self.__vol_adj = vol_adj
        self.__close_tday = close_tday
        self.__split_mode = split_mode
        self.__run_period = run_period
        self.__look_back = look_back
        self.__cleartimes = cleartimes
        self.__minlist = []
        self.__bar_cnt = max(self.__look_back, 4) + 4
        if 's' in split_mode:
            self.__period = 'm15'
            self.__bar_cnt = max(self.__look_back + 1, 2)*48
            if split_mode == 's1':
                self.__minlist = [300]
            elif split_mode == 's2':
                self.__minlist = [300, 1500]
                self.__bar_cnt = self.__bar_cnt // 2
            elif split_mode == 's3':
                self.__minlist = [300, 1500, 1900]
                self.__bar_cnt = self.__bar_cnt // 3
        else:
            self.__period = split_mode

        self.__ratio = ratio
        self.__min_rng = min_range
        self.__atr_win = atr_win
        self.__atr = np.nan
        self.__rng = np.nan
        self.__open = np.nan
        self.__cur_min = 0
        self.__rng_bars = pd.DataFrame()
        self.__code = code
        self.__is_stk = isForStk
        self.__theCode = code
        if self.__is_stk:
            self.__theCode = self.__theCode + "Q"

    def on_init(self, context: CtaContext):
        context.stra_prepare_bars(self.__code, 'd1', self.__atr_win + 5, isMain=False)
        self.update_range(context)
        self.__open = self.__rng_bars['open'].iloc[-1]
        self.__cur_min = wt_time_to_min_id(self.__rng_bars['bartime'].iloc[-1] % 10000)
        context.stra_prepare_bars(self.__code, self.__run_period, 10, isMain=True)
        context.stra_log_text("OpenBreak inited")

    def on_session_begin(self, context:CtaContext, curTDate:int):
        pInfo = context.stra_get_comminfo(self.__code)
        vol_scale = pInfo.volscale
        self.__cur_min = wt_time_to_min_id(context.stra_get_time())
        daily_bars = context.stra_get_bars(self.__code, 'd1', self.__atr_win + 5, isMain=False)
        daily_bars = daily_bars.to_df()
        self.__atr = ATR(daily_bars, self.__atr_win).iloc[-1]
        self.update_range(context)
        self.__open = np.nan
        if self.__vol_adj:
            self.__trade_unit = int(self.__risk_factor / (self.__atr * vol_scale))
        else:
            self.__trade_unit = int(self.__risk_factor)

    def update_range(self, context:CtaContext):
        df_bars = context.stra_get_bars(self.__code, self.__period,
                                         self.__bar_cnt,
                                         isMain=False)
        df_bars = df_bars.to_df()
        if 's' in self.__split_mode:
            df_bars['min_id'] = df_bars['bartime'].apply(lambda x: wt_time_to_min_id(x % 10000))
            df_bars = df_bars.rename(columns={'bartime': 'datetime',
                                              'hold': 'openInterest',
                                              'diff': 'diff_oi',})
            df_bars = day_split1(df_bars, minlist=self.__minlist, index_col=None)
            df_bars = df_bars.rename(columns={'datetime': 'bartime', 'openInterest': 'hold'})
            print(df_bars, df_bars.columns)
        lookback = self.__look_back
        closes = df_bars['close'].values
        highs = df_bars['high'].values
        lows = df_bars['low'].values
        hh = np.amax(highs[-lookback:])
        hc = np.amax(closes[-lookback:])
        ll = np.amin(lows[-lookback:])
        lc = np.amin(closes[-lookback:])
        self.__rng = max(hh-lc, hc-ll) * self.__ratio
        self.__rng_bars = df_bars

    def on_calculate(self, context: CtaContext):
        code = self.__code
        df_bars = context.stra_get_bars(self.__theCode, self.__run_period, 10, isMain=True)
        highpx = df_bars.highs[-1]
        lowpx = df_bars.lows[-1]
        if np.isnan(self.__open):
            self.__open = df_bars.opens[-1]
        upper_bound = self.__open + self.__ratio * self.__rng
        lower_bound = self.__open - self.__ratio * self.__rng
        trdUnit = 1
        if self.__is_stk:
            trdUnit = 100
        curPos = context.stra_get_position(code) / trdUnit
        curTime = context.stra_get_time()
        if self.__close_tday and (curPos != 0):
            for tmPair in self.__cleartimes:
                if curTime >= tmPair[0] and curTime <= tmPair[1]:
                    context.stra_set_positions(code, 0, "clear")
                    context.stra_log_text("尾盘清仓")
                    return
        if (curPos <= 0) and (highpx >= upper_bound):
            context.stra_set_position(code, self.__trade_unit * trdUnit, 'enterlong')
            context.stra_log_text("向上突破%.2f>=%.2f，多仓进场" % (highpx, upper_bound))
        elif (curPos >= 0) and (lowpx <= lower_bound):
            context.stra_set_position(code, -self.__trade_unit * trdUnit, 'entershort')
            context.stra_log_text("向下突破%.2f<=%.2f，空仓进场" % (lowpx, lower_bound))
        return

    def on_bar(self, context:CtaContext, code:str, period:str, newBar:dict):
        #context.stra_log_text(f"on bar fired: {code}-{period}")
        if period == self.__run_period:
            cur_min = wt_time_to_min_id(context.stra_get_time())
            if self.__split_mode in ['s2', 's3']:
                for min in self.__minlist:
                    if (cur_min >= min) and (self.__cur_min < min):
                        self.update_range(context)
                        self.__open = newBar['open']
                        break
            self.__cur_min = cur_min
        elif (period == self.__period) and ('s' not in self.__split_mode):
            self.update_range(context)
            self.__open = np.nan

    def on_tick(self, context: CtaContext, stdCode: str, newTick: dict):
        # context.stra_log_text ("on tick fired")
        return