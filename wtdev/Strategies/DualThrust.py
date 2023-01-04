from wtpy import BaseCtaStrategy
from wtpy import CtaContext
import numpy as np

class StraDualThrust(BaseCtaStrategy):
    
    def __init__(self, name:str, code:str, period:str, look_back:int, ratio:float, isForStk:bool = False):
        BaseCtaStrategy.__init__(self, name)
        self.__lookback = look_back
        self.__ratio = ratio
        self.__period = period
        self.__code = code
        self.__theCode = code
        self.__is_stk = isForStk
        self.__cur_tdate = 0
        if self.__is_stk:
            self.__theCode = self.__theCode + "Q"
        self.__rng = np.nan

    def on_init(self, context:CtaContext):
        df_bars = context.stra_get_bars(self.__theCode, 'd1', self.__lookback+5, isMain=False)
        self.__cur_tdate = df_bars.bartimes[-1]
        context.stra_prepare_bars(self.__theCode, self.__period, 30, isMain=True)
        context.stra_log_text("DualThrust inited")

    def update_range(self, context:CtaContext):
        df_bars = context.stra_get_bars(self.__theCode, 'd1', self.__lookback + 5, isMain=False)
        lookback = self.__lookback
        closes = df_bars.closes
        highs = df_bars.highs
        lows = df_bars.lows
        hh = np.amax(highs[-lookback:])
        hc = np.amax(closes[-lookback:])
        ll = np.amin(lows[-lookback:])
        lc = np.amin(closes[-lookback:])
        self.__rng = max(hh-lc, hc-ll) * self.__ratio

    def on_calculate(self, context:CtaContext):
        cur_tdate = context.stra_get_tdate()
        if cur_tdate > self.__cur_tdate:
            self.update_range(context)
            self.__cur_tdate = cur_tdate
        code = self.__code
        openpx = context.stra_get_day_price(self.__theCode, 0)
        upper_bound = openpx + self.__rng
        lower_bound = openpx - self.__rng
        trdUnit = 1
        if self.__is_stk:
            trdUnit = 100
        df_bars = context.stra_get_bars(self.__theCode, self.__period, 10, isMain=True)
        highpx = df_bars.highs[-1]
        lowpx = df_bars.lows[-1]
        closepx = df_bars.closes[-1]
        curPos = context.stra_get_position(code) / trdUnit
        if (curPos <= 0) and (highpx >= upper_bound):
            context.stra_set_position(code, 1 * trdUnit, 'enterlong')
            context.stra_log_text("向上突破%.2f>=%.2f，多仓进场: open=%.2f,rng=%.2f, close=%.2f" %
                                  (highpx, upper_bound, openpx, self.__rng, closepx))
        elif (curPos >= 0) and (lowpx <= lower_bound):
            context.stra_set_position(code, -1 * trdUnit, 'entershort')
            context.stra_log_text("向下突破%.2f<=%.2f，空仓进场: open=%.2f,rng=%.2f, close=%.2f" %
                                  (lowpx, lower_bound, openpx, self.__rng, closepx))
        return

    def on_bar(self, context:CtaContext, stdCode:str, period:str, newBar:dict):
        return

    def on_tick(self, context:CtaContext, stdCode:str, newTick:dict):
        #context.stra_log_text ("on tick fired")
        return