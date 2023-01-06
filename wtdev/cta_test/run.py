from ConsoleIdxWriter import ConsoleIdxWriter
from wtpy import WtEngine, EngineType
from wtdev.Strategies.DualThrust import StraDualThrust
from wtdev.Strategies.PortfolioTrader import StraPortTrader
#from wtdev.Strategies.OpenBreak import StraOpenBreak
#from wtdev.Strategies.FactorPort_Sel import StraFactorPortSel


if __name__ == "__main__":
    #创建一个运行环境，并加入策略
    env = WtEngine(EngineType.ET_CTA)
    env.init('../common/', "config.yaml")

    strat_name = 'PT_FACTPORT3'
    stra_info = StraPortTrader(strat_name, pos_loc='C:/dev/pyktrader3/process/pt_test3',
                 reload_times=[2105, 2148, 905, 931, 1035, 1332, 1430],
                 exec_times=[2107, 2250, 907, 933, 1036, 1334])
    env.add_cta_strategy(stra_info)
    idxWriter = ConsoleIdxWriter()
    env.set_writer(idxWriter)
    env.run()
    kw = input('press any key to exit\n')