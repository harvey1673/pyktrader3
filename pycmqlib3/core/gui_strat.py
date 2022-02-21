#-*- coding:utf-8 -*-
import tkinter as tk
import tkinter.ttk as ttk
from . gui_misc import type2str, keepdigit, type2str, get_type_var, str2type, ScrolledFrame

class StratGui(object):
    def __init__(self, strat, app, master):
        self.root = master
        self.name = strat.name
        self.app = app
        self.underliers = strat.underliers
        self.entries = {}
        self.stringvars = {}
        self.entry_fields = []
        self.status_fields = []
        self.shared_fields = []
        self.button_fields = []
        self.field_types = {}
        self.scr_frame = None
                
    def get_params(self):
        fields = self.entry_fields + self.status_fields
        params = self.app.get_strat_params(self.name, fields)
        for field in fields:
            if field in self.shared_fields:
                value = params[field]
                if field in self.entry_fields:
                    ent = self.entries[field]
                    ent.delete(0, tk.END)
                    ent.insert(0, value)
                elif field in self.status_fields:
                    self.stringvars[field].set(keepdigit(value, 4))
            else:
                for idx, underlier in enumerate(self.underliers):
                    under_key = ','.join(underlier)
                    value = params[field][idx]
                    if type(value).__name__ in ['float', 'float64']:
                        value = round(value, 2)
                    vtype = self.field_types[field]
                    value = type2str(value, vtype)
                    if field in self.entry_fields:
                        ent = self.entries[under_key][field]
                        ent.delete(0, tk.END)
                        ent.insert(0, value)
                    elif field in self.status_fields:
                        self.stringvars[under_key][field].set(keepdigit(value, 4))
        
    def set_params(self):
        params = {}
        for field in self.entry_fields:
            if field in self.shared_fields:
                ent = self.entries[field]
                value = ent.get()
                vtype = self.field_types[field]
                value = str2type(value, vtype)
                params[field] = value
            else:
                params[field] = []
                for underlier in self.underliers:
                    under_key = ','.join(underlier)
                    ent = self.entries[under_key][field]
                    value = ent.get()
                    vtype = self.field_types[field]
                    value = str2type(value, vtype)
                    params[field].append(value)
        self.app.set_strat_params(self.name, params)

    def OnFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
    def set_frame(self, root):
        self.scr_frame = ScrolledFrame(root)
        entries = {}
        stringvars = {}
        row_id = 0
        set_btn = ttk.Button(self.scr_frame.frame, text='Set', command=self.set_params)
        set_btn.grid(row=row_id, column=3, sticky="ew")
        refresh_btn = ttk.Button(self.scr_frame.frame, text='Refresh', command=self.get_params)
        refresh_btn.grid(row=row_id, column=4, sticky="ew")
        recalc_btn = ttk.Button(self.scr_frame.frame, text='Recalc', command=self.recalc)
        recalc_btn.grid(row=row_id, column=5, sticky="ew")
        save_btn = ttk.Button(self.scr_frame.frame, text='SaveConfig', command=self.save_config)
        save_btn.grid(row=row_id, column=6, sticky="ew")
        row_id += 1
        for idx, field in enumerate(self.shared_fields):
            lbl = ttk.Label(self.scr_frame.frame, text = field, anchor='w', width = 8)
            lbl.grid(row=row_id, column=idx+2, sticky="ew")
            if field in self.entry_fields:
                ent = ttk.Entry(self.scr_frame.frame, width=4)
                ent.grid(row=row_id+1, column=idx+2, sticky="ew")
                ent.insert(0, "0")
                entries[field] = ent
            elif field in self.status_fields:
                v= get_type_var(self.field_types[field])
                lab = ttk.Label(self.scr_frame.frame, textvariable = v, anchor='w', width = 8)
                lab.grid(row=row_id+1, column=idx+2, sticky="ew")
                v.set('0')
                stringvars[field] = v
            else:
                for (f, func) in self.button_fields:
                    if field == f:
                        button = ttk.Button(self.scr_frame.frame, text= field, width=6,
                                                command=lambda: self.run_func(func))
                        button.grid(column= idx+2, row=row_id + 1, sticky="ew")
                        break
        row_id += 2
        local_entry_fields = [ f for f in self.entry_fields if f not in self.shared_fields]
        local_status_fields = [ f for f in self.status_fields if f not in self.shared_fields]
        local_button_fields = [ (f, func) for (f, func) in self.button_fields if f not in self.shared_fields]
        fields = ['assets'] + [''] * len(local_button_fields) + local_entry_fields + local_status_fields
        for idx, field in enumerate(fields):
            lbl = ttk.Label(self.scr_frame.frame, text = field, anchor='w', width = 12)
            lbl.grid(row=row_id, column=idx, sticky="ew")
        row_id += 1
        for i, underlier in enumerate(self.underliers):
            under_key = ','.join(underlier)
            inst_lbl = ttk.Label(self.scr_frame.frame, text=under_key, anchor="w", width = 12)
            inst_lbl.grid(row=row_id, column=0, sticky="ew")
            col_id = 1
            entries[under_key] = {}
            for txt, func in local_button_fields:
                button = ttk.Button(self.scr_frame.frame, text = txt, width=6, command=lambda: self.run_func(func, idx))
                button.grid(column=col_id, row=row_id, sticky="ew")
                col_id += 1
            for idx, field in enumerate(local_entry_fields):
                ent = ttk.Entry(self.scr_frame.frame, width=5)
                ent.grid(row=row_id, column=col_id+idx, sticky="ew")
                ent.insert(0, "0")
                entries[under_key][field] = ent
            col_id += len(local_entry_fields)
            stringvars[under_key] = {}            
            for idx, field in enumerate(local_status_fields):
                v= get_type_var(self.field_types[field])
                lab = ttk.Label(self.scr_frame.frame, textvariable = v, anchor='w', width = 8)
                lab.grid(row=row_id, column=col_id+idx, sticky="ew")
                v.set('0')
                stringvars[under_key][field] = v       
            row_id +=1
        self.entries = entries
        self.stringvars = stringvars        
        self.get_params()
    
    def recalc(self):
        self.app.run_strat_func(self.name, 'initialize')

    def save_config(self):
        self.app.run_strat_func(self.name, 'save_config')

    def run_func(self, func, idx = None):
        if idx:
            self.app.run_strat_func(self.name, func, (idx,))
        else:
            self.app.run_strat_func(self.name, func)

class ManualTradeGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'RunFlag', 'AllocW', 'CloseTday', 'IsDisabled', 'MaxVol', 'TimePeriod', 'PriceType', \
                             'LongPrice', 'LongStop', 'ShortPrice', 'ShortStop', 'TickNum', 'OrderOffset' ]
        self.status_fields = ['TradeUnit', 'CurrPos', 'CurrPrices']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.button_fields = [('long', 'open_long'), ('short', 'open_short'), ('liq', 'liquidate_tradepos')]
        self.field_types = {'RunFlag':'int',
                            'TradeUnit':'int',
                            'CurrPos': 'int',
                            'MaxVol':'int',
                            'TimePeriod': 'int',
                            'PriceType': 'str',
                            'LongPrice': 'float',
                            'LongStop': 'float',
                            'ShortPrice': 'float',
                            'ShortStop': 'float',
                            'TickNum': 'int',
                            'OrderOffset': 'bool',
                            'CloseTday': 'bool',
                            'IsDisabled': 'bool',
                            'CurrPrices': 'float',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            }

class BulkTradeGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'TradeFile', 'RunFlag', 'AllocW', 'IsDisabled', 'TimePeriod', 'PriceType', 'Direction', \
                             'LimitPrice', 'TickNum']
        self.status_fields = ['TradeUnit', 'CurrPos', 'CurrPrices']
        self.shared_fields = ['PosScaler', 'IsDisabled', 'TradeFile']
        self.field_types = {'RunFlag': 'int',
                            'TradeUnit': 'int',
                            'CurrPos': 'int',
                            'MaxVol': 'int',
                            'TimePeriod': 'int',
                            'PriceType': 'str',
                            'Direction': 'int',
                            'LimitPrice': 'float',
                            'TickNum': 'int',
                            'IsDisabled': 'bool',
                            'CurrPrices': 'float',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            'TradeFile': 'str',
                            }
    def set_frame(self, root):
        StratGui.set_frame(self, root)
        row_id = 0
        col_id = 7
        load_btn = ttk.Button(self.scr_frame.frame, text='LoadTrades', command=self.load_trades)
        load_btn.grid(row = row_id, column = col_id, sticky="ew")
        col_id += 1
        bulk_btn = ttk.Button(self.scr_frame.frame, text='BulkTrades', command=self.bulk_trades)
        bulk_btn.grid(row = row_id, column= col_id, sticky="ew")

    def load_trades(self):
        self.app.run_strat_func(self.name, 'load_trades')

    def bulk_trades(self):
        self.app.run_strat_func(self.name, 'bulk_trades')

class XSMOMRetGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'AllocW']
        self.status_fields = ['Batch', 'TradeUnit', 'Ret', 'CurrPos', 'CurrPrices']
        self.shared_fields = ['PosScaler']
        self.field_types = {'RunFlag': 'int',
                            'Batch': 'str',
                            'Ret': 'float',
                            'TradeUnit': 'int',
                            'CurrPos': 'int',
                            'CurrPrices': 'float',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            }

class FactorPortGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'IsDisabled']
        self.status_fields = ['VolWeight', 'CurrPos', 'TargetPos', 'CurrPrices', 'MinTradeSize']
        self.button_fields = [('LoadFact', 'load_fact_data'), ('TradeTarget', 'trade_target_pos')]
        self.shared_fields = ['IsDisabled', 'PosScaler', 'LoadFact', 'TradeTarget']
        self.field_types = {'RunFlag': 'int',
                            'TradeUnit': 'int',
                            'CurrPos': 'int',
                            'TargetPos': 'int',
                            'VolWeight': 'float',
                            'CurrPrices': 'float',
                            'PosScaler': 'float',
                            'IsDisabled': 'bool',
                            'MinTradeSize': 'int',
                            }

class DTSplitChanGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'RunFlag', 'Freq', 'AllocW', 'Channels', 'Lookbacks', 'Ratios', 'PriceMode', 'CloseTday', 'IsDisabled']
        self.status_fields = ['TdayOpen', 'VolRatio', 'TradeUnit', 'CurrPos', 'CurrPrices', 'BuyTrig', 'SellTrig', 'SplitMode', 'CurRng', 'ChanHigh', 'ChanLow', 'MaLevel', 'MaChan']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.field_types = {'RunFlag': 'int',
                            'TradeUnit': 'int',
                            'CurrPos': 'int',
                            'Lookbacks': 'int',
                            'Ratios': 'float',
                            'VolRatio': 'floatlist',
                            'PriceMode': 'str',
                            'SplitMode': 'str',
                            'CloseTday': 'bool',
                            'IsDisabled': 'bool',
                            'TdayOpen': 'float',
                            'CurrPrices': 'float',
                            'CurRng': 'float',
                            'MaLevel': 'float',
                            'ChanHigh': 'float',
                            'ChanLow': 'float',
                            'BuyTrig': 'float',
                            'SellTrig': 'float',
                            'Channels': 'int',
                            'MinRng': 'float',
                            'MaChan': 'int',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            'Freq': 'int',
                            }

class OpenBrkChanGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'RunFlag', 'Freq', 'AllocW', 'Lookbacks', 'Ratios', 'MaWidth', 'TrendFactor', 'PriceMode', 'CloseTday', 'IsDisabled']
        self.status_fields = ['TdayOpen', 'VolRatio', 'TradeUnit', 'CurrPos', 'CurrPrices', 'BuyTrig', 'SellTrig',
                              'SplitMode', 'CurRng', 'ChanHigh', 'ChanLow', 'MaLevel']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.field_types = {'RunFlag': 'int',
                            'TradeUnit': 'int',
                            'CurrPos': 'int',
                            'Lookbacks': 'int',
                            'Ratios': 'float',
                            'VolRatio': 'floatlist',
                            'PriceMode': 'str',
                            'SplitMode': 'str',
                            'CloseTday': 'bool',
                            'IsDisabled': 'bool',
                            'TdayOpen': 'float',
                            'CurrPrices': 'float',
                            'CurRng': 'float',
                            'MaLevel': 'float',
                            'ChanHigh': 'float',
                            'ChanLow': 'float',
                            'BuyTrig': 'float',
                            'SellTrig': 'float',
                            'Channels': 'int',
                            'MinRng': 'float',
                            'MaChan': 'int',
                            'MaWidth': 'float',
                            'TrendFactor': 'float',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            'Freq': 'int',
                            }

class ChanBreakGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'RunFlag', 'AllocW', 'EntryChan', 'ExitChan', 'Stoploss', 'CloseTday', 'IsDisabled']
        self.status_fields = ['TradeUnit', 'CurrPos', 'CurrPrices', 'LongEntry', 'LongExit', 'ShortEntry', 'ShortExit', 'Atr', 'SplitMode']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.field_types = {'RunFlag':'int',
                            'TradeUnit':'int',
                            'CurrPos': 'int',
                            'Stoploss': 'float',
                            'CloseTday': 'bool',
                            'IsDisabled': 'bool',
                            'SplitMode': 'str',
                            'CurrPrices': 'float',
                            'LongEntry': 'float',
                            'LongExit': 'float',
                            'ShortEntry': 'float',
                            'ShortExit': 'float',
                            'Atr': 'float',
                            'EntryChan': 'int',
                            'ExitChan': 'int',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            }

class BBandPChanStratGui(StratGui):   
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'RunFlag', 'AllocW',  'Ratios', 'CloseTday', 'IsDisabled']
        self.status_fields = ['TradeUnit', 'Freq', 'CurrPos', 'CurrPrices', 'BandWin', 'UpperBand', 'MidBand', 'LowerBand', 'Channels', 'ChanHigh', 'ChanLow']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.field_types = {'RunFlag':'int',
                            'TradeUnit':'int',
                            'CurrPos': 'int',
                            'IsDisabled': 'bool',
                            'Ratios': 'float',
                            'CloseTday': 'bool',
                            'CurrPrices': 'float',
                            'MidBand': 'float',
                            'UpperBand': 'float',
                            'LowerBand': 'float',
                            'BandWin': 'int',
                            'ChanHigh': 'float',
                            'ChanLow': 'float',
                            'Channels': 'int',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            'Freq': 'int',
                            }

class MASystemStratGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'RunFlag', 'AllocW', 'CloseTday', 'IsDisabled']
        self.status_fields = ['TradeUnit', 'Freq', 'CurrPos', 'CurrPrices', 'MaWin', 'MaFast', 'MaMedm', 'MaSlow', 'Channels', 'ChanHigh', 'ChanLow']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.field_types = {'RunFlag':'int',
                            'TradeUnit':'int',
                            'CurrPos': 'int',
                            'Ratios': 'float',
                            'CloseTday': 'bool',
                            'IsDisabled': 'bool', 
                            'CurrPrices': 'float',
                            'MaWin': 'intlist',
                            'MaFast': 'float',
                            'MaMedm': 'float',
                            'MaSlow': 'float',
                            'ChanHigh': 'float',
                            'ChanLow': 'float',
                            'Channels': 'int',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            'Freq': 'int',
                            }

class RsiAtrStratGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.entry_fields = ['PosScaler', 'RunFlag', 'AllocW', 'CloseTday', 'IsDisabled', 'Stoploss']
        self.status_fields = ['TradeUnit', 'Freq', 'CurrPos', 'CurrPrices', 'RsiTh', 'Rsi', 'Atr', 'Atrma', 'AtrWin', 'AtrmaWin', 'RsiWin']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.field_types = {'RunFlag':'int',
                            'TradeUnit':'int',
                            'CurrPos': 'int',
                            'CloseTday': 'bool',
                            'IsDisabled': 'bool',
                            'CurrPrices': 'float',
                            'RsiWin': 'int',
                            'RsiTh': 'float',
                            'Rsi': 'float',
                            'Atr': 'float',
                            'Atrma': 'float',
                            'AtrWin': 'int',
                            'AtrmaWin': 'int',
                            'Stoploss': 'float',
                            'AllocW': 'float',
                            'PosScaler': 'float',
                            'Freq': 'int',
                            }

class RBStratGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.root = master
        self.entry_fields = ['PosScaler', 'RunFlag', 'EntryLimit', 'DailyCloseBuffer', 'AllocW', 'MinRng', 'TrailLoss', 'Ratios', 'StartMinId', 'IsDisabled']
        self.status_fields = ['TradeUnit', 'CurrPos', 'CurrPrices', 'Sbreak', 'Bsetup', 'Benter', 'Senter', 'Ssetup', 'Bbreak']
        self.shared_fields = ['PosScaler', 'EntryLimit', 'DailyCloseBuffer', 'IsDisabled']
        self.field_types = {'RunFlag':'int',
                            'AllocW': 'float',
                            'IsDisabled': 'bool',
                            'PosScaler': 'float',
                            'TradeUnit':'int',
                            'CurrPos': 'int',
                            'MinRng':'float', 
                            'Ratios': 'floatlist', 
                            'StartMinId': 'int',
                            'CurrPrices': 'float',
                            'Sbreak': 'float', 
                            'Bbreak':'float',
                            'Bsetup':'float', 
                            'Benter':'float', 
                            'Senter':'float', 
                            'Ssetup':'float',
                            'TrailLoss':'float',
                            'EntryLimit': 'int',
                            'DailyCloseBuffer': 'int'}

class TLStratGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.root = master
        self.entry_fields = ['PosScaler', 'RunFlag', 'AllocW', 'Channels', 'MaxPos', 'TrailLoss', 'IsDisabled']
        self.status_fields = ['TradingFreq', 'TradeUnit', 'CurrPos', 'CurrPrices', 'CurrAtr', 'EntryHigh', 'EntryLow', 'ExitHigh', 'ExitLow']
        self.shared_fields = ['PosScaler', 'IsDisabled']
        self.field_types = {'RunFlag':'int',        
                            'TradeUnit':'int',
                            'CurrPos': 'int',
                            'IsDisabled': 'bool',
                            'TradingFreq': 'str',
                            'TrailLoss': 'float',
                            'MaxPos': 'int',
                            'Channels': 'intlist',
                            'CurrPrices': 'float',
                            'CurrAtr':  'float',
                            'EntryHigh':'float',
                            'EntryLow': 'float',
                            'ExitHigh': 'float',
                            'ExitLow':  'float',
                            'AllocW': 'float',
                            'PosScaler': 'float'}

class OptionArbStratGui(StratGui):
    def __init__(self, strat, app, master):
        StratGui.__init__(self, strat, app, master)
        self.root = master
        self.entry_fields = ['RunFlag', 'ProfitRatio', 'ExitRatio', 'IsDisabled']
        self.status_fields = ['TradeUnit', 'BidPrices', 'AskPrices', 'DaysToExpiry', 'TradeMargin'] 
        self.shared_fields = ['ProfitRatio', 'ExitRatio', 'IsDisabled']
        self.field_types = {'RunFlag':'int',        
                           'TradeUnit':'int',
                           'IsDisabled': 'bool',
                            'BidPrices': 'float',
                            'AskPrices': 'float',
                            'DaysToExpiry':'int',
                            'TradeMargin':'floatlist',
                            'ProfitRatio': 'float',
                            'ExitRatio': 'float'}
