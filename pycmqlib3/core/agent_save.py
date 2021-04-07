import datetime
from pycmqlib3.utility.misc import day_shift
from pycmqlib3.utility import dbaccess
from . agent import Agent
from . trading_const import ProductType
from . event_type import EVENT_MIN_BAR, EVENT_TICK, EVENT_MKTDATA_EOD, EVENT_DB_WRITE, \
    EVENT_LOG, EVENT_TIMER, EVENT_DAYSWITCH

class SaveAgent(Agent):
    def __init__(self, config = {}, tday = datetime.date.today()):
        super(SaveAgent, self).__init__(config, tday)
        self.lookback = 1
        self.save_tick = config.get('save_tick', True)

    def init_init(self):
        self.register_event_handler()

    def restart(self):
        self.prepare_data_env(mid_day=True)
        for gway in self.gateways:
            gateway = self.gateways[gway]
            gateway.connect()
        self.event_engine.start()
        if not self.eod_flag:
            eod_marker = int(self.eod_marker)//100
            eod_time = max(datetime.datetime.combine(self.scur_day, \
                datetime.time(int(self.eod_marker[:2]), int(self.eod_marker[2:4]), int(self.eod_marker[4:6]))), \
                datetime.datetime.now()) + datetime.timedelta(minutes = 1)
            self.put_command(eod_time, self.run_eod)

    def write_mkt_data(self, event):
        inst = event.dict['instID']
        type = event.dict['type']
        data = event.dict['data']
        self.db_conn = dbaccess.connect(**dbaccess.dbconfig)
        if type == EVENT_MIN_BAR:
            dbaccess.insert_min_data(self.db_conn, inst, data, dbtable = self.min_db_table)
        elif type == EVENT_TICK:
            if self.save_tick:
                dbaccess.bulkinsert_tick_data(self.db_conn, inst, data, dbtable = self.tick_db_table)
        elif type == EVENT_MKTDATA_EOD:
            dbaccess.insert_daily_data(self.db_conn, inst, data, dbtable = self.daily_db_table)
        else:
            pass
        self.db_conn.close()

    def save_data(self):
        self.db_conn = dbaccess.connect(**dbaccess.dbconfig)
        ldate = day_shift(self.scur_day, '-%sb' % self.lookback)
        for inst in self.instruments:
            inst_obj = self.instruments[inst]
            if inst_obj.ptype == ProductType.Future:
                ra_m = self.bar_factory[inst]['m1'].bar_array
                if len(ra_m.data)>0:
                    dbaccess.bulkinsert_min_data(self.db_conn, inst, inst_obj.exchange, \
                                        ra_m.data, ldate, dbtable='fut_min', is_replace=False)
                ra_d = self.day_data[inst]
                if len(ra_d.data)>0:
                    dbaccess.bulkinsert_daily_data(self.db_conn, inst, inst_obj.exchange, \
                                        ra_d.data, ldate, dbtable='fut_daily', is_replace=False)

    def register_event_handler(self):
        for key in self.gateways:
            gateway = self.gateways[key]
            gateway.register_event_handler()
        self.event_engine.register(EVENT_DB_WRITE, self.write_mkt_data)
        self.event_engine.register(EVENT_LOG, self.log_handler)
        self.event_engine.register(EVENT_TICK, self.run_tick)
        self.event_engine.register(EVENT_DAYSWITCH, self.day_switch)
        self.event_engine.register(EVENT_TIMER, self.check_commands)
        #if 'CTP' in self.type2gateway:
        #   self.event_engine.register(EVENT_TDLOGIN + self.type2gateway['CTP'].gatewayName, self.ctp_qry_instruments)
        #   self.event_engine.register(EVENT_QRYINSTRUMENT + self.type2gateway['CTP'].gatewayName, self.add_ctp_instruments)
        #   self.type2gateway['CTP'].setAutoDbUpdated(True)

    #ef ctp_qry_instruments(self, event):
    #   dtime = datetime.datetime.now()
    #   min_id = get_min_id(dtime)
    #   if min_id < 250:
    #       gateway = self.type2gateway['CTP']
    #       gateway.qry_commands.append(gateway.tdApi.qryInstrument)

    #ef add_ctp_instruments(self, event):
    #   data = event.dict['data']
    #   last = event.dict['last']
    #   if last:
    #       gateway = self.type2gateway['CTP']
    #       for symbol in gateway.qry_instruments:
    #           if symbol not in self.instruments:
    #               self.add_instrument(symbol)

    def exit(self):
        for inst in self.instruments:
            self.min_switch(inst, False)
        self.event_engine.stop()
        for name in self.gateways:
            gateway = self.gateways[name]
            gateway.close()
            gateway.mdApi = None
            gateway.tdApi = None
