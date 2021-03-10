#-*- coding:utf-8 -*-
import datetime
from .agent import Agent
from .event_type import EVENT_MAIL
from .event_engine import Event
from pycmqlib3.utility.email_tool import send_email_by_outlook

class MAQSAgent(Agent):
    def __init__(self, config = {}, tday=datetime.date.today()):
        super(MAQSAgent, self).__init__(config, tday)
        self.report_email_addr = config.get("report_email", "")
        print("report email is set to %s" % self.report_email_addr)
        self.trade_email_addr = config.get("trade_email", {'default': self.report_email_addr})
        if 'default' not in self.trade_email_addr:
            print("no default email address is set")
        print("trade email is set to %s" % self.trade_email_addr)
        self.register_event_handler()

    def register_event_handler(self):
        super(MAQSAgent, self).register_event_handler()
        self.event_engine.register(EVENT_MAIL, self.email_distributor)

    def email_distributor(self, event):
        sender = event.dict['sender']
        msg = event.dict['body']
        recepient = event.dict['recepient']
        attach = event.dict.get('attach_files', [])
        subject = sender + ":" + event.dict['subject']
        send_email_by_outlook(recepient, subject, msg, attach)

    def restart(self):
        super(MAQSAgent, self).restart()
        eod_time = datetime.datetime.combine(self.scur_day, datetime.time(15, 16, 0))
        if not self.eod_flag:
            if datetime.datetime.now() < eod_time:
                self.put_command(eod_time, self.run_eod)

    def run_eod(self):
        super(MAQSAgent, self).run_eod()
        attach_files = []
        for name in self.gateways:
            filename = self.gateways[name].file_prefix + 'PNL_attribution_' + self.scur_day.strftime('%y%m%d') + '.csv'
            attach_files.append(filename)
            filename = self.gateways[name].file_prefix + 'EODPos_' + self.scur_day.strftime('%y%m%d') + '.csv'
            attach_files.append(filename)
        filename = self.folder + 'trade_' + self.scur_day.strftime('%y%m%d') + '.csv'
        attach_files.append(filename)
        event = Event(type=EVENT_MAIL)
        event.dict['sender'] = self.name
        event.dict['body'] = ""
        event.dict['recepient'] = self.report_email_addr
        event.dict['subject'] = "EOD position and PNL"
        event.dict['attach_files'] = attach_files
        self.event_engine.put(event)

    def submit_trade(self, xtrade):
        super(MAQSAgent, self).submit_trade(xtrade)
        insts = xtrade.instIDs
        inst_key = '_'.join(insts)
        if (inst_key in self.trade_email_addr) or ('default' in self.trade_email_addr):
            vol = xtrade.vol
            tick_size = self.instruments[insts[0]].tick_base
            direction = 1 if vol > 0 else -1
            limit_price = int(xtrade.limit_price/tick_size) * tick_size
            if len(insts) > 1:
                order_type = "Limit order"
            else:
                order_type = 'Market order'
            event = Event(type=EVENT_MAIL)
            event.dict['sender'] = self.name
            event.dict['recepient'] = self.trade_email_addr.get(inst_key, self.trade_email_addr['default'])
            event.dict['subject'] = "%s on %s, trade_id = %s" % (order_type, inst_key, xtrade.id)
            event.dict['body'] = "strategy = %s, trade_id =%s, direction = %s, contract = %s, volume = %s lots at price = %s, %s" % (\
                xtrade.strategy, xtrade.id, "BUY" if direction> 0 else "SELL", inst_key, str(abs(vol)), limit_price, order_type)
            self.event_engine.put(event)