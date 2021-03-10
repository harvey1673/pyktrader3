# -*- coding: utf-8 -*-
import socket
import threading
import json
import misc
from trading_const import *
from trading_object import TickData
from gateway import *

# 方向类型映射
DIRECTION_CMQ2TX = {
    Direction.LONG: 'Buy',
    Direction.SHORT: 'Sell'
}

DIRECTION_TX2VT = {v: k for k, v in DIRECTION_CMQ2TX.items()}

ORDERTYPE_CMQ2TX = {
    OrderType.LIMIT: 'Limit',
    OrderType.MARKET: 'Market',
    OrderType.STOP: 'Stop'
}

ORDERTYPE_TX2VT = {v: k for k, v in ORDERTYPE_CMQ2TX.items()}

def instID_to_contract(instID, exch):
    cont_mth = str(misc.inst2contmth(instID))
    cont_data = [exch, misc.inst2product(instID), cont_mth]
    contract = "\\".join(cont_data)
    return contract

def contract_to_instID(cont):
    cont_str = cont.split('\\')
    exchange = str(cont_str[0])
    product = str(cont_str[1])
    if exchange == 'CZCE':
        instID = product + str(cont_str[2])[-3:]
    else:
        instID = product + str(cont_str[2])[-4:]
    return (instID, product, exchange)

class SocketService(object):
    def __init__(self, gateway, address = '127.0.0.1', port = 55242, service_name = 'md'):
        self.gateway = gateway
        self.gateway_name = gateway.gateway_name
        self.address = address
        self.port = port
        self.socket = None
        self.connectionStatus = False
        self.run_flag = False
        self.thread = threading.Thread(target = self.run)
        self.service_name = service_name

    def send_msg(self, cmd, data):
        msg = "#" + cmd
        for key in data:
            msg += "\x01" + key + "=" + str(data[key])
        nlen = len(msg)
        msg = '#%010d' % nlen + msg + "\n"
        self.socket.sendall(msg.encode('ascii'))

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.address, self.port))
            self.connectionStatus = True
            data = {}
            data['type'] = self.service_name
            event = Event(type=EVENT_TRADEX)
            event.dict['data'] = data
            event.dict['func'] = 'onConnected'
            event.dict['gateway'] = self.gateway_name
            self.gateway.event_engine.put(event)
        except:
            self.connectionStatus = False
        if not self.run_flag:
            self.run_flag = True
            self.start()
        return self.connectionStatus

    def start(self):
        self.thread.start()

    def run(self):
        while self.run_flag:
            if self.connectionStatus:
                data = self.socket.recv(11)
                data_len = int(data.decode('ascii')[1:]) + 1
                msg = self.socket.recv(data_len)
                cmd_list = msg.decode('utf-8')[:-1].split("\x01")
                func = 'on' + str(cmd_list[0][1:])
                if func == 'onCargillBookStrategyCollection':
                    data = []
                    for i in range(1, len(cmd_list)):
                        keys = cmd_list[i].split('=')
                        data.append(tuple(keys))
                else:
                    data = {}
                    for i in range(1, len(cmd_list)):
                        keys = cmd_list[i].split('=')
                        data[str(keys[0])] = keys[1]
                event = Event(type=EVENT_TRADEX)
                event.dict['data'] = data
                event.dict['func'] = func
                event.dict['gateway'] = self.gateway_name
                self.gateway.event_engine.put(event)
            else:
                event = Event(type=EVENT_TRADEX)
                event.dict['data'] = {}
                event.dict['func'] = 'connect'
                event.dict['gateway'] = self.gateway_name
                self.gateway.event_engine.put(event)

    def close(self):
        self.run_flag = False
        self.connectionStatus = False
        self.thread.join()
        self.socket = None
        self.thread = None

class MdApi(SocketService):
    def __init__(self, gateway, address = '127.0.0.1', port = 55242, service_name = 'md'):
        """Constructor"""
        super(MdApi, self).__init__(gateway, address, port, service_name)

    def subscribe(self, contract):
        if self.connectionStatus:
            data = {'Contract': contract}
            self.send_msg("Subscribe", data)

    def unsubscribe(self, contract):
        if self.connectionStatus:
            data = {'Contract': contract}
            self.send_msg("Unsubscribe", data)

    def keep_alive(self):
        self.send_msg("KeepAlive", {})

class TdApi(SocketService):
    def __init__(self, gateway, address='127.0.0.1', port=55241, service_name = 'td'):
        """Constructor"""
        super(TdApi, self).__init__(gateway, address, port, service_name)

    def send_order(self, iorder):
        req = {}
        req['ClientOrderID'] = iorder.local_id
        req['Contract'] = instID_to_contract(iorder.instrument, iorder.exchange)
        product = req['Contract'].split('\\')[1]
        if product in self.gateway.account_map:
            req['Account'] = self.gateway.account_map[product]
        try:
            req['BuySell'] = DIRECTION_CMQ2TX[iorder.direction]
        except:
            log_content = "unsupported direction = %s for order_ref = %s" % (iorder.direction, iorder.order_ref)
            self.on_log(log_content, level=logging.WARNING)
            return
        req['Quantity'] = iorder.volume
        req['Price1'] = iorder.limit_price
        try:
            req['OrderType'] = ORDERTYPE_CMQ2TX.get(iorder.price_type, "")
        except:
            log_content = "unsupported price type = %s for TradeX API for order_ref = %s" % (iorder.price_type, iorder.order_ref)
            self.on_log(log_content, level=logging.WARNING)
            return
        if len(self.gateway.strategy_tag) > 0:
            fields = ['C-Strategy', self.gateway.strategy_tag]
            req['Options'] = "\x02".join(fields)
        self.send_msg("SendOrder", req)

    def cancel_order(self, iorder):
        req = {}
        req['ClientOrderID'] = iorder.local_id
        self.send_msg("CancelOrder", req)

    def amend_order(self, iorder, order_spec):
        req = {}
        for key in order_spec:
            setattr(iorder, key, order_spec[key])
            if key == 'volume':
                req['Quantity'] = order_spec[key]
            elif key == 'limit_price':
                req['Price1'] = order_spec[key]
        req['ClientOrderID'] = iorder.local_id
        self.send_msg("AmendOrder", req)

    def keep_alive(self):
        self.send_msg("KeepAlive", {})

class PaperTdApi(object):
    def __init__(self, gateway, address='127.0.0.1', port=55241, service_name = 'td'):
        """Constructor"""
        self.gateway = gateway
        self.gateway_name = gateway.gateway_name
        self.connectionStatus = True
        self.run_flag = True
        self.service_name = service_name

    def connect(self):
        return self.connectionStatus

    def close(self):
        pass

    def send_order(self, iorder):
        data = {}
        data['ClientOrderID'] = iorder.local_id
        data['Price'] = iorder.limit_price
        data['Quantity'] = iorder.volume
        data['FillID'] = iorder.local_id
        func = 'onFillReceived'
        event = Event(type=EVENT_TRADEX)
        event.dict['data'] = data
        event.dict['func'] = func
        event.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event)

    def cancel_order(self, iorder):
        data = {}
        data['ClientOrderID'] = iorder.local_id
        data['Price'] = iorder.limit_price
        data['Quantity'] = iorder.volume
        data['FillID'] = iorder.local_id
        data['Status'] = 'Cancelled'
        data['ServerOrderID'] = iorder.local_id
        data['AveragePrice'] = iorder.limit_price
        data['AmountFilled'] = 0
        func = 'onOrderUpdate'
        event = Event(type=EVENT_TRADEX)
        event.dict['data'] = data
        event.dict['func'] = func
        event.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event)

    def keep_alive(self):
        pass

class TradeXGateway(GrossGateway):
    """TradeX接口"""
    def __init__(self, agent, gateway_name='TradeX', config_file = "config.json"):
        """Constructor"""
        super(TradeXGateway, self).__init__(agent, gateway_name)
        file_name = self.file_prefix + config_file
        with open(file_name, 'r') as f:
            setting = json.load(f)
        md_api = setting.get('md_api', 'tradex.tradex_gateway.MdApi')
        td_api = setting.get('td_api', 'tradex.tradex_gateway.TdApi')
        addr = str(setting.get('address', '127.0.0.1'))
        md_port = int(setting.get('md_port', '55242'))
        td_port = int(setting.get('td_port', '55241'))
        self.mdApi = get_obj_by_name(md_api)(self, addr, md_port)
        self.tdApi = get_obj_by_name(td_api)(self, addr, td_port)
        self.account_map = setting.get('account_map', {})
        self.system_orders = []
        self.update_account = setting.get('update_account', False)
        self.trade_link_status = {}
        self.price_link_status = {}
        self.md_data_buffer = 0
        self.trading_day = datetime.date.today().strftime("%Y%m%d")
        self.trading_hour_index = 0
        self.trading_enabled = False
        self.order_prefix = setting.get("order_prefix", "MQ")
        self.strategy_tag =  setting.get("strategy_tag", "")
        self.subscribed_contracts = []
        self.order_error_timers = {}
        self.book_strat_list = []

    def day_finalize(self, tday):
        super(TradeXGateway, self).day_finalize(tday)
        for instID in self.order_error_timers:
            self.order_error_timers[instID] = [0, 10]

    def add_instrument(self, instID):
        super(TradeXGateway, self).add_instrument(instID)
        if instID not in self.order_error_timers:
            self.order_error_timers[instID] = [0, 10]

    def check_order_permission(self, instID, direction):
        if (self.agent.tick_id < self.order_error_timers[instID][0]):
            return False
        else:
            status = super(TradeXGateway, self).check_order_permission(instID, direction)
            self.order_error_timers[instID][1] = 10
            return status

    def set_trading_day(self):
        dt = datetime.datetime.today()
        curr_hour = (dt.hour + 6) % 24
        if (self.trading_hour_index < curr_hour) and curr_hour == 2:
            tday = dt.date()
            next_trade_day = misc.day_shift(tday, '1b', misc.CHN_Holidays)
            trading_day = next_trade_day.strftime("%Y%m%d")
            if trading_day != self.trading_day:
                self.trading_day = trading_day
                event = Event(type=EVENT_DAYSWITCH)
                event.dict['log'] = '换日: %s -> %s' % (self.agent.scur_day, next_trade_day)
                event.dict['date'] = next_trade_day
                self.event_engine.put(event)

    def set_order_id(self, iorder):
        iorder.local_id = self.order_prefix + str(iorder.order_ref)

    def get_pos_class(self, inst):
        return (position.GrossPosition, {})

    def register_event_handler(self):
        self.event_engine.register(EVENT_TRADEX, self.msg_handler)

    def msg_handler(self, event):
        data = event.dict['data']
        run_func = getattr(self, event.dict['func'])
        run_func(data)

    def connect(self, data = {}):
        if not self.mdApi.connectionStatus:
            self.mdApi.connect()
        if not self.tdApi.connectionStatus:
            self.tdApi.connect()

    def onErrorMessage(self, data):
        self.on_log("Error message: %s" % data, level=logging.WARNING)

    def onConnected(self, data):
        if data['type'] == 'md':
            for contract in self.subscribed_contracts:
                self.mdApi.subscribe(contract)
            self.on_log('TradeX MarketData is connected', level=logging.INFO)
        elif data['type'] == 'td':
            self.on_log('TradeX TradeData is connected', level=logging.INFO)

    def onTradeLinkStatus(self, data):
        link_id = str(data['UpstreamID'])
        if data['Status'] == 'Up':
            self.trade_link_status[link_id] = True
        else:
            self.trade_link_status[link_id] = False
        self.on_log("TradeX TradeLink Upstream LinkID=%s status = %s" % (link_id, self.trade_link_status[link_id]), level=logging.INFO)

    def onPriceLinkStatus(self, data):
        link_id = str(data['UpstreamID'])
        if data['Status'] == 'Up':
            self.price_link_status[link_id] = True
        else:
            self.price_link_status[link_id] = False
        self.on_log("TradeX PriceLink Upstream LinkID=%s status = %s" % (link_id, self.price_link_status[link_id]), level = logging.INFO)

    def onPriceUpdate(self, data):
        if data['Contract'] not in self.subscribed_contracts:
            return
        instID, product, exchange = contract_to_instID(data['Contract'])
        timestr = str(self.trading_day) + ' ' + str(data['Timestamp'])
        timestamp = datetime.datetime.strptime(timestr, '%Y%m%d %H:%M:%S')
        hrs = trading_hours(product, exchange)
        bad_tick = True
        tick_id = get_tick_id(timestamp)
        for ptime in hrs:
            if (tick_id >= ptime[0] * 1000 - self.md_data_buffer) and (tick_id < ptime[1] * 1000 + self.md_data_buffer):
                bad_tick = False
                break
        if bad_tick:
            return
        tick = TickData(
            instID = instID,
            exchange = exchange,
            timestamp = timestamp,
            gateway_name = self.gateway_name,
            tick_id = tick_id
        )
        for (tick_key, data_key) in [('open', 'Open'), ('high', 'High'), ('low', "Low")]:
            setattr(tick, tick_key, float(str(data[data_key])))
        tick.volume = int(float(str(data['Total'])))
        for data_key, tick_key in zip(['Bid', 'Offer', 'Last'], \
                       [('bid_price1', 'bid_vol1'), ('ask_price1', 'ask_vol1'), ('price', '')]):
            msg = data[data_key]
            v, p = msg.split(',')
            setattr(tick, tick_key[0], float(p))
            if len(tick_key[1]) > 0:
                setattr(tick, tick_key[1], int(v))
        event1 = Event(type=EVENT_TICK)
        event1.dict['data'] = tick
        self.event_engine.put(event1)

        event2 = Event(type=EVENT_TICK + tick.instID)
        event2.dict['data'] = tick
        self.event_engine.put(event2)

    def onAccountUpdate(self, data):
        self.set_trading_day()
        if self.update_account:
            self.account_info['curr_capital'] = data['Balance']
            self.account_info['locked_margin'] = data['Margin']
            self.account_info['used_margin'] = data['Margin']
            self.account_info['available'] = data['Cash']
            self.account_info['yday_pnl'] = data['UnPL']
            self.account_info['tday_pnl'] = data['ClosePL']
            self.account_info['pnl_total'] = data['ClosePL'] + data['UnPL']

    def onPositionUpdate(self, data):
        cont = data['Contract']
        if cont not in self.subscribed_contracts:
            #logContent = "unexpected position for contract = %s" % cont
            #self.on_log(logContent, level=logging.INFO)
            return
        instID, product, exch = contract_to_instID(cont)
        #account = data['Account']
        #pos_info = {'instID': instID, 'product': product, 'exchange': exch, 'account': account, }
        self.qry_pos[instID] = {'tday': [int(data['Buys']), int(data['Sells'])], \
                                'yday': [int(data['PrevBuys']), int(data['PrevSells'])]}
        #pos_info['avg_price'] = [float(data['BuyAveragePrice']), float(data['BuyAveragePrice'])]

    def onOrderUpdate(self, data):
        local_id = str(data['ClientOrderID']).strip()
        sys_id = str(data['ServerOrderID']).strip()
        if (local_id not in self.id2order):
            #logContent = 'receive order update from other agents, Contract=%s, ClientOrderID=%s, ServerOrderID=%s' % \
            #             (data['Contract'], local_id, data['ServerOrderID'])
            #self.on_log(logContent, level=logging.WARNING)
            return
        myorder = self.id2order[local_id]
        myorder.sys_id = sys_id
        if data['Status'] == 'Cancelled':
            myorder.on_cancel()
            status = True
        elif data['Status'] == 'Rejected':
            instID = myorder.instrument
            myorder.on_cancel()
            status = True
            self.order_error_timers[instID][0] = min(self.agent.tick_id + self.order_error_timers[instID][1], 2059570)
            if (self.agent.tick_id < self.order_error_timers[instID][0]):
                self.order_error_timers[instID][1] = min(2 * self.order_error_timers[instID][0], 600)
            else:
                self.order_error_timers[instID][1] = 10
        else:
            status = myorder.on_order(sys_id = str(data['ServerOrderID']).strip(), \
                                      price=float(data['AveragePrice']), \
                                      volume=int(data['AmountFilled']))
        if status:
            event = Event(type=EVENT_ETRADEUPDATE)
            event.dict['trade_ref'] = myorder.trade_ref
            self.event_engine.put(event)

    def onFillReceived(self, data):
        if not self.trading_enabled:
            return
        local_id = str(data['ClientOrderID']).strip()
        if local_id in self.id2order:
            myorder = self.id2order[local_id]
            myorder.on_trade(price = float(data['Price']), volume = int(data['Quantity']), trade_id = str(data['FillID']))
            event = Event(type=EVENT_ETRADEUPDATE)
            event.dict['trade_ref'] = myorder.trade_ref
            self.event_engine.put(event)
        else:
            #logContent = 'receive fill update from other agents, Contract=%s, OrderRef=%s' % (data['Contract'], local_id)
            #self.on_log(logContent, level = logging.WARNING)
            pass

    def onSnapshotDone(self, data):
        self.trading_enabled = True
        self.on_log("TradeX snapshot replay is done.", level = logging.INFO)

    def onCargillBookStrategyCollection(self, data):
        n = len(data)//3
        self.book_strat_list = []
        for i in range(len(n)):
            self.book_strat_list.append((data[i*3][1], data[i*3+1][1]))
        print(self.book_strat_list)

    def send_order(self, iorder):
        if not self.trading_enabled:
            return
        self.tdApi.send_order(iorder)
        iorder.status = OrderStatus.Sent
        super(TradeXGateway, self).send_order(iorder)

    def cancel_order(self, iorder):
        if not self.trading_enabled:
            return
        self.tdApi.cancel_order(iorder)
        super(TradeXGateway, self).cancel_order(iorder)

    def onACKKeepAlive(self, data):
        self.on_log("TradeX Ack KeepAlive.")

    def subscribe(self, subscribeReq):
        self.add_instrument(subscribeReq.symbol)
        contract = instID_to_contract(subscribeReq.symbol, subscribeReq.exchange)
        if contract not in self.subscribed_contracts:
            self.subscribed_contracts.append(contract)
        self.mdApi.subscribe(contract)

    def close(self):
        self.mdApi.close()
        self.tdApi.close()

if __name__=="__main__":
    pass