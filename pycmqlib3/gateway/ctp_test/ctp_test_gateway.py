from pycmqlib3.core.agent_pseudo import PseudoAgent
from .vnctp_gateway import *

class TestVnctpGateway(VnctpGateway):
    def __init__(self, agent, gateway_name='CTP'):
        super(TestVnctpGateway, self).__init__(agent, gateway_name)
        self.reqID = 1
        self.order_ref = 0

    def register_event_handler(self):
        self.event_engine.register(EVENT_MARKETDATA+self.gateway_name, self.rsp_market_data)
        self.event_engine.register(EVENT_QRYACCOUNT+self.gateway_name, self.rsp_qry_account)
        self.event_engine.register(EVENT_QRYPOSITION+self.gateway_name, self.rsp_qry_position)
        self.event_engine.register(EVENT_QRYTRADE+self.gateway_name, self.rsp_qry_order)
        self.event_engine.register(EVENT_QRYORDER+self.gateway_name, self.rsp_qry_order)
        self.event_engine.register(EVENT_QRYINVESTOR+self.gateway_name, self.rsp_qry_investor)
        self.event_engine.register(EVENT_QRYINSTRUMENT+self.gateway_name, self.rsp_qry_instrument)
        self.event_engine.register(EVENT_ERRORDERINSERT+self.gateway_name, self.err_order_insert)
        self.event_engine.register(EVENT_ERRORDERCANCEL+self.gateway_name, self.err_order_action)
        self.event_engine.register(EVENT_RTNTRADE+self.gateway_name, self.rtn_trade)
        self.event_engine.register(EVENT_RTNORDER+self.gateway_name, self.rtn_order)

    def subscribe(self, instID):
        self.mdApi.subscribeMarketData(instID)

    def send_order(self, iorder):
        """发单"""
        self.tdApi.reqID += 1
        self.tdApi.order_ref = max(self.tdApi.order_ref, int(iorder.local_id))
        req = {}
        req['ExchangeID'] = iorder.exchange
        req['InstrumentID'] = iorder.instrument
        req['LimitPrice'] = iorder.limit_price
        req['VolumeTotalOriginal'] = int(iorder.volume)
        req['Direction'] = DIRECTION_CMQ2CTP.get(iorder.direction, "")
        req['CombOffsetFlag'] = OFFSET_CMQ2CTP.get(iorder.action_type, "")
        req['OrderPriceType'] = ORDERTYPE_CMQ2CTP.get(iorder.price_type, "")
        req['TimeCondition'] = THOST_FTDC_TC_GFD
        req['VolumeCondition'] = THOST_FTDC_VC_AV
        if iorder.price_type == OrderType.FAK:
            req['OrderPriceType'] = THOST_FTDC_OPT_LimitPrice
            req['TimeCondition'] = THOST_FTDC_TC_IOC
            req['VolumeCondition'] = THOST_FTDC_VC_AV
        elif iorder.price_type == OrderType.FOK:
            req['OrderPriceType'] = THOST_FTDC_OPT_LimitPrice
            req['TimeCondition'] = THOST_FTDC_TC_IOC
            req['VolumeCondition'] = THOST_FTDC_VC_CV
        req['OrderRef'] = iorder.local_id
        req['InvestorID'] = self.tdApi.userID
        req['UserID'] = self.tdApi.userID
        req['BrokerID'] = self.tdApi.brokerID
        req['CombHedgeFlag'] = THOST_FTDC_HF_Speculation                        # 投机单
        req['ContingentCondition'] = THOST_FTDC_CC_Immediately                  # 立即发单
        req['ForceCloseReason'] = THOST_FTDC_FCC_NotForceClose                  # 非强平
        req['IsAutoSuspend'] = 0                                                # 非自动挂起
        req['MinVolume'] = 1                                                    # 最小成交量为1
        self.tdApi.reqOrderInsert(req, self.tdApi.reqID)

    def cancelOrder(self, iorder):
        """撤单"""
        self.tdApi.reqID += 1
        req = {}
        req['InstrumentID'] = str(iorder.instrument)
        req['ExchangeID'] = str(iorder.exchange)
        req['ActionFlag'] = THOST_FTDC_AF_Delete
        req['BrokerID'] = self.tdApi.brokerID
        req['InvestorID'] = self.tdApi.userID
        req['OrderRef'] = iorder.local_id
        req['FrontID'] = int(self.tdApi.frontID)
        req['SessionID'] = int(self.tdApi.sessionID)
        self.tdApi.reqOrderAction(req, self.tdApi.reqID)

    def rtn_order(self, event):
        data = event.dict['data']
        print(data)

    def rtn_trade(self, event):
        data = event.dict['data']
        print(data)

    def rsp_market_data(self, event):
        data = event.dict['data']
        print(data)

    def rsp_qry_account(self, event):
        data = event.dict['data']
        print(data)

    def rsp_qry_instrument(self, event):
        data = event.dict['data']
        print(data)

    def rsp_qry_investor(self, event):
        data = event.dict['data']
        print(data)

    def rsp_qry_position(self, event):
        data = event.dict['data']
        print(data)

    def rsp_qry_order(self, event):
        data = event.dict['data']
        print(data)

    def err_order_insert(self, event):
        data = event.dict['data']
        print(data)

    def err_order_action(self, event):
        data = event.dict['data']
        print(data)

def create_test_gway(folder = "gway_test", gateway_name = "SIMNOW1CTP"):
    config = {'folder': "C:\\dev\\pycmqlib3\\" + folder + "\\"}
    tday = datetime.date.today()
    agent = PseudoAgent(config, tday)
    gway = TestVnctpGateway(agent, gateway_name)
    gway.register_event_handler()
    agent.event_engine.start()
    return gway

