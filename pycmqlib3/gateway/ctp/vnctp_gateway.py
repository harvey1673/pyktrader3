# -*- coding: utf-8 -*-
import os
import time
import logging
import datetime
from pycmqlib3.core.event_engine import Event
from pycmqlib3.core.trading_const import OrderType
from pycmqlib3.core.event_type import EVENT_DAYSWITCH, EVENT_LOG, EVENT_MARKETDATA, EVENT_QRYACCOUNT, \
    EVENT_QRYPOSITION, EVENT_QRYORDER, EVENT_QRYTRADE, EVENT_QRYINVESTOR, \
    EVENT_QRYINSTRUMENT, EVENT_TIMER, EVENT_TDLOGIN, EVENT_ETRADEUPDATE, \
    EVENT_RTNORDER, EVENT_RTNTRADE, EVENT_ERRORDERINSERT, EVENT_ERRORDERCANCEL
from . vnctpmd import MdApi
from . vnctptd import TdApi
from . ctp_gateway import CtpGateway, DIRECTION_CMQ2CTP, OFFSET_CMQ2CTP, ORDERTYPE_CMQ2CTP
from . ctp_constant import *

class VnctpGateway(CtpGateway):
    def __init__(self, agent, gateway_name='CTP'):
        super(VnctpGateway, self).__init__(agent, gateway_name, \
            md_api = 'pycmqlib3.gateway.ctp.vnctp_gateway.VnctpMdApi', \
            td_api = 'pycmqlib3.gateway.ctp.vnctp_gateway.VnctpTdApi')


class VnctpMdApi(MdApi):
    """CTP行情API实现"""
    def __init__(self, gateway):
        """Constructor"""
        super(VnctpMdApi, self).__init__()        
        self.gateway = gateway                  # gateway对象
        self.gateway_name = gateway.gateway_name  # gateway对象名称
        self.reqID = 0
        self.connect_status = False       # 连接状态
        self.login_status = False            # 登录状态
        self.userID =''
        self.password =''
        self.brokerID = ''
        self.address = ''
        self.trading_day = 20160101
        self.subscribed = set()
    
    def onFrontConnected(self):
        """服务器连接"""
        self.connect_status = True
        logContent = '行情服务器连接成功'
        self.gateway.on_log(logContent, level = logging.INFO)
        self.login()
      
    def onFrontDisconnected(self, n):
        """服务器断开"""
        self.connect_status = False
        self.login_status = False
        logContent = '行情服务器连接断开'
        self.gateway.on_log(logContent, level = logging.INFO)
     
    def onHeartBeatWarning(self, n):
        pass
       
    def onRspError(self, error, n, last):
        """错误回报"""
        self.gateway.write_error("错误回报", error)
    
    def onRspUserLogin(self, data, error, n, last):
        """登陆回报"""
        # 如果登录成功，推送日志信息
        if (error['ErrorID'] == 0) and last:
            self.login_status = True
            logContent = '行情服务器登录完成'
            self.gateway.on_log(logContent, level = logging.INFO)
            # 重新订阅之前订阅的合约
            for instID in self.subscribed:
                self.subscribeMarketData(instID)
            trade_day_str = self.getTradingDay()
            if len(trade_day_str) > 0:
                try:
                    self.trading_day = int(trade_day_str)
                    tradingday = datetime.datetime.strptime(trade_day_str, '%Y%m%d').date()
                    if tradingday > self.gateway.agent.scur_day:
                        event = Event(type=EVENT_DAYSWITCH)
                        event.dict['log'] = '换日: %s -> %s' % (self.gateway.agent.scur_day, self.trading_day)
                        event.dict['date'] = tradingday
                        self.gateway.event_engine.put(event)
                except ValueError:
                    pass
        # 否则，推送错误信息
        else:
            self.gateway.write_error("推送错误信息", error)

    def onRspUserLogout(self, data, error, n, last):
        """登出回报"""
        # 如果登出成功，推送日志信息
        if error['ErrorID'] == 0:
            self.login_status = False
            logContent = '行情服务器登出完成'
            self.gateway.on_log(logContent, level = logging.INFO)
                
        # 否则，推送错误信息
        else:
            self.gateway.write_error("推送错误信息", error)
      
    def onRspSubMarketData(self, data, error, n, last):
        """订阅合约回报"""
        # 通常不在乎订阅错误，选择忽略
        pass

    def onRspUnSubMarketData(self, data, error, n, last):
        """退订合约回报"""
        # 同上
        pass  

    def onRtnDepthMarketData(self, data):
        """行情推送"""
        min_ba = min(data['BidPrice1'], data['AskPrice1'])
        max_ba = max(data['BidPrice1'], data['AskPrice1'])
        if (min_ba > data['UpperLimitPrice']) or (max_ba < data['LowerLimitPrice']) \
                or (data['LastPrice'] > data['UpperLimitPrice']) or (data['LastPrice'] < data['LowerLimitPrice']):
            logContent = 'MD:error in market data for %s LastPrice=%s, BidPrice=%s, AskPrice=%s' % \
                             (data['InstrumentID'], data['LastPrice'], data['BidPrice1'], data['AskPrice1'])
            self.gateway.on_log(logContent, level = logging.DEBUG)
            return
        if (data['BidPrice1'] > data['UpperLimitPrice']) or (data['BidPrice1'] < data['LowerLimitPrice']):
            data['BidPrice1'] = data['AskPrice1']
        elif (data['AskPrice1'] > data['UpperLimitPrice']) or (data['AskPrice1'] < data['LowerLimitPrice']):
            data['AskPrice1'] = data['BidPrice1']
        if (data['BidPrice1'] > data['AskPrice1']):
            logContent = 'MD:error in market data for %s LastPrice=%s, BidPrice=%s, AskPrice=%s' % \
                             (data['InstrumentID'], data['LastPrice'], data['BidPrice1'], data['AskPrice1'])
            self.gateway.on_log(logContent, level = logging.DEBUG)
            return
        event = Event(type = EVENT_MARKETDATA + self.gateway_name)
        event.dict['data'] = data
        event.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event)

    def onRspSubForQuoteRsp(self, data, error, n, last):
        """订阅期权询价"""
        pass
    
    def onRspUnSubForQuoteRsp(self, data, error, n, last):
        """退订期权询价"""
        pass 

    def onRtnForQuoteRsp(self, data):
        """期权询价推送"""
        pass        

    def connect(self, userID, password, brokerID, address):
        """初始化连接"""
        self.userID = userID                # 账号
        self.password = password            # 密码
        self.brokerID = brokerID            # 经纪商代码
        self.address = address              # 服务器地址

        # 如果尚未建立服务器连接，则进行连接
        if not self.connect_status:
            # 创建C++环境中的API对象，这里传入的参数是需要用来保存.con文件的文件夹路径
            path = self.gateway.file_prefix + 'tmp' + os.path.sep
            if not os.path.exists(path):
                os.makedirs(path)

            self.createFtdcMdApi(str(path))
            # 注册服务器地址
            self.registerFront(self.address)
            
            # 初始化连接，成功会调用onFrontConnected
            self.init()
            
        # 若已经连接但尚未登录，则进行登录
        else:
            if not self.login_status:
                self.login()
    
    def subscribe(self, subscribeReq):
        """订阅合约"""
        # 这里的设计是，如果尚未登录就调用了订阅方法
        # 则先保存订阅请求，登录完成后会自动订阅
        symbol = str(subscribeReq.symbol)
        self.subscribed.add(symbol)
        if self.login_status:
            self.subscribeMarketData(symbol)

    def login(self):
        """登录"""
        # 如果填入了用户名密码等，则登录
        if self.userID and self.password and self.brokerID:
            req = {}
            req['UserID'] = self.userID
            req['Password'] = self.password
            req['BrokerID'] = self.brokerID
            self.reqID += 1
            self.reqUserLogin(req, self.reqID)    

    def logout(self):
        if self.userID and self.brokerID:
            req = {}
            req['UserID'] = self.userID
            req['BrokerID'] = self.brokerID
            self.reqID += 1
            self.reqUserLogout(req, self.reqID)

    def close(self):
        """关闭"""
        if self.connect_status:
            self.exit()


class VnctpTdApi(TdApi):
    """CTP交易API实现"""
    def __init__(self, gateway):
        """API对象的初始化函数"""
        super(VnctpTdApi, self).__init__()

        self.gateway = gateway                  # gateway对象
        self.gateway_name = gateway.gateway_name  # gateway对象名称

        self.reqID = 0
        self.order_ref = 0

        self.connect_status = False       # 连接状态
        self.login_status = False            # 登录状态
        self.authStatus = False             # 验证状态
        self.loginFailed = False            # 登录失败（账号密码错误）

        self.userID = ''
        self.password = ''
        self.brokerID = ''
        self.address = ''
        self.frontID = 0
        self.sessionID = 0
        self.requireAuthentication = False

    def onFrontConnected(self):
        """服务器连接"""
        self.connect_status = True
        logContent = '交易服务器连接成功'
        self.gateway.on_log(logContent, level = logging.INFO)

        if self.requireAuthentication:
            self.authenticate()
        else:
            self.login()

    def onFrontDisconnected(self, n):
        """服务器断开"""
        self.connect_status = False
        self.login_status = False

        logContent = '交易服务器连接断开'
        self.gateway.on_log(logContent, level = logging.INFO)

    def onHeartBeatWarning(self, n):
        """"""
        pass

    def onRspAuthenticate(self, data, error, n, last):
        """"""
        if error['ErrorID'] == 0:
            self.authStatus = True
            logContent = '交易服务器认证成功'
            self.gateway.on_log(logContent, level=logging.INFO)
            self.login()
        else:
            self.gateway.write_error("交易服务器认证失败", error)

    def onRspUserLogin(self, data, error, n, last):
        """登陆回报"""
        # 如果登录成功，推送日志信息
        if not error['ErrorID']:
            self.frontID = str(data['FrontID'])
            self.sessionID = str(data['SessionID'])
            self.login_status = True
            logContent = '交易服务器登录完成, frontID=%s, sessionID=%s' % (self.frontID, self.sessionID)
            self.gateway.on_log(logContent, level = logging.INFO)

            # 确认结算信息
            req = {}
            req['BrokerID'] = self.brokerID
            req['InvestorID'] = self.userID
            self.reqID += 1
            self.reqSettlementInfoConfirm(req, self.reqID)

        # 否则，推送错误信息
        else:
            self.login_status = False
            self.gateway.write_error("交易服务器登录s失败", error)
            time.sleep(30)
            self.login()

    def onRspUserLogout(self, data, error, n, last):
        """登出回报"""
        # 如果登出成功，推送日志信息
        if not error['ErrorID']:
            self.login_status = False
            logContent = '交易服务器登出完成'
            self.gateway.on_log(logContent, level = logging.INFO)

        # 否则，推送错误信息
        else:
            self.gateway.write_error("交易服务器连接失败", error)

    def onRspUserPasswordUpdate(self, data, error, n, last):
        """"""
        pass

    def onRspTradingAccountPasswordUpdate(self, data, error, n, last):
        """"""
        pass

    def onRspOrderInsert(self, data, error, n, last):
        """发单错误（柜台）"""
        self.gateway.write_error("发单错误（柜台）", error)

        event2 = Event(type=EVENT_ERRORDERINSERT + self.gateway_name)
        event2.dict['data'] = data
        event2.dict['error'] = error
        event2.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event2)

    def onRtnOrder(self, data):
        """报单回报"""
        # 更新最大报单编号
        event = Event(type=EVENT_RTNORDER + self.gateway_name)
        event.dict['data'] = data
        self.gateway.event_engine.put(event)

    def onRtnTrade(self, data):
        """成交回报"""
        # 创建报单数据对象
        event = Event(type=EVENT_RTNTRADE+self.gateway_name)
        event.dict['data'] = data
        self.gateway.event_engine.put(event)

    def onErrRtnOrderInsert(self, data, error):
        """发单错误回报（交易所）"""
        event = Event(type=EVENT_ERRORDERINSERT + self.gateway_name)
        event.dict['data'] = data
        event.dict['error'] = error
        self.gateway.event_engine.put(event)
        self.gateway.write_error("发单错误回报（交易所）", error)

    def onErrRtnOrderAction(self, data, error):
        """撤单错误回报（交易所）"""
        event = Event(type=EVENT_ERRORDERCANCEL + self.gateway_name)
        event.dict['data'] = data
        event.dict['error'] = error
        event.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event)
        self.gateway.write_error("撤单错误回报", error)

    def onRspOrderAction(self, data, error, n, last):
        """撤单错误（柜台）"""
        self.gateway.write_error("撤单错误（柜台）", error)
        event2 = Event(type=EVENT_ERRORDERCANCEL + self.gateway_name)
        event2.dict['data'] = data
        event2.dict['error'] = error
        event2.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event2)

    def onRspQueryMaxOrderVolume(self, data, error, n, last):
        """"""
        pass

    def onRspSettlementInfoConfirm(self, data, error, n, last):
        """确认结算信息回报"""
        event = Event(type=EVENT_TDLOGIN+self.gateway_name)
        self.gateway.event_engine.put(event)

        logContent = '结算信息确认完成'
        self.gateway.on_log(logContent, level = logging.INFO)
        while True:
            self.reqID += 1
            n = self.reqQryInstrument({}, self.reqID)
            if not n:
                break
            else:
                time.sleep(1)

    def onRspQryTradingAccount(self, data, error, n, last):
        """资金账户查询回报"""
        if not error:
            event = Event(type=EVENT_QRYACCOUNT + self.gateway_name )
            event.dict['data'] = data
            event.dict['last'] = last
            self.gateway.event_engine.put(event)
        else:
            logContent = '资金账户查询回报，错误代码：' + str(error['ErrorID']) + ',' + '错误信息：' + error['ErrorMsg']
            self.gateway.on_log(logContent, level = logging.DEBUG)

    def onRspParkedOrderInsert(self, data, error, n, last):
        """"""
        pass

    def onRspParkedOrderAction(self, data, error, n, last):
        """"""
        pass

    def onRspRemoveParkedOrder(self, data, error, n, last):
        """"""
        pass

    def onRspRemoveParkedOrderAction(self, data, error, n, last):
        """"""
        pass

    def onRspExecOrderInsert(self, data, error, n, last):
        """"""
        self.gateway.write_error("ExecOrder报单出错", error)
        event2 = Event(type=EVENT_ERRORDERINSERT + self.gateway_name)
        event2.dict['data'] = data
        event2.dict['error'] = error
        event2.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event2)

    def onRspExecOrderAction(self, data, error, n, last):
        """"""
        self.gateway.write_error("ExecOrder撤单出错", error)
        event2 = Event(type=EVENT_ERRORDERCANCEL + self.gateway_name)
        event2.dict['data'] = data
        event2.dict['error'] = error
        event2.dict['gateway'] = self.gateway_name
        self.gateway.event_engine.put(event2)

    def onRspForQuoteInsert(self, data, error, n, last):
        """"""
        pass

    def onRspQuoteInsert(self, data, error, n, last):
        """"""
        pass

    def onRspQuoteAction(self, data, error, n, last):
        """"""
        pass

    def onRspQryOrder(self, data, error, n, last):
        """"""
        '''请求查询报单响应'''
        if not error:
            event = Event(type=EVENT_QRYORDER + self.gateway_name )
            event.dict['data'] = data
            event.dict['last'] = last
            self.gateway.event_engine.put(event)
        else:
            logContent = '交易错误回报，错误代码：' + str(error['ErrorID']) + ',' + '错误信息：' + error['ErrorMsg']
            self.gateway.on_log(logContent, level = logging.DEBUG)

    def onRspQryTrade(self, data, error, n, last):
        """"""
        if not error:
            event = Event(type=EVENT_QRYTRADE + self.gateway_name )
            event.dict['data'] = data
            event.dict['last'] = last
            self.gateway.event_engine.put(event)
        else:
            event = Event(type=EVENT_LOG)
            logContent = '交易错误回报，错误代码：' + str(error['ErrorID']) + ',' + '错误信息：' + error['ErrorMsg']
            self.gateway.on_log(logContent, level = logging.DEBUG)

    def onRspQryInvestorPosition(self, data, error, n, last):
        """持仓查询回报"""
        if not error:
            event = Event(type=EVENT_QRYPOSITION + self.gateway_name )
            event.dict['data'] = data
            event.dict['last'] = last
            self.gateway.event_engine.put(event)
        else:
            logContent = '持仓查询回报，错误代码：' + str(error['ErrorID']) + ',' + '错误信息：' + error['ErrorMsg']
            self.gateway.on_log(logContent, level = logging.DEBUG)

    def onRspQryInvestor(self, data, error, n, last):
        """投资者查询回报"""
        if not error:
            event = Event(type=EVENT_QRYINVESTOR + self.gateway_name )
            event.dict['data'] = data
            event.dict['last'] = last
            self.gateway.event_engine.put(event)
        else:
            logContent = '合约投资者回报，错误代码：' + str(error['ErrorID']) + ',' + '错误信息：' + error['ErrorMsg']
            self.gateway.on_log(logContent, level = logging.DEBUG)

    def onRspQryTradingCode(self, data, error, n, last):
        """"""
        pass

    def onRspQryInstrumentMarginRate(self, data, error, n, last):
        """"""
        pass

    def onRspQryInstrumentCommissionRate(self, data, error, n, last):
        """"""
        pass

    def onRspQryExchange(self, data, error, n, last):
        """"""
        pass

    def onRspQryProduct(self, data, error, n, last):
        """"""
        pass

    def onRspQryInstrument(self, data, error, n, last):
        """合约查询回报"""
        if not error:
            event = Event(type=EVENT_QRYINSTRUMENT + self.gateway_name )
            event.dict['data'] = data
            event.dict['last'] = last
            self.gateway.event_engine.put(event)
        else:
            logContent = '交易错误回报，错误代码：' + str(error['ErrorID']) + ',' + '错误信息：' + error['ErrorMsg']
            self.gateway.on_log(logContent, level = logging.DEBUG)

    def onRspQryDepthMarketData(self, data, error, n, last):
        """"""
        pass

    def onRspQrySettlementInfo(self, data, error, n, last):
        """查询结算信息回报"""
        pass

    def onRspQryTransferBank(self, data, error, n, last):
        """"""
        pass

    def onRspQryInvestorPositionDetail(self, data, error, n, last):
        """"""
        pass

    def onRspQryNotice(self, data, error, n, last):
        """"""
        pass

    def onRspQrySettlementInfoConfirm(self, data, error, n, last):
        """"""
        pass

    def onRspQryInvestorPositionCombineDetail(self, data, error, n, last):
        """"""
        pass

    def onRspQryCFMMCTradingAccountKey(self, data, error, n, last):
        """"""
        pass

    def onRspQryEWarrantOffset(self, data, error, n, last):
        """"""
        pass

    def onRspQryInvestorProductGroupMargin(self, data, error, n, last):
        """"""
        pass

    def onRspQryExchangeMarginRate(self, data, error, n, last):
        """"""
        pass

    def onRspQryExchangeMarginRateAdjust(self, data, error, n, last):
        """"""
        pass

    def onRspQryExchangeRate(self, data, error, n, last):
        """"""
        pass

    def onRspQrySecAgentACIDMap(self, data, error, n, last):
        """"""
        pass

    def onRspQryOptionInstrTradeCost(self, data, error, n, last):
        """"""
        pass

    def onRspQryOptionInstrCommRate(self, data, error, n, last):
        """"""
        pass

    def onRspQryExecOrder(self, data, error, n, last):
        """"""
        pass

    def onRspQryForQuote(self, data, error, n, last):
        """"""
        pass

    def onRspQryQuote(self, data, error, n, last):
        """"""
        pass

    def onRspQryTransferSerial(self, data, error, n, last):
        """"""
        pass

    def onRspQryAccountregister(self, data, error, n, last):
        """"""
        pass

    def onRspError(self, error, n, last):
        """错误回报"""
        self.gateway.write_error("错误回报", error)

    def onRtnInstrumentStatus(self, data):
        """"""
        pass

    def onRtnTradingNotice(self, data):
        """"""
        pass

    def onRtnErrorConditionalOrder(self, data):
        """"""
        pass

    def onRtnExecOrder(self, data):
        """"""
        # 更新最大报单编号
        event = Event(type=EVENT_RTNORDER + self.gateway_name)
        event.dict['data'] = data
        self.gateway.event_engine.put(event)

    def onErrRtnExecOrderInsert(self, data, error):
        """"""
        pass

    def onErrRtnExecOrderAction(self, data, error):
        """"""
        pass

    def onErrRtnForQuoteInsert(self, data, error):
        """"""
        pass

    def onRtnQuote(self, data):
        """"""
        pass

    def onErrRtnQuoteInsert(self, data, error):
        """"""
        pass

    def onErrRtnQuoteAction(self, data, error):
        """"""
        pass

    def onRtnForQuoteRsp(self, data):
        """"""
        pass

    def onRspQryContractBank(self, data, error, n, last):
        """"""
        pass

    def onRspQryParkedOrder(self, data, error, n, last):
        """"""
        pass

    def onRspQryParkedOrderAction(self, data, error, n, last):
        """"""
        pass

    def onRspQryTradingNotice(self, data, error, n, last):
        """"""
        pass

    def onRspQryBrokerTradingParams(self, data, error, n, last):
        """"""
        pass

    def onRspQryBrokerTradingAlgos(self, data, error, n, last):
        """"""
        pass

    def onRtnFromBankToFutureByBank(self, data):
        """"""
        pass

    def onRtnFromFutureToBankByBank(self, data):
        """"""
        pass

    def onRtnRepealFromBankToFutureByBank(self, data):
        """"""
        pass

    def onRtnRepealFromFutureToBankByBank(self, data):
        """"""
        pass

    def onRtnFromBankToFutureByFuture(self, data):
        """"""
        pass

    def onRtnFromFutureToBankByFuture(self, data):
        """"""
        pass

    def onRtnRepealFromBankToFutureByFutureManual(self, data):
        """"""
        pass

    def onRtnRepealFromFutureToBankByFutureManual(self, data):
        """"""
        pass

    def onRtnQueryBankBalanceByFuture(self, data):
        """"""
        pass

    def onErrRtnBankToFutureByFuture(self, data, error):
        """"""
        pass

    def onErrRtnFutureToBankByFuture(self, data, error):
        """"""
        pass

    def onErrRtnRepealBankToFutureByFutureManual(self, data, error):
        """"""
        pass

    def onErrRtnRepealFutureToBankByFutureManual(self, data, error):
        """"""
        pass

    def onErrRtnQueryBankBalanceByFuture(self, data, error):
        """"""
        pass

    def onRtnRepealFromBankToFutureByFuture(self, data):
        """"""
        pass

    def onRtnRepealFromFutureToBankByFuture(self, data):
        """"""
        pass

    def onRspFromBankToFutureByFuture(self, data, error, n, last):
        """"""
        pass

    def onRspFromFutureToBankByFuture(self, data, error, n, last):
        """"""
        pass

    def onRspQueryBankAccountMoneyByFuture(self, data, error, n, last):
        """"""
        pass

    def onRtnOpenAccountByBank(self, data):
        """"""
        pass

    def onRtnCancelAccountByBank(self, data):
        """"""
        pass

    def onRtnChangeAccountByBank(self, data):
        """"""
        pass

    def connect(self, userID, password, brokerID, address, authCode, appID):
        """初始化连接"""
        self.userID = userID                # 账号
        self.password = password            # 密码
        self.brokerID = brokerID            # 经纪商代码
        self.address = address              # 服务器地址
        self.appID = appID                  # 产品信息
        self.authCode = authCode            # 验证码

        # 如果尚未建立服务器连接，则进行连接
        if not self.connect_status:
            # 创建C++环境中的API对象，这里传入的参数是需要用来保存.con文件的文件夹路径
            path = self.gateway.file_prefix + 'tmp' + os.path.sep
            if not os.path.exists(path):
                os.makedirs(path)
            self.createFtdcTraderApi(str(path))

            # THOST_TERT_RESTART = 0, THOST_TERT_RESUME = 1, THOST_TERT_QUICK = 2
            self.subscribePublicTopic(self.gateway.td_conn_mode)
            self.subscribePrivateTopic(self.gateway.td_conn_mode)
            # 注册服务器地址
            self.registerFront(self.address)

            # 初始化连接，成功会调用onFrontConnected
            self.init()

        # 若已经连接但尚未登录，则进行登录
        else:
            if self.requireAuthentication and not self.authStatus:
                self.authenticate()
            elif not self.login_status:
                self.login()

    def login(self):
        """连接服务器"""
        # 如果填入了用户名密码等，则登录
        if self.userID and self.password and self.brokerID:
            req = {}
            req['UserID'] = self.userID
            req['Password'] = self.password
            req['BrokerID'] = self.brokerID
            #req['UserProductInfo'] = self.gateway.product_info
            self.reqID += 1
            self.reqUserLogin(req, self.reqID)

    def authenticate(self):
        """申请验证"""
        if self.userID and self.brokerID and self.authCode and self.appID:
            req = {}
            req['UserID'] = self.userID
            req['BrokerID'] = self.brokerID
            req['AuthCode'] = self.authCode
            req['AppID'] = self.appID
            self.reqID += 1
            self.reqAuthenticate(req, self.reqID)

    def qryOrder(self):
        self.reqID += 1
        req = {}
        req['BrokerID'] = self.brokerID
        req['InvestorID'] = self.userID
        self.reqQryOrder(req, self.reqID)

    def qryTrade(self):
        self.reqID += 1
        req = {}
        req['BrokerID'] = self.brokerID
        req['InvestorID'] = self.userID
        self.reqQryTrade(req, self.reqID)

    def qryAccount(self):
        """查询账户"""
        self.reqID += 1
        self.reqQryTradingAccount({}, self.reqID)

    def qryPosition(self):
        """查询持仓"""
        self.reqID += 1
        req = {}
        req['BrokerID'] = self.brokerID
        req['InvestorID'] = self.userID
        self.reqQryInvestorPosition(req, self.reqID)

    def qryInstrument(self):
        self.reqID += 1
        req = {}
        self.reqQryInstrument(req, self.reqID)

    def sendOrder(self, iorder):
        """发单"""
        self.reqID += 1
        self.order_ref = max(self.order_ref, int(iorder.local_id))
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
        req['InvestorID'] = self.userID
        req['UserID'] = self.userID
        req['BrokerID'] = self.brokerID
        req['CombHedgeFlag'] = THOST_FTDC_HF_Speculation                        # 投机单
        req['ContingentCondition'] = THOST_FTDC_CC_Immediately                  # 立即发单
        req['ForceCloseReason'] = THOST_FTDC_FCC_NotForceClose                  # 非强平
        req['IsAutoSuspend'] = 0                                                # 非自动挂起
        req['MinVolume'] = 1                                                    # 最小成交量为1
        self.reqOrderInsert(req, self.reqID)

    def cancelOrder(self, iorder):
        """撤单"""
        self.reqID += 1
        req = {}
        req['InstrumentID'] = str(iorder.instrument)
        req['ExchangeID'] = str(iorder.exchange)
        req['ActionFlag'] = THOST_FTDC_AF_Delete
        req['BrokerID'] = self.brokerID
        req['InvestorID'] = self.userID
        req['OrderRef'] = iorder.local_id
        req['FrontID'] = int(self.frontID)
        req['SessionID'] = int(self.sessionID)
        self.reqOrderAction(req, self.reqID)

    def sendExecOrder(self, exec_order):
        inst = exec_order.instrument
        self.reqID += 1
        req = {}
        req['BrokerID'] = self.brokerID
        req['InvestorID'] = self.userID
        req['ExchangeID'] = str(exec_order.exchange)

        req['InstrumentID'] = exec_order.instrument
        req['ExecOrderRef'] = exec_order.local_id
        req['OffsetFlag'] = exec_order.action_type
        req['HedgeFlag'] = THOST_FTDC_HF_Speculation
        req['ActionType'] = THOST_FTDC_ACTP_Exec
        req['PositionDirection'] = THOST_FTDC_PD_Long if exec_order.direction == THOST_FTDC_D_Buy \
                                 else THOST_FTDC_PD_Short
        if exec_order.exchange == 'CFFEX':
            close_flag = THOST_FTDC_EOCF_AutoClose
            reserve_flag = THOST_FTDC_EOPF_UnReserve
        else:
            close_flag = THOST_FTDC_EOCF_NotToClose
            reserve_flag = THOST_FTDC_EOPF_Reserve
        req['ReservePositionFlag'] = reserve_flag
        req['CloseFlag'] = close_flag
        #req['UserProductInfo'] = self.gateway.product_info
        self.reqExecOrderInsert(req, self.reqID)

    def cancelExecOrder(self, exec_order):
        inst = exec_order.instrument
        self.reqID += 1
        req = {}
        req['BrokerID'] = self.brokerID
        req['InvestorID'] = self.userID
        req['InstrumentID'] = exec_order.instrument
        req['ExecOrderActionRef'] = ''
        req['ExecOrderRef'] = exec_order.local_id
        if len(exec_order.sys_id) >0:
            req['ExecOrderSysID'] = exec_order.sys_id
        else:
            req['ExecOrderRef'] = exec_order.local_id
            req['FrontID'] = self.frontID
            req['SessionID'] = self.sessionID
        req['ExchangeID'] = exec_order.exchange
        req['ActionType'] = THOST_FTDC_ACTP_Exec
        self.reqExecOrderAction(req, self.reqID)

    def sendRFQ(self, req):
        self.order_ref += 1
        ctp_req = {
            "InstrumentID": req.symbol,
            "ExchangeID": req.exchange,
            "ForQuoteRef": str(self.order_ref),
            "BrokerID": self.brokerid,
            "InvestorID": self.userid
        }
        self.reqID += 1
        self.reqForQuoteInsert(ctp_req, self.reqID)

    def close(self):
        if self.connect_status:
            self.exit()
