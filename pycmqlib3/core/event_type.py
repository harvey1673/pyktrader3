#-*- coding:utf-8 -*-
EVENT_TIMER = 'eTimer.'                  # 计时器事件，每隔1秒发送一次
EVENT_LOG = 'eLog.'                      # 日志事件，通常使用某个监听函数直接显示
EVENT_MAIL = 'eMail.'

EVENT_TDLOGIN = 'eTdLogin.'                  # 交易服务器登录成功事件
EVENT_TDDISCONNECTED = 'eTdDisconnected.'

# Wind接口相关事件
EVENT_WIND_CONNECTREQ = 'eWindConnectReq.'  # Wind接口请求连接事件

EVENT_MARKETDATA = 'eMarketData.'            # 行情推送事件
EVENT_MARKETDATA_CONTRACT = 'eMarketData.'  # 特定合约的行情事件

EVENT_TICK = 'eTick.'            # 行情推送事件
EVENT_MIN_BAR = 'eMinBar.'
EVENT_CONTRACT = 'eContract.'   #
EVENT_RPCMKTDATA = 'eRPCMktData.'
EVENT_MKTDATA_EOD ='eEOD.'

EVENT_RTNTRADE = 'eRtnTrade.'                      # 成交推送事件
EVENT_TRADE = 'eTrade.'                      # 成交推送事件
EVENT_TRADE_CONTRACT = 'eTrade.'            # 特定合约的成交事件

EVENT_RTNORDER = 'eRtnOrder.'                      # 报单推送事件
EVENT_ORDER = 'eOrder.'                      # 报单推送事件
EVENT_ORDER_ORDERREF = 'eOrder.'            # 特定报单号的报单事件

EVENT_POSITION = 'ePosition.'                # 持仓查询回报事件
EVENT_QRYPOSITION = 'eQryPosition.'                # 持仓查询回报事件
EVENT_INSTRUMENT = 'eInstrument.'            # 合约查询回报事件
EVENT_QRYINSTRUMENT = 'eQryInstrument.'            # 合约查询回报事件
EVENT_INVESTOR = 'eInvestor.'                # 投资者查询回报事件
EVENT_QRYINVESTOR = 'eQryInvestor.'                # 投资者查询回报事件
EVENT_QRYACCOUNT = 'eQryAccount.'                  # 账户查询回报事件
EVENT_ACCOUNT = 'eAccount.'                  # 账户查询回报事件

EVENT_MARGINRATE = 'eMarginRate.'
EVENT_DAYSWITCH = 'eDaySwitch.'
EVENT_DB_WRITE = 'eDatabaseWrite.'

EVENT_XTRADESTATUS = 'eXtradeStatus'
EVENT_ETRADEUPDATE = 'eTradeUpdate.'
EVENT_ERRORDERINSERT = 'eErrOrderInsert.'
EVENT_ERRORDERCANCEL = 'eErrOrderCancel.'
EVENT_QRYORDER = 'eQryOrder.'
EVENT_QRYTRADE = 'eQryTrade.'
EVENT_TRADEX = 'eTradeX.'

#----------------------------------------------------------------------
def test():
    """检查是否存在内容重复的常量定义"""
    check_dict = {}
    
    global_dict = globals()    
    
    for key, value in list(global_dict.items()):
        if '__' not in key:                       # 不检查python内置对象
            if value in check_dict:
                check_dict[value].append(key)
            else:
                check_dict[value] = [key]
            
    for key, value in list(check_dict.items()):
        if len(value)>1:
            print('存在重复的常量定义:' + str(key)) 
            for name in value:
                print(name)
            print('')
        
    print('测试完毕')
    

# 直接运行脚本可以进行测试
if __name__ == '__main__':
    test()
