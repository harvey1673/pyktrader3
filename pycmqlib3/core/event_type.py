#-*- coding:utf-8 -*-
EVENT_TIMER = 'eTimer.'                  # timer event triggerd each sec
EVENT_LOG = 'eLog.'                      # log event for logging
EVENT_MAIL = 'eMail.'                    # email event for sending emails

EVENT_TDLOGIN = 'eTdLogin.'                  # trade data connection login event
EVENT_TDDISCONNECTED = 'eTdDisconnected.'    # trade data disconnection event

# Wind接口相关事件
EVENT_WIND_CONNECTREQ = 'eWindConnectReq.'  # Wind connection request event

EVENT_MARKETDATA = 'eMarketData.'            # market data event
EVENT_MARKETDATA_CONTRACT = 'eMarketData.'  # market data event for partocular contract

EVENT_TICK = 'eTick.'                       # tick data event
EVENT_MIN_BAR = 'eMinBar.'                  # min bar data event
EVENT_CONTRACT = 'eContract.'               #  contract data event
EVENT_RPCMKTDATA = 'eRPCMktData.'           # RPC market data event
EVENT_MKTDATA_EOD ='eEOD.'                  # EOD event

EVENT_RTNTRADE = 'eRtnTrade.'               # trade return event on trade error/success
EVENT_TRADE = 'eTrade.'                     # trade event
EVENT_TRADE_CONTRACT = 'eTrade.'            # trade event on particular contract

EVENT_RTNORDER = 'eRtnOrder.'               # order return event on order error/success
EVENT_ORDER = 'eOrder.'                     # order event
EVENT_ORDER_ORDERREF = 'eOrder.'            # order event on particular contract

EVENT_POSITION = 'ePosition.'               # position event 
EVENT_QRYPOSITION = 'eQryPosition.'         # query position event
EVENT_INSTRUMENT = 'eInstrument.'           # instrument event
EVENT_QRYINSTRUMENT = 'eQryInstrument.'     # query instrument event
EVENT_INVESTOR = 'eInvestor.'               # investor event
EVENT_QRYINVESTOR = 'eQryInvestor.'         # query investor event
EVENT_QRYACCOUNT = 'eQryAccount.'           # query account event
EVENT_ACCOUNT = 'eAccount.'                 # account event

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
    check_dict = {}
    
    global_dict = globals()    
    
    for key, value in list(global_dict.items()):
        if '__' not in key: 
            if value in check_dict:
                check_dict[value].append(key)
            else:
                check_dict[value] = [key]
            
    for key, value in list(check_dict.items()):
        if len(value)>1:
            print('there are redundent definitions:' + str(key)) 
            for name in value:
                print(name)
            print('')
        
    print('test is done')
    

if __name__ == '__main__':
    test()
