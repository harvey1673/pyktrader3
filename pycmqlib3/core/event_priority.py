#-*- coding:utf-8 -*-
from . event_type import EVENT_TIMER, EVENT_LOG, EVENT_MAIL, EVENT_MARKETDATA, EVENT_RPCMKTDATA, \
    EVENT_TICK, EVENT_RTNTRADE, EVENT_TRADE, EVENT_TRADEX, EVENT_RTNORDER, EVENT_ORDER, EVENT_POSITION, \
    EVENT_ERRORDERINSERT, EVENT_ERRORDERCANCEL, EVENT_ETRADEUPDATE, EVENT_DB_WRITE, EVENT_WIND_CONNECTREQ
Event_Priority_Basic = {}

Event_Priority_Basic[EVENT_TIMER] = 50
Event_Priority_Basic[EVENT_LOG] = 1000
Event_Priority_Basic[EVENT_MAIL] = 70
Event_Priority_Basic[EVENT_MARKETDATA] = 50
Event_Priority_Basic[EVENT_RPCMKTDATA] = 100
Event_Priority_Basic[EVENT_TICK] = 50
Event_Priority_Basic[EVENT_RTNTRADE] = 40
Event_Priority_Basic[EVENT_TRADE] = 40
Event_Priority_Basic[EVENT_TRADEX] = 40
Event_Priority_Basic[EVENT_RTNORDER] = 40
Event_Priority_Basic[EVENT_ORDER] = 40
Event_Priority_Basic[EVENT_POSITION] = 60
Event_Priority_Basic[EVENT_ERRORDERINSERT] = 40
Event_Priority_Basic[EVENT_ERRORDERCANCEL] = 40
Event_Priority_Basic[EVENT_ETRADEUPDATE] = 40
Event_Priority_Basic[EVENT_DB_WRITE] = 5000
Event_Priority_Basic[EVENT_WIND_CONNECTREQ] = 50

Event_Priority_Realtime = {}
