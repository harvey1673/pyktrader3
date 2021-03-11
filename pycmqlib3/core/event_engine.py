#-*- coding:utf-8 -*-
import queue
import sys, traceback
import threading
from dataclasses import dataclass, field
from typing import Any
from . event_type import EVENT_TIMER
from . event_priority import Event_Priority_Basic

@dataclass(order=True)
class PrioritizedItem:
    key: int
    item: Any = field(compare=False)


class RepeatTimer():

    def __init__(self,t,hFunction):
        self.t=t
        self.hFunction = hFunction
        self.thread = threading.Timer(self.t,self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = threading.Timer(self.t,self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()


class RepeatTimer2(threading.Thread):
    def __init__(self, interval, callable, *args, **kwargs):
        threading.Thread.__init__(self)
        self.interval = interval
        self.callable = callable
        self.args = args
        self.kwargs = kwargs
        self.event = threading.Event()
        self.event.set()

    def run(self):
        while self.event.is_set():
            t = threading.Timer(self.interval, self.callable,
                                self.args, self.kwargs)
            t.start()
            t.join()

    def cancel(self):
        self.event.clear()


class EventEngine(object):
    def __init__(self, timerFreq = 1.0):
        self.queue = queue.Queue()
        self.is_active = False
        self.thread = threading.Thread(target = self.run)
        self.timer = RepeatTimer2(timerFreq, self.onTimer)
        self.handlers = {}
    
    def run(self):
        while self.is_active == True:
            try:
                event = self.queue.get(block = True, timeout = 1)
                self.process(event)
            except queue.Empty:
                pass
            except:
                e = sys.exc_info()[0]
                print('error msg:', str(e))
                traceback.print_exc()
            finally:
                pass
    
    def process(self, event):
        """processing event"""
        if event.type in self.handlers:
            [handler(event) for handler in self.handlers[event.type]] 

    def onTimer(self):
        if self.is_active:
            event = Event(type=EVENT_TIMER)
            self.put(event)

    def start(self):
        self.is_active = True
        self.thread.start()
        self.timer.start()

    def stop(self):
        self.is_active = False
        self.timer.cancel()
        self.timer = None
        self.thread.join()
        self.thread = None
    
    def register(self, etype, handler):
        try:
            handlerList = self.handlers[etype]
        except KeyError:
            handlerList = []
            self.handlers[etype] = handlerList

        if handler not in handlerList:
            handlerList.append(handler)

    def unregister(self, etype, handler):
        try:
            handlerList = self.handlers[etype]
            
            if handler in handlerList:
                handlerList.remove(handler)

            if not handlerList:
                del self.handlers[etype]
        except KeyError:
            pass
    
    def put(self, event):
        self.queue.put(event)


class PriEventEngine(EventEngine):
    def __init__(self, timerFreq = 1.0):
        super(PriEventEngine, self).__init__(timerFreq)
        self.queue = queue.PriorityQueue()

    def put(self, event):
        event_obj = PrioritizedItem(event.priority, event)
        self.queue.put(event_obj)

    def process(self, event_obj):
        event = event_obj.item
        if event.type in self.handlers:
            [handler(event) for handler in self.handlers[event.type]] 


class Event(object):
    def __init__(self, type=None, priority = None, priority_map = Event_Priority_Basic):
        """Constructor"""
        self.type = type      # event type
        if (priority == None) and (type!= None):
            dict_key = type.split('.')[0] + '.'
            self.priority = priority_map.get(dict_key, 100)
        else:
            self.priority = priority            
        self.dict = {}         # a dict to store the dat aassociated with the event


def test():
    """test function"""
    import sys, time
    from datetime import datetime
    def simpletest(event):
        print('processing event of timer：%s' % str(datetime.now()))
    
    ee = EventEngine(0.5)
    ee.register(EVENT_TIMER, simpletest)
    ee.start()
    for i in range(1000):
        time.sleep(1.0)
        print("loop %s" % i)
        if i>5:
            ee.stop()
            break


if __name__ == '__main__':
    test()
