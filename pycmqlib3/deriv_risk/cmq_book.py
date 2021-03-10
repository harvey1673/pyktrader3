# -*- coding:utf-8 -*-
import json
import datetime
import itertools
import cmq_inst
import dbaccess
import misc

class CMQDealStatus:
    Perspective, PendingSignoff, Live, Matured, Unwinded, Cancelled = list(range(6))

class CMQBookStatus:
    Test, UAT, Prod = list(range(3))

def agg_mkt_deps(mkt_deps, inst_list):
    for inst in inst_list:
        for key in inst.mkt_deps:
            if key not in mkt_deps:
                mkt_deps[key] = {}
            for idx in inst.mkt_deps[key]:
                if idx not in mkt_deps[key]:
                    mkt_deps[key][idx] = []
                mkt_deps[key][idx] = list(set(mkt_deps[key][idx]).union(set(inst.mkt_deps[key][idx])))
                mkt_deps[key][idx].sort()
    return mkt_deps

class CMQDeal(object):
    id_generator = itertools.count(int(datetime.datetime.strftime(datetime.datetime.now(), '%d%H%M%S')))
    class_params = {'trader': 'harvey', 'sales': 'harvey', 'status': CMQDealStatus.Perspective, \
                    'cpty': 'dummy', 'strategy': 'test',\
                    'enter_date': datetime.date.today(), 'last_updated': datetime.datetime.now(), \
                    'last_date': datetime.date.today(), \
                    'external_id': 'dummy', 'external_src': 'dummy', \
                    'internal_id': 'dummy', 'business': 'commod', \
                    'desk': 'CST', 'portfolio': 'test', 'product': 'SGIRO', \
                    'day1_comments': '', 'commission': 0.0, 'premium': 0.0, 'reporting_ccy': 'USD',}
    def __init__(self, deal_data):
        self.mkt_deps = {}
        self.ccy_converter = {}
        if isinstance(deal_data, str):
            deal_data = json.loads(deal_data)
        self.id = deal_data.get('id', next(self.id_generator))
        self.update_deal_data(deal_data)
        self.ccy_converter[self.reporting_ccy] = 1.0

    def set_market_data(self, market_data):
        for inst, pos in self.positions:
            inst.set_market_data(market_data)
            if inst.ccy != self.reporting_ccy:
                self.ccy_converter[inst.ccy] = misc.conv_fx_rate(inst.ccy, self.reporting_ccy, market_data['FXFwd'])

    def add_instrument(self, inst_data, pos):
        new_inst = self.create_instrument(inst_data)
        self.positions.append([new_inst, pos])

    def create_instrument(self, inst_data):
        if 'inst_type' in inst_data:
            inst_type = inst_data["inst_type"]
            cls_name = cmq_inst.inst_type_map[inst_type]
            cls_str = cls_name.split('.')
            inst_cls = getattr(__import__(str(cls_str[0])), str(cls_str[1]))
            return inst_cls.create_instrument(inst_data)
        else:
            print('inst_type key is missing in the instrument data')
            return None

    def price(self):
        return sum([inst.price() * self.ccy_converter[inst.ccy] * pos for inst, pos in self.positions])

    def update_deal_data(self, deal_data):
        d = self.__dict__
        for key in self.class_params:
            d[key] = deal_data.get(key, self.class_params[key])
            if d[key] == None:
                d[key] = self.class_params[key]
        if isinstance(deal_data['positions'], str):
            pos_data =  json.loads(deal_data['positions'])
        else:
            pos_data = deal_data['positions']
        self.positions = [ [self.create_instrument(inst_data), pos] for inst_data, pos in pos_data ]
        self.update_mkt_deps()

    def update_mkt_deps(self):
        self.mkt_deps = {}
        inst_list = [inst for inst, pos in self.positions]
        agg_mkt_deps(self.mkt_deps, inst_list)
        for inst in inst_list:
            if inst.ccy != self.reporting_ccy:
                fx_pair = None
                fx_direction = misc.get_mkt_fxpair(self.reporting_ccy, inst.ccy)
                if fx_direction > 0 :
                    fx_pair =  '/'.join([self.reporting_ccy, inst.ccy])
                elif fx_direction < 0:
                    fx_pair = '/'.join([inst.ccy, self.reporting_ccy])
                else:
                    print("ERROR: unsupported FX pair: %s - %s" % (self.reporting_ccy, inst.ccy))
                    fx_pair = None
                if fx_pair != None:
                    if 'FXFwd' not in self.mkt_deps:
                        self.mkt_deps['FXFwd'] = {}
                    self.mkt_deps['FXFwd'][fx_pair] = ['ALL']

    def remove_instrument(self, inst_obj):
        self.positions = [ [inst, pos] for inst, pos in self.positions if inst != inst_obj ]
        self.update_mkt_deps()

    def __str__(self):
        param_list = []
        for param in self.class_params:
            data = getattr(self, param)
            if type(data).__name__ in ['date', 'datetime']:
                data = str(data)
            param_list.append((param, data))
        output = dict(param_list)
        output['positions'] = [ [str(inst), pos] for inst, pos in self.positions ]
        return json.dumps(output)

class CMQBook(object):
    class_params = {'name': 'test_book', 'owner': 'harvey', 'reporting_ccy': 'USD', 'status': CMQBookStatus.Test}
    def __init__(self, book_data):
        self.mkt_deps = {}
        if isinstance(book_data, str):
            book_data = json.loads(book_data)
        self.load_book(book_data)

    def get_deal_by_id(self, key, field = 'internal_id'):
        index = None
        for idx, deal in enumerate(self.deal_list):
            if getattr(deal, field) == key:
                index = idx
                break
        if index:
            return self.deal_list[index]
        else:
            return None

    def load_book(self, book_data):
        d = self.__dict__
        for key in self.class_params:
            d[key] = book_data.get(key, self.class_params[key])
        self.deal_list = [ CMQDeal(deal_data) for deal_data in book_data.get('deal_list', [])]
        self.update_inst_dict()

    def book_deal(self, cmq_deal):
        self.deal_list.append(cmq_deal)
        for inst, pos in cmq_deal.positions:
            if inst not in self.inst_dict:
                self.inst_dict[inst] = 0
                agg_mkt_deps(self.mkt_deps, [inst])
            self.inst_dict[inst] += pos

    def update_inst_dict(self):
        self.inst_dict = {}
        self.mkt_deps= {}
        for cmq_deal in self.deal_list:
            cmq_deal.update_mkt_deps()
            for inst, pos in cmq_deal.positions:
                if inst not in self.inst_dict:
                    self.inst_dict[inst] = 0
                self.inst_dict[inst] += pos
        agg_mkt_deps(self.mkt_deps, self.deal_list)

    def price(self):
        return sum([ deal.price() for deal in self.deal_list])

    def set_market_data(self, market_data):
        for deal in self.deal_list:
            deal.set_market_data(market_data)

    def __str__(self):
        param_list = []
        for param in self.class_params:
            data = getattr(self, param)
            if type(data).__name__ in ['date', 'datetime']:
                data = str(data)
            param_list.append((param, data))
        output = dict(param_list)
        output['deal_list'] = [ str(deal) for deal in self.deal_list]
        return json.dumps(output)

def get_book_from_db(book = '', strategy = '', status = [2, 0], dbtable = 'trade_data'):
    cnx = dbaccess.connect(**dbaccess.trade_dbconfig)
    df = dbaccess.load_deal_data(cnx, dbtable, book = book, strategy = strategy, deal_status = status)
    deal_list = df.to_dict(orient = 'record')
    book_data = {'book': '_'.join([book, strategy]),  'owner': 'harvey', 'reporting_ccy': 'USD', \
                 'status': CMQBookStatus.Prod, 'deal_list': deal_list }
    book_obj = CMQBook(book_data)
    return book_obj

def save_book_into_db(book, dbtable = 'trade_data'):
    columns = ['internal_id', 'external_id', 'business', 'strategy', 'book', \
               'product', 'external_src', 'cpty', 'premium', \
               'status', 'positions', 'enter_date', 'last_date', 'reporting_ccy', ]
    stmt = "insert into {table} ({variables}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(
        table=dbtable, variables=','.join(columns))
    cnx = dbaccess.connect(**dbaccess.trade_dbconfig)
    cursor = cnx.cursor()
    for d in book.deal_list:
        deal_dict = {}
        for col in columns:
            if col == 'book':
                data =  book.name
            else:
                data = getattr(d, col)
            if type(data).__name__ in ['date', 'datetime']:
                data = str(data)
            elif col == 'positions':
                data = json.dumps([[inst.dict(), pos] for inst, pos in data])
            deal_dict[col] = data
        args = tuple([deal_dict[col] for col in columns])
        cursor.execute(stmt, args)
        cnx.commit()


if __name__ == '__main__':
    pass