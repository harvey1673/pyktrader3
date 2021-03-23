import cmq_book
import cmq_market_data
import datetime
import cmq_risk_engine

def get_book(book_name):
    return cmq_book.get_book_from_db(book_name, [2], 'trade_data')

def get_market(book, today):
    return cmq_market_data.load_market_data(book.mkt_deps, value_date = today)

def get_engine(book, mkt, greeks):
    return cmq_risk_engine.CMQRiskEngine(book, mkt, greeks)