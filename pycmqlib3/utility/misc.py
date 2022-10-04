# -*- coding:utf-8 -*-
import workdays
import datetime
from dateutil.relativedelta import relativedelta
import calendar
import math
import copy
import pandas as pd
from . import dbaccess

BDAYS_PER_YEAR = 245.0
AMERICAN_OPTION_STEPS = 40
MKT_DATA_BIGNUMBER = 10000000
NO_ENTRY_TIME = datetime.datetime(1970, 1, 1, 0, 0, 0)

sign = lambda x: math.copysign(1, x)

month_code_map = {'f': 1,
                  'g': 2,
                  'h': 3,
                  'j': 4,
                  'k': 5,
                  'm': 6,
                  'n': 7,
                  'q': 8,
                  'u': 9,
                  'v': 10,
                  'x': 11,
                  'z': 12}

CHN_Holidays = [datetime.date(2000, 1, 3), datetime.date(2000, 1, 31), datetime.date(2000, 2, 1),
                datetime.date(2000, 2, 2), datetime.date(2000, 2, 3), datetime.date(2000, 2, 4),
                datetime.date(2000, 2, 7), datetime.date(2000, 2, 8), datetime.date(2000, 2, 9),
                datetime.date(2000, 2, 10), datetime.date(2000, 2, 11), datetime.date(2000, 5, 1),
                datetime.date(2000, 5, 2), datetime.date(2000, 5, 3), datetime.date(2000, 5, 4),
                datetime.date(2000, 5, 5), datetime.date(2000, 10, 2), datetime.date(2000, 10, 3),
                datetime.date(2000, 10, 4), datetime.date(2000, 10, 5), datetime.date(2000, 10, 6),
                datetime.date(2001, 1, 1), datetime.date(2001, 1, 22), datetime.date(2001, 1, 23),
                datetime.date(2001, 1, 24), datetime.date(2001, 1, 25), datetime.date(2001, 1, 26),
                datetime.date(2001, 1, 29), datetime.date(2001, 1, 30), datetime.date(2001, 1, 31),
                datetime.date(2001, 2, 1), datetime.date(2001, 2, 2), datetime.date(2001, 5, 1),
                datetime.date(2001, 5, 2), datetime.date(2001, 5, 3), datetime.date(2001, 5, 4),
                datetime.date(2001, 5, 7), datetime.date(2001, 10, 1), datetime.date(2001, 10, 2),
                datetime.date(2001, 10, 3), datetime.date(2001, 10, 4), datetime.date(2001, 10, 5),
                datetime.date(2002, 1, 1), datetime.date(2002, 1, 2), datetime.date(2002, 1, 3),
                datetime.date(2002, 2, 11), datetime.date(2002, 2, 12), datetime.date(2002, 2, 13),
                datetime.date(2002, 2, 14), datetime.date(2002, 2, 15), datetime.date(2002, 2, 18),
                datetime.date(2002, 2, 19), datetime.date(2002, 2, 20), datetime.date(2002, 2, 21),
                datetime.date(2002, 2, 22),datetime.date(2002, 5, 1),datetime.date(2002, 5, 2),
                datetime.date(2002, 5, 3),datetime.date(2002, 5, 6),datetime.date(2002, 5, 7),
                datetime.date(2002, 9, 30),datetime.date(2002, 10, 1),datetime.date(2002, 10, 2),
                datetime.date(2002, 10, 3),datetime.date(2002, 10, 4),datetime.date(2002, 10, 7),
                datetime.date(2003, 1, 1),datetime.date(2003, 1, 30),datetime.date(2003, 1, 31),
                datetime.date(2003, 2, 3),datetime.date(2003, 2, 4),datetime.date(2003, 2, 5),
                datetime.date(2003, 2, 6),datetime.date(2003, 2, 7),datetime.date(2003, 5, 1),
                datetime.date(2003, 5, 2),datetime.date(2003, 5, 5),datetime.date(2003, 5, 6),
                datetime.date(2003, 5, 7),datetime.date(2003, 5, 8),datetime.date(2003, 5, 9),
                datetime.date(2003, 10, 1),datetime.date(2003, 10, 2),datetime.date(2003, 10, 3),
                datetime.date(2003, 10, 6),datetime.date(2003, 10, 7),datetime.date(2004, 1, 1),
                datetime.date(2004, 1, 19),datetime.date(2004, 1, 20),datetime.date(2004, 1, 21),
                datetime.date(2004, 1, 22),datetime.date(2004, 1, 23),datetime.date(2004, 1, 26),
                datetime.date(2004, 1, 27),datetime.date(2004, 1, 28),datetime.date(2004, 5, 3),
                datetime.date(2004, 5, 4),datetime.date(2004, 5, 5),datetime.date(2004, 5, 6),
                datetime.date(2004, 5, 7),datetime.date(2004, 10, 1),datetime.date(2004, 10, 4),
                datetime.date(2004, 10, 5),datetime.date(2004, 10, 6),datetime.date(2004, 10, 7),
                datetime.date(2005, 1, 3),datetime.date(2005, 2, 7),datetime.date(2005, 2, 8),
                datetime.date(2005, 2, 9),datetime.date(2005, 2, 10),datetime.date(2005, 2, 11),
                datetime.date(2005, 2, 14),datetime.date(2005, 2, 15),datetime.date(2005, 5, 2),
                datetime.date(2005, 5, 3),datetime.date(2005, 5, 4),datetime.date(2005, 5, 5),
                datetime.date(2005, 5, 6),datetime.date(2005, 10, 3),datetime.date(2005, 10, 4),
                datetime.date(2005, 10, 5),datetime.date(2005, 10, 6),datetime.date(2005, 10, 7),
                datetime.date(2006, 1, 2),datetime.date(2006, 1, 3),datetime.date(2006, 1, 26),
                datetime.date(2006, 1, 27),datetime.date(2006, 1, 30),datetime.date(2006, 1, 31),
                datetime.date(2006, 2, 1),datetime.date(2006, 2, 2),datetime.date(2006, 2, 3),
                datetime.date(2006, 5, 1),datetime.date(2006, 5, 2),datetime.date(2006, 5, 3),
                datetime.date(2006, 5, 4),datetime.date(2006, 5, 5),datetime.date(2006, 10, 2),
                datetime.date(2006, 10, 3),datetime.date(2006, 10, 4),datetime.date(2006, 10, 5),
                datetime.date(2006, 10, 6),
                datetime.date(2007, 1, 1),datetime.date(2007, 1, 2),datetime.date(2007, 1, 3),
                datetime.date(2007, 2, 19),datetime.date(2007, 2, 20),datetime.date(2007, 2, 21),
                datetime.date(2007, 2, 22),datetime.date(2007, 2, 23),datetime.date(2007, 5, 1),
                datetime.date(2007, 5, 2),datetime.date(2007, 5, 3),datetime.date(2007, 5, 4),
                datetime.date(2007, 5, 7),datetime.date(2007, 10, 1),datetime.date(2007, 10, 2),
                datetime.date(2007, 10, 3),datetime.date(2007, 10, 4),datetime.date(2007, 10, 5),
                datetime.date(2007, 12, 31),
                datetime.date(2008, 1, 1),datetime.date(2008, 2, 6),datetime.date(2008, 2, 7),
                datetime.date(2008, 2, 8),datetime.date(2008, 2, 11),datetime.date(2008, 2, 12),
                datetime.date(2008, 4, 4),datetime.date(2008, 5, 1),datetime.date(2008, 5, 2),
                datetime.date(2008, 6, 9),datetime.date(2008, 9, 15),datetime.date(2008, 9, 29),
                datetime.date(2008, 9, 30),datetime.date(2008, 10, 1),datetime.date(2008, 10, 2),
                datetime.date(2008, 10, 3),
                datetime.date(2009, 1, 1),datetime.date(2009, 1, 2),datetime.date(2009, 1, 26),
                datetime.date(2009, 1, 27),datetime.date(2009, 1, 28),datetime.date(2009, 1, 29),
                datetime.date(2009, 1, 30),datetime.date(2009, 4, 6),datetime.date(2009, 5, 1),
                datetime.date(2009, 5, 28),datetime.date(2009, 5, 29),datetime.date(2009, 10, 1),
                datetime.date(2009, 10, 2),datetime.date(2009, 10, 5),datetime.date(2009, 10, 6),
                datetime.date(2009, 10, 7),datetime.date(2009, 10, 8),
                datetime.date(2010, 1, 1),datetime.date(2010, 2, 15),datetime.date(2010, 2, 16),
                datetime.date(2010, 2, 17),datetime.date(2010, 2, 18),datetime.date(2010, 2, 19),
                datetime.date(2010, 4, 5),datetime.date(2010, 5, 3),datetime.date(2010, 6, 14),
                datetime.date(2010, 6, 15),datetime.date(2010, 6, 16),datetime.date(2010, 9, 22),
                datetime.date(2010, 9, 23),datetime.date(2010, 9, 24),datetime.date(2010, 10, 1),
                datetime.date(2010, 10, 4),datetime.date(2010, 10, 5),datetime.date(2010, 10, 6),
                datetime.date(2010, 10, 7),
                datetime.date(2011, 1, 3),datetime.date(2011, 2, 2),datetime.date(2011, 2, 3),
                datetime.date(2011, 2, 4),datetime.date(2011, 2, 7),datetime.date(2011, 2, 8),
                datetime.date(2011, 4, 4),datetime.date(2011, 4, 5),datetime.date(2011, 5, 2),
                datetime.date(2011, 6, 6),datetime.date(2011, 9, 12),datetime.date(2011, 10, 3),
                datetime.date(2011, 10, 4),datetime.date(2011, 10, 5),datetime.date(2011, 10, 6),
                datetime.date(2011, 10, 7),
                datetime.date(2012, 1, 2),datetime.date(2012, 1, 3),datetime.date(2012, 1, 23),
                datetime.date(2012, 1, 24),datetime.date(2012, 1, 25),datetime.date(2012, 1, 26),
                datetime.date(2012, 1, 27),datetime.date(2012, 4, 2),datetime.date(2012, 4, 3),
                datetime.date(2012, 4, 4),datetime.date(2012, 4, 30),datetime.date(2012, 5, 1),
                datetime.date(2012, 6, 22),datetime.date(2012, 10, 1),datetime.date(2012, 10, 2),
                datetime.date(2012, 10, 3),datetime.date(2012, 10, 4),datetime.date(2012, 10, 5),
                datetime.date(2013, 1, 1),datetime.date(2013, 1, 2),datetime.date(2013, 1, 3),
                datetime.date(2013, 2, 11),datetime.date(2013, 2, 12),datetime.date(2013, 2, 13),
                datetime.date(2013, 2, 14),datetime.date(2013, 2, 15),datetime.date(2013, 4, 4),
                datetime.date(2013, 4, 5),datetime.date(2013, 4, 29),datetime.date(2013, 4, 30),
                datetime.date(2013, 5, 1),datetime.date(2013, 6, 10),datetime.date(2013, 6, 11),
                datetime.date(2013, 6, 12),datetime.date(2013, 9, 19),datetime.date(2013, 9, 20),
                datetime.date(2013, 10, 1),datetime.date(2013, 10, 2),datetime.date(2013, 10, 3),
                datetime.date(2013, 10, 4),datetime.date(2013, 10, 7),
                datetime.date(2014, 1, 1), datetime.date(2014, 1, 2), datetime.date(2014, 1, 3),
                datetime.date(2014, 1, 31), datetime.date(2014, 2, 3), datetime.date(2014, 2, 4),
                datetime.date(2014, 2, 5), datetime.date(2014, 2, 6), datetime.date(2014, 4, 7),
                datetime.date(2014, 5, 1), datetime.date(2014, 5, 2), datetime.date(2014, 6, 2),
                datetime.date(2014, 9, 8), datetime.date(2014, 10, 1), datetime.date(2014, 10, 2),
                datetime.date(2014, 10, 3), datetime.date(2014, 10, 6), datetime.date(2014, 10, 7),
                datetime.date(2015, 1, 1), datetime.date(2015, 1, 2), datetime.date(2015, 2, 18),
                datetime.date(2015, 2, 19),
                datetime.date(2015, 2, 20), datetime.date(2015, 2, 23), datetime.date(2015, 2, 24),
                datetime.date(2015, 4, 6), datetime.date(2015, 5, 1), datetime.date(2015, 6, 22),
                datetime.date(2015, 9, 3), datetime.date(2015, 9, 4), datetime.date(2015, 10, 1),
                datetime.date(2015, 10, 2), datetime.date(2015, 10, 5), datetime.date(2015, 10, 6),
                datetime.date(2015, 10, 7),
                datetime.date(2016, 1, 1), datetime.date(2016, 2, 8), datetime.date(2016, 2, 9),
                datetime.date(2016, 2, 10), datetime.date(2016, 2, 11), datetime.date(2016, 2, 12),
                datetime.date(2016, 4, 4), datetime.date(2016, 5, 2), datetime.date(2016, 6, 9),
                datetime.date(2016, 6, 10), datetime.date(2016, 9, 15), datetime.date(2016, 9, 16),
                datetime.date(2016, 10, 3), datetime.date(2016, 10, 4), datetime.date(2016, 10, 5),
                datetime.date(2016, 10, 6), datetime.date(2016, 10, 7),
                datetime.date(2017, 1, 2), datetime.date(2017, 1, 27), datetime.date(2017, 1, 30),
                datetime.date(2017, 1, 31), datetime.date(2017, 2, 1), datetime.date(2017, 2, 2),
                datetime.date(2017, 4, 3), datetime.date(2017, 4, 4), datetime.date(2017, 5, 1),
                datetime.date(2017, 5, 29), datetime.date(2017, 5, 30),
                datetime.date(2017, 10, 2), datetime.date(2017, 10, 3), datetime.date(2017, 10, 4),
                datetime.date(2017, 10, 5), datetime.date(2017, 10, 6),
                datetime.date(2018, 1, 1), datetime.date(2018, 2, 15), datetime.date(2018, 2, 16),
                datetime.date(2018, 2, 19), datetime.date(2018, 2, 20), datetime.date(2018, 2, 21),
                datetime.date(2018, 4, 5), datetime.date(2018, 4, 6),  datetime.date(2018, 4, 30),
                datetime.date(2018, 5, 1), datetime.date(2018, 6, 18),
                datetime.date(2018, 9, 24), datetime.date(2018, 10, 1), datetime.date(2018, 10, 2),
                datetime.date(2018, 10, 3), datetime.date(2018, 10, 4), datetime.date(2018, 10, 5),
                datetime.date(2018, 12, 31), datetime.date(2019, 1, 1), datetime.date(2019, 2, 4),
                datetime.date(2019, 2, 5),  datetime.date(2019, 2, 6), datetime.date(2019, 2, 7),
                datetime.date(2019, 2, 8),  datetime.date(2019, 4, 5), datetime.date(2019, 5, 1),
                datetime.date(2019, 5, 2),  datetime.date(2019, 5, 3), datetime.date(2019, 6, 7),
                datetime.date(2019, 9, 13), datetime.date(2019, 10, 1), datetime.date(2019, 10, 2),
                datetime.date(2019, 10, 3), datetime.date(2019, 10, 4), datetime.date(2019, 10, 7),
                datetime.date(2020, 1, 1),  datetime.date(2020, 1, 24), datetime.date(2020,1, 27),
                datetime.date(2020, 1, 28), datetime.date(2020, 1, 29), datetime.date(2020, 1, 30),
                datetime.date(2020, 1, 31),
                datetime.date(2020, 4, 6),  datetime.date(2020, 5, 1), datetime.date(2020, 5, 4),
                datetime.date(2020, 5, 5),  datetime.date(2020, 6, 25), datetime.date(2020, 6, 26),
                datetime.date(2020, 10, 1), datetime.date(2020, 10, 2), datetime.date(2020, 10, 5), 
                datetime.date(2020, 10, 6), datetime.date(2020, 10, 7), datetime.date(2020, 10, 8),
                datetime.date(2021, 1, 1), datetime.date(2021, 2, 11), datetime.date(2021, 2, 12),
                datetime.date(2021, 2, 15), datetime.date(2021, 2, 16), datetime.date(2021, 2, 17),
                datetime.date(2021, 4, 5), datetime.date(2021, 5, 3), datetime.date(2021, 5, 4),
                datetime.date(2021, 5, 5), datetime.date(2021, 6, 12), datetime.date(2021, 6, 13),
                datetime.date(2021, 6, 14), datetime.date(2021, 9, 20), datetime.date(2021, 9, 21),
                datetime.date(2021, 10, 1), datetime.date(2021, 10, 4), datetime.date(2021, 10, 5),
                datetime.date(2021, 10, 6), datetime.date(2021, 10, 7),
                datetime.date(2022, 1, 3), datetime.date(2022, 1, 31), datetime.date(2022, 2, 1),
                datetime.date(2022, 2, 2), datetime.date(2022, 2, 3), datetime.date(2022, 2, 4),
                datetime.date(2022, 4, 4), datetime.date(2022, 4, 5), datetime.date(2022, 5, 2),
                datetime.date(2022, 5, 3), datetime.date(2022, 5, 4), datetime.date(2022, 6, 3), 
                datetime.date(2022, 9, 12), datetime.date(2022, 10, 3), datetime.date(2022, 10, 4),
                datetime.date(2022, 10, 5), datetime.date(2022, 10, 6), datetime.date(2022, 10, 7),
                ]

PLIO_Holidays = [datetime.date(2014, 1, 1),
                datetime.date(2015, 1, 1),
                datetime.date(2016, 1, 1),
                datetime.date(2017, 1, 2),
                datetime.date(2018, 1, 1), datetime.date(2018,12,24), datetime.date(2018,12,25), datetime.date(2018,12,31),
                datetime.date(2019, 1, 1), datetime.date(2019,2,5), datetime.date(2019,2,6),
                datetime.date(2019, 4,19), datetime.date(2019,12,25),
                datetime.date(2020, 1, 1), datetime.date(2020,1,27), datetime.date(2020,5,1),
                datetime.date(2020, 5, 7), datetime.date(2020,5,25), datetime.date(2020,7,31), datetime.date(2020,8,10), datetime.date(2020, 12, 25),
                datetime.date(2021, 1, 1), datetime.date(2021,2,12), datetime.date(2021,2,13),
                datetime.date(2021, 4, 2), datetime.date(2021,5,26), datetime.date(2021,8,9), datetime.date(2021,11,4), datetime.date(2021,12,25),
                ]

WASDE_Dates = [datetime.date(2010, 1, 12), datetime.date(2010, 2, 9), datetime.date(2010, 3, 10),
               datetime.date(2010, 4, 9), datetime.date(2010, 5, 11), datetime.date(2010, 6, 10),
               datetime.date(2010, 7, 9), datetime.date(2010, 8, 12), datetime.date(2010, 9, 10),
               datetime.date(2010, 10, 8), datetime.date(2010, 11, 9), datetime.date(2010, 12, 10),
               datetime.date(2011, 1, 12), datetime.date(2011, 2, 9), datetime.date(2011, 3, 10),
               datetime.date(2011, 4, 8), datetime.date(2011, 5, 11), datetime.date(2011, 6, 9),
               datetime.date(2011, 7, 12), datetime.date(2011, 8, 11), datetime.date(2011, 9, 12),
               datetime.date(2011, 10, 12), datetime.date(2011, 11, 9), datetime.date(2011, 12, 9),
               datetime.date(2012, 1, 12), datetime.date(2012, 2, 9), datetime.date(2012, 3, 9),
               datetime.date(2012, 4, 10), datetime.date(2012, 5, 10), datetime.date(2012, 6, 12),
               datetime.date(2012, 7, 11), datetime.date(2012, 8, 10), datetime.date(2012, 9, 12),
               datetime.date(2012, 10, 11), datetime.date(2012, 11, 9), datetime.date(2012, 12, 11),
               datetime.date(2013, 1, 11), datetime.date(2013, 2, 8), datetime.date(2013, 3, 8),
               datetime.date(2013, 4, 10), datetime.date(2013, 5, 10), datetime.date(2013, 6, 12),
               datetime.date(2013, 7, 11), datetime.date(2013, 8, 12), datetime.date(2013, 9, 12),
               datetime.date(2013, 11, 8), datetime.date(2013, 12, 10), datetime.date(2014, 1, 10),
               datetime.date(2014, 2, 10), datetime.date(2014, 3, 10), datetime.date(2014, 4, 9),
               datetime.date(2014, 5, 9), datetime.date(2014, 6, 11), datetime.date(2014, 7, 11),
               datetime.date(2014, 8, 12), datetime.date(2014, 9, 11), datetime.date(2014, 10, 10),
               datetime.date(2014, 11, 10), datetime.date(2014, 12, 10), datetime.date(2015, 1, 12),
               datetime.date(2015, 2, 10), datetime.date(2015, 3, 10), datetime.date(2015, 4, 9),
               datetime.date(2015, 5, 12), datetime.date(2015, 6, 10), datetime.date(2015, 7, 10),
               datetime.date(2015, 8, 12), datetime.date(2015, 9, 11), datetime.date(2015, 10, 9),
               datetime.date(2015, 11, 10), datetime.date(2015, 12, 9), datetime.date(2016, 1, 12),
               datetime.date(2016, 2, 9), datetime.date(2016, 3, 9), datetime.date(2016, 4, 12),
               datetime.date(2016, 5, 10), datetime.date(2016, 6, 10), datetime.date(2016, 7, 12),
               datetime.date(2016, 8, 12), datetime.date(2016, 9, 12), datetime.date(2016, 10, 12),
               datetime.date(2016, 11, 9), datetime.date(2016, 12, 9), datetime.date(2017, 1, 12),
               datetime.date(2017, 2, 9), datetime.date(2017, 3, 9), datetime.date(2017, 4, 11),
               datetime.date(2017, 5, 10), datetime.date(2017, 6, 9), datetime.date(2017, 7, 12),
               datetime.date(2017, 8, 10), datetime.date(2017, 9, 12), datetime.date(2017, 10, 12),
               datetime.date(2017, 11, 9), datetime.date(2017, 12, 12)]


Holiday_Map = { 'CHN': CHN_Holidays, 'PLIO': PLIO_Holidays}

product_code = {'SHFE': ['cu', 'cu_Opt', 'al', 'zn', 'al_Opt', 'zn_Opt', 'pb', 'wr', 'rb', 'fu', 'ru', 'ru_Opt', \
                         'bu', 'hc', 'ag', 'au', 'au_Opt', 'sn', 'ni', 'sp', 'ss'],
                'CFFEX': ['IF', 'TF', 'IO_Opt', 'T', 'TS', 'IH', 'IC', 'IM', 'MO_Opt',],
                'DCE': ['c', 'c_Opt', 'cs', 'j', 'jd', 'a', 'a_Opt', 'b', 'b_Opt','m', 'm_Opt', 'y', 'y_Opt', 'p', 'p_Opt',\
                        'l', 'v', 'pp', 'l_Opt', 'v_Opt', 'pp_Opt', \
                        'jm', 'i', 'i_Opt', 'fb', 'bb', 'eg', 'rr', 'eb', 'pg', 'pg_Opt', 'lh'],
                'CZCE': ['ER', 'RO', 'WS', 'WT', 'WH', 'PM', 'CF', 'CF_Opt', 'CY', 'SR', 'SR_Opt', \
                         'TA', 'TA_Opt', 'OI', 'OI_Opt', 'RI', 'ME', 'FG', 'RS', 'RM', 'RM_Opt', 'TC', \
                         'JR', 'LR', 'MA', 'MA_Opt', 'SM', 'SF', 'ZC', 'ZC_Opt', 'AP', 'CJ', 'UR', 'SA', 'PF', 'PK', 'PK_Opt'],
                'INE': ['sc', 'nr', 'lu', 'bc'],
                'SGX': ['fef', 'iolp', 'iac', 'm65f',], \
                'LME': ['lsc', 'lsr', 'lhc',], \
                'NYMEX': ['nhr', ],}

CHN_Stock_Exch = {
    'SSE': ["000300", "510180", "510050", "11000011", "11000016", "11000021", "11000026", "000002", "000003", "000004",
            "000005", "000006", "11000031", "11000036", "10000036"],
    'SZE': ['399001', '399004', '399007']}

option_market_products = ['m_Opt', 'c_Opt', 'a_Opt', 'b_Opt', 'y_Opt', 'p_Opt', \
    'i_Opt', 'pg_Opt', 'l_Opt', 'pp_Opt', 'v_Opt', \
    'OI_Opt', 'PK_Opt', 'SR_Opt', 'CF_Opt', 'TA_Opt', 'MA_Opt', 'RM_Opt', 'ZC_Opt',\
    'cu_Opt', 'al_Opt', 'zn_Opt', 'ru_Opt', 'au_Opt', \
    'ETF_Opt', 'IO_Opt', 'MO_Opt',]

night_session_markets = {'cu': 1,
                         'cu_Opt': 1,
                         'bc': 1,
                         'al': 1,
                         'zn': 1,
                         'al_Opt': 1,
                         'zn_Opt': 1,
                         'pb': 1,
                         'rb': 3,
                         'hc': 3,
                         'sp': 3,
                         'bu': 3,
                         'sn': 1,
                         'ni': 1,
                         'ss': 1,
                         'ag': 2,
                         'au': 2,
                         'au_Opt': 2,
                         'ru': 3,
                         'nr': 3,
                         'ru_Opt': 3,
                         'p': 3,
                         'p_Opt': 3,
                         'j': 3,
                         'a': 3,
                         'b': 3,
                         'a_Opt': 3,
                         'b_Opt': 3,
                         'm': 3,
                         'm_Opt': 3,
                         'y': 3,
                         'y_Opt': 3,
                         'jm': 3,
                         'i': 3,
                         'i_Opt': 3,
                         'l': 3,
                         'l_Opt': 3,
                         'v': 3,
                         'v_Opt': 3,
                         'pp': 3,
                         'pp_Opt': 3,
                         'eg': 3,
                         'eb': 3,
                         'rr': 3,
                         'c': 3,
                         'c_Opt': 3,
                         'pg': 3,
                         'pg_Opt': 3,
                         'cs': 3,
                         'CF': 3,
                         'CF_Opt': 3,
                         'CY': 3,
                         'SR': 3,
                         'SR_Opt': 3,
                         'RM': 3,
                         'RM_Opt': 3,
                         'TA': 3,
                         'TA_Opt': 3,
                         'MA': 3,
                         'MA_Opt': 3,
                         'ME': 3,
                         'SA': 3,
                         'OI': 3,
                         'OI_Opt': 3,
                         'TC': 3,
                         'ZC': 3,
                         'ZC_Opt': 3,
                         'FG': 3,
                         'PF': 3,
                         'fu': 3,
                         'lu': 3,
                         'sc': 2,
                         'sc_Opt': 2,
                         }

night_trading_hrs = {1: (300, 700),
                     2: (300, 830),
                     3: (300, 500),
                     4: (300, 530),
                     }

day_split_dict = {'s1': [300, 2115],
                  's2': [300, 1500, 2115],
                  's3': [300, 1500, 1900, 2115],
                  's4': [300, 1500, 1630, 1900, 2115],}
                  
bar_shift_table1 = {1: [(1630, -15), (1800, -120)],
                    2: [(1500, -390), (1630, -15), (1800, -120)],
                    3: [(1630, -15), (1800, -120)],
                    4: [(1500, -570), (1630, -15), (1800, -120)],
                    }

product_class_map = {'zn': ('Ind', "BaseMetal"), 
                   'zn_Opt': ('Ind', "BaseMetal"),
                   'cu': ('Ind', "BaseMetal"),
                   'cu_Opt': ('Ind', "BaseMetal"),
                   'bc': ('Ind', "BaseMetal"),
                   'ru': ('Ind', 'NonFerrous'),
                   'nr': ('Ind', 'NonFerrous'),
                   'ru_Opt': ('Ind', 'NonFerrous'),
                   'rb': ('Ind', "Ferrous"),
                   'fu': ('Ind', "Petro"),
                   'al': ('Ind', "BaseMetal"),
                   'al_Opt': ('Ind', "BaseMetal"),
                   'au': ('Macro', 'PreciousMetal'),
                   'au_Opt': ('Macro', 'PreciousMetal'),
                   'wr': ('Ind', "Ferrous"),
                   'pb': ('Ind', "BaseMetal"),
                   'ag': ('Macro', 'PreciousMetal'),
                   'bu': ('Ind', "Petro"),
                   'hc': ('Ind', "Ferrous"),
                   'sp': ('Ags', 'Soft'),
                   'ni': ('Ind', "BaseMetal"),
                   'sn': ('Ind', "BaseMetal"),
                   'ss': ('Ind', "BaseMetal"),
                   'WH': ('Ags', 'Grain'),
                   'PM': ('Ags', 'Grain'),
                   'CF': ('Ags', 'Soft'),
                   'CF_Opt': ('Ags', 'Soft'),
                   'CY': ('Ags', 'Soft'),
                   'SR': ('Ags', 'Soft'),
                   'SR_Opt': ('Ags', 'Soft'),
                   'TA': ('Ind', "Petro"),
                   'TA_Opt': ('Ind', "Petro"),
                   'OI': ('Ags', 'Grain'),
                   'OI_Opt': ('Ags', 'Grain'),
                   'RI': ('Ags', 'Grain'),
                   'ME': ('Ind', "Petro"),
                   'MA': ('Ind', "Petro"),
                   'MA_Opt': ('Ind', "Petro"),
                   'FG': ('Ind', "NonFerrous"),
                   'RS': ('Ags', 'Grain'),
                   'RM': ('Ags', 'Grain'),
                   'RM_Opt': ('Ags', 'Grain'),
                   'TC': ('Ind', "NonFerrous"),
                   'ZC': ('Ind', "NonFerrous"),
                   'ZC_Opt': ('Ind', "NonFerrous"),
                   'JR': ('Ags', 'Grain'),
                   'LR': ('Ags', 'Grain'),
                   'SM': ('Ind', "NonFerrous"),
                   'SF': ('Ind', "NonFerrous"),
                   'AP': ('Ags', 'Soft'),
                   'CJ': ('Ags', 'Soft'),
                   'UR': ('Ags', 'Soft'),
                   'SA': ('Ags', 'Soft'),
                   'PF': ('Ags', 'Soft'),
                   'PK': ('Ags', 'Soft'),
                   'PK_Opt': ('Ags', 'Soft'),
                   'c': ('Ags', 'Grain'),
                   'c_Opt': ('Ags', 'Grain'),
                   'cs': ('Ags', 'Grain'),
                   'j': ('Ind', "Ferrous"),
                   'jd': ('Ags', 'Soft'),
                   'lh': ('Ags', 'Soft'),
                   'a': ('Ags', 'Grain'),
                   'a_Opt': ('Ags', 'Grain'),
                   'b': ('Ags', 'Grain'),
                   'b_Opt': ('Ags', 'Grain'),
                   'm': ('Ags', 'Grain'),
                   'm_Opt': ('Ags', 'Grain'),
                   'y': ('Ags', 'Grain'),
                   'y_Opt': ('Ags', 'Grain'),
                   'p': ('Ags', 'Grain'),
                   'p_Opt': ('Ags', 'Grain'),
                   'l': ('Ind', "Petro"),
                   'v': ('Ind', "Petro"),
                   'pp': ('Ind', "Petro"),
                   'l_Opt': ('Ind', "Petro"),
                   'v_Opt': ('Ind', "Petro"),
                   'pp_Opt': ('Ind', "Petro"),
                   'eg': ('Ind', "Petro"),
                   'eb': ('Ind', "Petro"),
                   'pg': ('Ind', "Petro"),
                   'pg_Opt': ('Ind', "Petro"),
                   'rr': ('Ags', "Grain"),
                   'jm': ('Ind', "Ferrous"),
                   'i': ('Ind', "Ferrous"),
                   'i_Opt': ('Ind', "Ferrous"),
                   'fb': ('Ind', "NonFerrous"),
                   'bb': ('Ind', "NonFerrous"),
                   'IF': ('Macro', 'Equity'),
                   'IH': ('Macro', 'Equity'),
                   'IC': ('Macro', 'Equity'),
                   'IM': ('Macro', 'Equity'),
                   'MO_Opt': ('Macro', 'Equity'),
                   'TF': ('Macro', 'Bond'),
                   'T': ('Macro', 'Bond'),
                   'TS': ('Macro', 'Bond'),
                   'IO_Opt': ('Macro', 'Equity'),
                   'sc': ('Ind', "Petro"),
                   'sc_Opt': ('Ind', "Petro"),
                   'lu': ('Ind', "Petro"),
                   }

product_lotsize = {'zn': 5,
                   'zn_Opt': 5,
                   'cu': 5,
                   'cu_Opt': 5,
                   'bc': 5,
                   'ru': 10,
                   'nr': 10,
                   'ru_Opt': 10,
                   'rb': 10,
                   'fu': 10,
                   'al': 5,
                   'al_Opt': 5,
                   'au': 1000,
                   'au_Opt': 1000,
                   'wr': 10,
                   'pb': 5,
                   'ag': 15,
                   'bu': 10,
                   'hc': 10,
                   'sp': 10,
                   'ni': 1,
                   'sn': 1,
                   'ss': 5,
                   'WH': 20,
                   'PM': 50,
                   'CF': 5,
                   'CF_Opt': 5,
                   'CY': 5,
                   'SR': 10,
                   'SR_Opt': 10,
                   'TA': 5,
                   'TA_Opt': 5,
                   'OI': 10,
                   'OI_Opt': 10,
                   'RI': 20,
                   'ME': 50,
                   'MA': 10,
                   'MA_Opt': 10,
                   'FG': 20,
                   'RS': 10,
                   'RM': 10,
                   'RM_Opt': 10,
                   'TC': 200,
                   'ZC': 100,
                   'ZC_Opt': 100,
                   'JR': 20,
                   'LR': 20,
                   'SM': 5,
                   'SF': 5,
                   'AP': 10,
                   'CJ': 5,
                   'UR': 20,
                   'SA': 20,
                   'PF': 5,
                   'PK': 5,
                   'PK_Opt': 5,
                   'c': 10,
                   'c_Opt': 10,
                   'pg': 20,
                   'pg_Opt': 20,
                   'cs': 10,
                   'j': 100,
                   'jd': 10,
                   'a': 10,
                   'b': 10,
                   'a_Opt': 10,
                   'b_Opt': 10,
                   'm': 10,
                   'm_Opt': 10,
                   'y': 10,
                   'y_Opt': 10,
                   'p': 10,
                   'p_Opt': 10,
                   'l': 5,
                   'v': 5,
                   'pp': 5,
                   'l_Opt': 5,
                   'v_Opt': 5,
                   'pp_Opt': 5,
                   'eg': 10,
                   'eb': 5,
                   'rr': 10,
                   'jm': 60,
                   'i': 100,
                   'i_Opt': 100,
                   'lh': 16,
                   'fb': 500,
                   'bb': 500,
                   'IF': 300,
                   'IH': 300,
                   'IC': 200,
                   'IM': 200,
                   'TF': 10000,
                   'T': 10000,
                   'TS': 20000,
                   'IO_Opt': 100,
                   'MO_Opt': 100,
                   'sc': 1000,
                   'sc_Opt': 1000,
                   'lu': 10,
                   }

product_ticksize = {
                    'zn': 5,
                    'zn_Opt': 1,
                    'cu': 10,
                    'cu_Opt': 1,
                    'bc': 10,
                    'ru': 5,
                    'nr': 5,
                    'ru_Opt': 1,
                    'rb': 1,
                    'fu': 1,
                    'al': 5,
                    'al_Opt': 1,
                    'au': 0.02,
                    'au_Opt': 0.02,
                    'wr': 1,
                    'pb': 5,
                    'ag': 1,
                    'bu': 2,
                    'hc': 1,
                    'sp': 2,
                    'ss': 5.0,
                    'ni': 1.0,
                    'sn': 1.0,
                    'WH': 1,
                    'PM': 1,
                    'CF': 5,
                    'CF_Opt': 1,
                    'CY': 5,
                    'SR': 1,
                    'SR_Opt': 0.5,
                    'TA': 2,
                    'TA_Opt': 0.5,
                    'OI': 2,
                    'OI_Opt': 0.5,
                    'RI': 1,
                    'ME': 1,
                    'MA': 1,
                    'MA_Opt': 0.5,
                    'FG': 1,
                    'RS': 1,
                    'RM': 1,
                    'RM_Opt': 0.5,
                    'TC': 0.2,
                    'ZC': 0.2,
                    'ZC_Opt': 0.1,
                    'JR': 1,
                    'LR': 1,
                    'SF': 2,
                    'SM': 2,
                    'AP': 1,
                    'CJ': 5.0,
                    'UR': 1.0,
                    'SA': 1.0,
                    'PF': 2.0,
                    'PK': 2.0,
                    'PK_Opt': 0.5,
                    'c': 1,
                    'c_Opt': 0.5,
                    'j': 0.5,
                    'jd': 1,
                    'a': 1,
                    'b': 1,
                    'a_Opt': 0.5,
                    'b_Opt': 0.5,
                    'm': 1,
                    'm_Opt': 0.5,
                    'y': 2,
                    'y_Opt': 0.5,
                    'p': 2,
                    'p_Opt': 0.5,
                    'l': 1,
                    'v': 1,
                    'pp': 1,
                    'l_Opt': 0.5,
                    'v_Opt': 0.5,
                    'pp_Opt': 0.5,
                    'eg': 1,
                    'eb': 1.0,
                    'rr': 1,
                    'jm': 0.5,
                    'i': 0.5,
                    'i_Opt': 0.1,
                    'pg': 1.0,
                    'pg_Opt': 0.2,
                    'lh': 5.0,
                    'fb': 0.05,
                    'bb': 0.05,
                    'IF': 0.2,
                    'IH': 0.2,
                    'IC': 0.2,
                    'IM': 0.2,
                    'TF': 0.005,
                    'TS': 0.005,
                    'T': 0.005,
                    'IO_Opt': 0.2,
                    'MO_Opt': 0.2,
                    'sc': 0.1,
                    'sc_Opt': 0.05,
                    'lu': 1.0,
                    }


def xl2date(num):
    return datetime.date(1970,1,1) + datetime.timedelta(days = int(num) - 25569)


def date2xl(d):
    return (d - datetime.date(1970, 1, 1)).days + 25569.0


def datetime2xl(dt):
    t = dt - datetime.datetime(1970, 1, 1, 0, 0, 0)
    return 25569.0 + t.days + t.seconds / 60.0 / 60.0 / 24.0


def time2exp(opt_expiry, curr_time):
    curr_date = curr_time.date()
    exp_date = opt_expiry.date()
    if curr_time > opt_expiry:
        return 0.0
    elif exp_date < curr_date:
        return workdays.networkdays(curr_date, exp_date, CHN_Holidays) / BDAYS_PER_YEAR
    else:
        delta = opt_expiry - curr_time
        return (delta.hour * 3600 + delta.min * 60 + delta.second) / 3600.0 / 5.5 / BDAYS_PER_YEAR


def conv_expiry_date(curr_date, expiry, accrual = 'act365', hols = []):
    if expiry < curr_date:
        return 0.0
    year_conv = int(accrual[-3:])
    if year_conv >= 360:
        return float((expiry - curr_date).days + 1)/year_conv
    else:
        return workdays.networkdays(curr_date, expiry, hols) / float(year_conv)


def get_obj_by_name(obj_name):
    obj_split = obj_name.split('.')
    mod_str = '.'.join(obj_split[:-1])
    module = __import__(mod_str, fromlist=[obj_split[-1]])
    obj = getattr(module,obj_split[-1])
    return obj


def merge_dict(src_dict, dest_dict, w_src = 1, w_dest = 1):
    for key, value in list(src_dict.items()):
        if isinstance(value, dict):
            # get node or create one
            node = dest_dict.setdefault(key, {})
            merge_dict(value, node, w_src, w_dest)
        else:
            if key not in dest_dict:
                dest_dict[key] = 0.0
            dest_dict[key] = dest_dict[key] * w_dest + value * w_src
    return dest_dict


def min2time(min_id):
    return int((min_id // 100 - 6) % 24) / 24.0 + (min_id % 100) / 1440.0


def get_tick_id(dt):
    return ((dt.hour + 6) % 24) * 100000 + dt.minute * 1000 + dt.second * 10 + dt.microsecond // 100000


def get_min_id(dt):
    return ((dt.hour+6)%24)*100+dt.minute


def is_workday(d, calendar = '', we_cutoff = 5):
    return (d.weekday() < we_cutoff) and (d not in Holiday_Map.get(calendar, []))


def get_mkt_fxpair(fx1, fx2):
    ccy1 = fx1.upper()
    ccy2 = fx2.upper()
    if ccy1 == ccy2:
        return 0
    ccy_list = ['GBP', 'EUR', 'AUD', 'NZD', 'USD', 'CNY', 'JPY']
    rank1 = 100
    rank2 = 100
    if ccy1 in ccy_list:
        rank1 = ccy_list.index(ccy1)
    if ccy2 in ccy_list:
        rank2 = ccy_list.index(ccy2)
    if (rank1 > 99) or (rank2 > 99):
        direction = 0
    elif rank1 < rank2:
        direction = 1
    elif rank1 > rank2:
        direction = -1
    else:
        direction = 0
    return direction


def conv_fx_rate(ccy, reporting_ccy, fx_market):
    fx_pair = None
    if ccy == reporting_ccy:
        return 1.0
    fx_direction = get_mkt_fxpair(reporting_ccy, ccy)
    fx_rate = 1.0
    if fx_direction > 0:
        fx_pair = '/'.join([reporting_ccy, ccy])
        fx_rate = 1 / fx_market[fx_pair][0][2]
    elif fx_direction < 0:
        fx_pair = '/'.join([ccy, reporting_ccy])
        fx_rate = fx_market[fx_pair][0][2]
    return fx_rate


def filter_main_cont(sdate, filter=False):
    insts, prods = dbaccess.load_alive_cont(sdate)
    if not filter:
        return insts
    main_cont = {}
    prods = [pc for pc in prods if '_Opt' not in pc]
    for pc in prods:
        main_cont[pc], exch = dbaccess.prod_main_cont_exch(pc)
    main_insts = []
    for inst in insts:
        pc = inst2product(inst)
        if pc not in prods:
            continue
        mth = int(inst[-2:])
        if mth in main_cont[pc]:
            main_insts.append(inst)
    return main_insts


def get_first_day_of_month(t_date):
    return t_date.replace(day=1)
    

def trading_hours(product, exch):
    if product in ['sc']:
        hrs = [(1500, 1730), (1930, 2100)]
    else:
        hrs = [(1500, 1615), (1630, 1730), (1930, 2100)]

    if exch in ['SSE', 'SZE']:
        hrs = [(1530, 1730), (1900, 2100)]
    elif product in ['TF', 'T', 'TS']:
        hrs = [(1530, 1730), (1900, 2115)]
    elif product in ['IF', 'IH', 'IC', 'IM', 'IO_Opt', 'MO_Opt',]:
        hrs = [(1530, 1730), (1900, 2100)]
    else:
        if product in night_session_markets:
            night_idx = night_session_markets[product]
            hrs = [night_trading_hrs[night_idx]] + hrs
    return hrs


def check_trading_range(tick_id, product, exch, tick_buffer = 0):
    in_range = False
    hrs = trading_hours(product, exch)    
    for ptime in hrs:
        if (tick_id>=ptime[0]*1000 - tick_buffer) and (tick_id < ptime[1] *1000 + tick_buffer):
            in_range = True
            break
    return in_range


def spreadinst2underlying(inst_name):
    spread_keys = inst_name.split(' ')
    instIDs = spread_keys[1].split('&')
    units = [1, -1]
    return (instIDs, units)


def instID_adjust(instID, exch, ref_date):
    adj_inst = instID
    if exch == 'CZCE':
        if len(instID) == 5:
            if ref_date < datetime.date(2015, 1, 1):
                if (int(instID[2]) <= 4) and ref_date >= datetime.datetime.date(2005, 1, 1):
                    adj_inst = instID[:2] + '1' + instID[-3:]
                else:
                    adj_inst = instID[:2] + '0' + instID[-3:]
    elif exch == 'DCE':
        if (len(instID) == 4) and ref_date<datetime.date(2010, 1, 1):
            adj_inst = instID[:1] + '0' + instID[-3:]
    return adj_inst


def inst2product(inst):
    if inst[4].isalpha():
        key = inst[:5]
    elif inst[3].isalpha():
        key = inst[:4]
    elif inst[2].isalpha():
        key = inst[:3]
    elif inst[1].isalpha():
        key = inst[:2]
    else:
        key = inst[:1]
    if len(inst) > 8 and (('C' in inst[5:]) or ('P' in inst[5:])):
        key = key + '_Opt'
    return key


def inst2contmth(instID):
    exch = inst2exch(instID)
    if exch == 'CZCE':
        if len(instID) == 6:
            cont_mth = 200000 + int(instID[-4:])
        elif int(instID[-3]) >= 5:
            cont_mth = 201000 + int(instID[-3:])
        else:
            cont_mth = 202000 + int(instID[-3:])
    else:
        cont_mth = 200000 + int(instID[-4:])
    return cont_mth


def inst2cont(instID):
    cont_mth = inst2contmth(instID)
    year = int(cont_mth /100)
    mth = cont_mth % 100
    return datetime.date(year, mth, 1)


def inst2exch(inst):
    if inst.isdigit():
        return "SSE"
    prod = inst2product(inst)
    return prod2exch(prod)


def prod2exch(prod):
    for exch in list(product_code.keys()):
        if prod in product_code[exch]:
            return exch
    return "NA"


def inst_to_exch(inst):
    key = inst2product(inst)
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    cursor = cnx.cursor()
    stmt = "select exchange from trade_products where product_code='{prod}' ".format(prod=key)
    cursor.execute(stmt)
    out = [exchange for exchange in cursor]
    cnx.close()
    return str(out[0][0])


def get_hols_by_exch(exch):
    hols = []
    if exch in ['DCE', 'CFFEX', 'CZCE', 'SHFE', 'INE', 'SSE', 'SZSE']:
        hols = CHN_Holidays
    return hols


def get_option_map(products):
    option_map = {}
    for under in products:
        for cont_mth in products[under]:
            for strike in products[under][cont_mth]:
                for otype in ['C', 'P']:
                    key = (str(under), cont_mth, otype, strike)
                    option_map[key] = get_opt_name(under, otype, strike)
    return option_map


def get_opt_name(fut_inst, otype, strike):
    cont_mth = inst2contmth(fut_inst)
    key = (str(fut_inst), cont_mth, otype, strike)
    instID = fut_inst
    exch = inst2exch(instID)
    if instID[:2] == "IF":
        instID = instID.replace('IF', 'IO')
    elif instID[:2] == "IM":
        instID = instID.replace('IM', 'MO')
    if exch in ['CZCE', 'SHFE']:
        instID = instID + otype + str(int(strike))
    else:
        instID = instID + '-' + otype + '-' + str(int(strike))
    return instID


def get_opt_expiry(fut_inst, cont_mth, exch=''):
    cont_yr = int(cont_mth / 100)
    cont_mth = cont_mth % 100
    expiry_month = datetime.date(cont_yr, cont_mth, 1)
    wkday = expiry_month.weekday()
    product = ''.join(i for i in fut_inst[:5] if not i.isdigit())
    if fut_inst[:6].isdigit():
        nbweeks = 4
        if wkday <= 2:
            nbweeks = 3
        expiry = expiry_month + datetime.timedelta(days=nbweeks * 7 - wkday + 1)
        expiry = workdays.workday(expiry, 1, CHN_Holidays)
    elif product in ['IF', 'IM',]:
        nbweeks = 2
        if wkday >= 5:
            nbweeks = 3
        expiry = expiry_month + datetime.timedelta(days=nbweeks * 7 - wkday + 3)
        expiry = workdays.workday(expiry, 1, CHN_Holidays)
    elif product in ['SR', 'CF', 'TA', 'MA', 'RM', 'PF', 'OI', 'ZC']:
        if cont_mth > 1:
            expiry_month = datetime.date(cont_yr, cont_mth - 1, 1)
        else:
            expiry_month = datetime.date(cont_yr - 1, 12, 1)
        expiry = workdays.workday(workdays.workday(expiry_month, -1, CHN_Holidays), 3, CHN_Holidays)
    elif product in ['m', 'c', 'i', 'pg', 'l', 'pp', 'v', 'a', 'b', 'p', 'y']:
        if cont_mth > 1:
            expiry_month = datetime.date(cont_yr, cont_mth - 1, 1) + datetime.timedelta(days=-1)
        else:
            expiry_month = datetime.date(cont_yr - 1, 11, 30)
        expiry = workdays.workday(expiry_month, 5, CHN_Holidays)
    elif product in ['cu', 'ru', 'au', 'al', 'zn']:
        expiry = workdays.workday(expiry_month, -5, CHN_Holidays)
    elif product in ['fef']:
        if cont_mth < 12:
            expiry_month = datetime.date(cont_yr, cont_mth + 1, 1)
        else:
            expiry_month = datetime.date(cont_yr + 1, 1, 1)
        expiry = workdays.workday(expiry_month, -1, PLIO_Holidays)
    return datetime.datetime.combine(expiry, datetime.time(15, 0))


def cont_expiry_list(prodcode, start_date, end_date, roll_rule='-0d'):
    cont_mth, exch = dbaccess.prod_main_cont_exch(prodcode)
    hols = get_hols_by_exch(exch)
    contlist, tenor_list = contract_range(prodcode, exch, cont_mth, start_date, day_shift(end_date, '12m', hols))
    exp_dates = [day_shift(contract_expiry(cont), roll_rule, hols) for cont in contlist]
    return contlist, exp_dates, tenor_list


def default_next_main_contract(inst, start_date=datetime.date(2006, 1, 1), end_date=datetime.date.today()):
    prodcode = inst2product(inst)
    contlist, _, _ = cont_expiry_list(prodcode, start_date, end_date)
    return [c for c in contlist if c > inst][0]


def nearby(prodcode, n = 1, start_date = None, end_date = None, roll_rule = '-20b', freq = 'd', shift_mode = 0, database = None, dbtbl_prefix = ''):
    contlist, exp_dates, _ = cont_expiry_list(prodcode, start_date, end_date, roll_rule)
    if prodcode == 'sn':
        if 'sn2001' in contlist:
            idx = contlist.index('sn2001')
            exp_dates[idx] = max(datetime.date(2019,12,26), exp_dates[idx])
    elif prodcode == 'ni':
        if ('ni1905' in contlist) and ('ni1901' in contlist):
            idx = contlist.index('ni1901')
            exp_dates[idx] = max(datetime.date(2018,12,27), exp_dates[idx])
    sdate = start_date
    dbconf = copy.deepcopy(dbaccess.dbconfig)
    if database:
        dbconf['database'] = database
    cnx = dbaccess.connect(**dbconf)
    df = pd.DataFrame()
    for idx, exp in enumerate(exp_dates):
        if exp < start_date:
            continue
        elif sdate > end_date:
            break
        nb_cont = contlist[idx + n - 1]
        if freq == 'd':
            new_df = dbaccess.load_daily_data_to_df(cnx, dbtbl_prefix + 'fut_daily', nb_cont, sdate, min(exp, end_date))
        else:
            minid_start = 1500
            minid_end = 2114
            if prodcode in night_session_markets:
                minid_start = 300
            new_df = dbaccess.load_min_data_to_df(cnx, dbtbl_prefix + 'fut_min', nb_cont, sdate, min(exp, end_date), minid_start,
                                                  minid_end)
        if len(new_df) > 0:
            new_df['contract'] = nb_cont
            new_df['shift'] = 0.0
        else:
            print("continuous contract stopped at %s for start = %s, expiry= %s" % (nb_cont, sdate, exp))
            continue
        if len(df) > 0 and shift_mode > 0:
            if isinstance(df.index[-1], datetime.datetime):
                last_date = df.index[-1].date()
            else:
                last_date = df.index[-1]
            tmp_df = dbaccess.load_daily_data_to_df(cnx, dbtbl_prefix + 'fut_daily', nb_cont, last_date, last_date)
            if shift_mode == 1:
                shift = tmp_df['close'][-1] - df['close'][-1]
                df['shift'] = df['shift'] + shift
                for ticker in ['open', 'high', 'low', 'close']:
                    df[ticker] = df[ticker] + shift
            else:
                shift = float(tmp_df['close'][-1])/float(df['close'][-1])
                df['shift'] = df['shift'] + math.log(shift)
                for ticker in ['open', 'high', 'low', 'close']:
                    df[ticker] = df[ticker] * shift
        df = df.append(new_df)
        sdate = exp + datetime.timedelta(days=1)
    return df


def rolling_hist_data(product, n, start_date, end_date, cont_roll, freq, win_roll='-20b', database='hist_data'):
    if start_date > end_date:
        return None
    cnx = dbaccess.connect(**dbaccess.dbconfig)
    cursor = cnx.cursor()
    stmt = "select exchange, contract from trade_products where product_code='{prod}' ".format(prod=product)
    cursor.execute(stmt)
    out = [(exchange, contract) for (exchange, contract) in cursor]
    exch = str(out[0][0])
    hols = get_hols_by_exch(exch)
    cont = str(out[0][1])
    cont_mth = [month_code_map[c] for c in cont]
    cnx.close()
    contlist, _ = contract_range(product, exch, cont_mth, start_date, end_date)
    exp_dates = [day_shift(contract_expiry(cont), cont_roll, hols) for cont in contlist]
    sdate = start_date
    all_data = {}
    i = 0
    dbconfig = copy.deepcopy(dbaccess.dbconfig)
    dbconfig['database'] = database
    cnx = dbaccess.connect(**dbconfig)
    for idx, exp in enumerate(exp_dates):
        if exp < start_date:
            continue
        elif sdate > end_date:
            break
        nb_cont = contlist[idx + n - 1]
        if freq == 'd':
            df = dbaccess.load_daily_data_to_df(cnx, 'fut_daily', nb_cont, day_shift(sdate, win_roll, hols),
                                                min(exp, end_date))
        else:
            df = dbaccess.load_min_data_to_df(cnx, 'fut_min', nb_cont, day_shift(sdate, win_roll, hols), min(exp, end_date))
        all_data[i] = {'contract': nb_cont, 'data': df}
        i += 1
        sdate = min(exp, end_date) + datetime.timedelta(days=1)
    cnx.close()
    return all_data


def day_shift(d, roll_rule, hols = []):
    if 'b' in roll_rule:
        days = int(roll_rule[:-1])
        shft_day = workdays.workday(d, days, hols)
    elif 'm' in roll_rule:
        mths = int(roll_rule[:-1])
        shft_day = d + relativedelta(months=mths)
    elif 'd' in roll_rule:
        days = int(roll_rule[:-1])
        shft_day = d + datetime.timedelta(days=days)
    elif 'y' in roll_rule:
        years = int(roll_rule[:-1])
        shft_day = d + relativedelta(years=years)
    elif 'w' in roll_rule:
        weeks = int(roll_rule[:-1])
        shft_day = d + relativedelta(weeks=weeks)
    elif 'MEND' in roll_rule:
        mths = int(roll_rule[:-4]) + 1
        shft_day = d + relativedelta(months=mths)
        shft_day = shft_day.replace(day=1)
        shft_day = shft_day - datetime.timedelta(days=1)
    return shft_day


def process_min_id(df, adj_datetime = False, hols = CHN_Holidays):
    df['min_id'] = df['datetime'].apply(lambda x: ((x.hour + 6) % 24)*100 + x.minute)
    flag = df['min_id'] >= 1000
    df.loc[flag, 'date'] = df['datetime'][flag].apply(lambda x: x.date())
    df['date'] = df['date'].fillna(method = 'bfill')
    flag = pd.isnull(df['date'])
    df.loc[flag,'date'] = df['datetime'][flag].apply(lambda x: day_shift(x.date(),'1b', hols))
    if adj_datetime:
        df['datetime'] = df.apply(lambda x: datetime.datetime.combine(x['date'], x['datetime'].time()), axis = 1)
    return df


def tenor_to_expiry(tenor_label, prod_code = 'fef'):
    exch = prod2exch(prod_code)
    if 'Cal' in tenor_label:
        ten_str = tenor_label.split(' ')
        year = 2000 + int(ten_str[1])
        return datetime.date(year, 12, 31)
    if 'Q' in tenor_label:
        ten_str = tenor_label.replace("'", " ").split(" ")
        year = 2000 + int(ten_str[1])
        mth = int(ten_str[0][-1])*3
        return datetime.date(year, mth, calendar.monthrange(year, mth)[1])
    else:
        cont_date = datetime.datetime.strptime(tenor_label, "%Y-%m-%d").date()
        return cont_date_expiry(cont_date, prod_code, exch)

def contract_expiry(cont, hols=CHN_Holidays):
    if cont == 'sn2005':
        return datetime.date(2020, 2, 12)
    if type(hols) == list:
        prod_code = inst2product(cont)
        exch = prod2exch(prod_code)
        mth = int(cont[-2:])
        if exch == 'CZCE' and len(cont) == 5:
            if int(cont[-3:-2])>=5:
                yr = 2010 + int(cont[-3:-2])
            else:
                yr = 2020 + int(cont[-3:-2])
        else:
            yr = 2000 + int(cont[-4:-2])
        cont_date = datetime.date(yr, mth, 1)
        expiry = cont_date_expiry(cont_date, prod_code, exch)
    else:
        cnx = dbaccess.connect(**dbaccess.dbconfig)
        cursor = cnx.cursor()
        stmt = "select expiry from contract_list where instID='{inst}' ".format(inst=cont)
        cursor.execute(stmt)
        out = [exp for exp in cursor]
        if len(out) > 0:
            expiry = out[0][0]
            if isinstance(expiry, str):
                expiry = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()
        else:
            expiry = contract_expiry(cont, CHN_Holidays)
        cnx.close()
    return expiry


def cont_date_expiry(cont_date, prod_code, exch):
    hols = CHN_Holidays
    yr = cont_date.year
    mth = cont_date.month
    if prod_code in ['fu', 'sc', 'lu']:
        expiry = workdays.workday(cont_date, -1, hols)
    elif prod_code in ['lh', 'jd', 'pg', 'eb', 'eg']:
        expiry = workdays.workday(cont_date + relativedelta(months=1), -4, hols)
    elif prod_code in ['IF', 'IH', 'IC', 'IM']:
        wkday = cont_date.weekday()
        expiry = cont_date + datetime.timedelta(days=13+(11-wkday)%7)
        expiry = workdays.workday(expiry, 1, CHN_Holidays)
    elif prod_code in ['T', 'TF', 'TS',]:
        wkday = cont_date.weekday()
        expiry = cont_date + datetime.timedelta(days=6+(11-wkday)%7)
        expiry = workdays.workday(expiry, 1, CHN_Holidays)                    
    elif prod_code in ['ZC', 'TC']:
        expiry = workdays.workday(cont_date - datetime.timedelta(days=1), 5, hols)   
    elif exch in ['DCE', 'CZCE']:
        expiry = workdays.workday(cont_date - datetime.timedelta(days=1), 10, hols)    
    elif exch in ['SHFE', 'INE']:
        expiry = datetime.date(yr, mth, 14)
        expiry = workdays.workday(expiry, 1, CHN_Holidays)
    elif exch in ['SGX', 'LME', 'NYMEX', 'OTC']:
        expiry = workdays.workday(cont_date + relativedelta(months=1), -1, PLIO_Holidays)
    else:
        expiry = 0
    return expiry

def _contract_range(product, exch, cont_mth, start_date, end_date, tenor = '2y'):
    st_year = start_date.year
    cont_list = []
    tenor_list = []
    hols = get_hols_by_exch(exch)
    for yr in range(st_year, end_date.year + 2):
        for mth in range(1, 13):
            if (mth in cont_mth):
                cont_ten = datetime.date(yr, mth, 1)
                if (cont_ten >= start_date) and (cont_ten <= day_shift(end_date, tenor, hols)):
                    prod = product
                    if product == 'ZC' and cont_ten < datetime.date(2016, 5, 1):
                        prod = 'TC'
                    elif product == 'MA' and cont_ten <= datetime.date(2015, 5, 1):
                        prod = 'ME'
                        if cont_ten == datetime.date(2015, 5, 1):
                            mth = 6
                            prod = 'MA'
                    if exch == 'CZCE' and cont_ten >= datetime.date(2015, 1, 1):
                        contLabel = prod + "%01d" % (yr % 10) + "%02d" % mth
                    else:
                        contLabel = prod + "%02d" % (yr % 100) + "%02d" % mth
                    tenor_list.append(datetime.date(yr, mth, 1))
                    cont_list.append(contLabel)
    return cont_list, tenor_list


def contract_range(product, exch, cont_mth, start_date, end_date):
    product_cont_map = {'ni': [datetime.date(2019, 5, 1), [1, 5, 9]],
                        'sn': [datetime.date(2020, 5, 1), [1, 5, 9]],
                        'eg': [datetime.date(2020, 5, 1), [1, 5, 9]],
                        #'ZC': [datetime.date(2020, 9, 1), [1, 5, 9]],
                        }
    if product in product_cont_map:
        cont_list, tenor_list = _contract_range(product, exch, cont_mth, start_date, end_date)
        res_cont = []
        res_ten = []
        for cont, ten in zip(cont_list, tenor_list):
            if (ten <= product_cont_map[product][0]) and ten.month not in product_cont_map[product][1]:
                continue
            res_cont.append(cont)
            res_ten.append(ten)
        return res_cont, res_ten
    else:
        return _contract_range(product, exch, cont_mth, start_date, end_date)


def get_asset_tradehrs(asset):
    exch = 'SHFE'
    for ex in product_code:
        if asset in product_code[ex]:
            exch = ex
            break
    if asset in ['sc']:
        hrs = [(1500, 1730), (1930, 2100)]
    else:
        hrs = [(1500, 1615), (1630, 1730), (1930, 2100)]
    if (exch in ['SSE', 'SZE']) or (asset in ['IF', 'IC', 'IH', 'IM', 'IO_Opt', 'MO_Opt']):
        hrs = [(1530, 1730), (1900, 2100)]
    elif asset in ['TF', 'T', 'TS']:
        hrs = [(1515, 1730), (1900, 2115)]
    else:
        if asset in night_session_markets:
            night_idx = night_session_markets[asset]
            hrs = [night_trading_hrs[night_idx]] + hrs
    return hrs


def cleanup_dailydata(df, asset, index_col = 'date'):
    if index_col == 'date':
        xdf = df.reset_index()
    flag = (xdf['date'] < datetime.date(2020, 1, 1))
    xdf.loc[flag, 'volume'] = xdf.loc[flag, 'volume'] / 2
    xdf.loc[flag, 'openInterest'] = xdf.loc[flag, 'openInterest'] / 2
    if index_col == 'date':
        xdf = xdf.set_index('date')
    return xdf


def cleanup_mindata(df, asset, index_col='datetime', skip_hl=False):
    cond = None
    if index_col == None:
        xdf = df.set_index('datetime')
    else:
        xdf = df
    tradehrs = get_asset_tradehrs(asset)
    for idx, hrs in enumerate(tradehrs):
        if idx == 0:
            cond = (xdf.min_id >= tradehrs[idx][0]) & (xdf.min_id < tradehrs[idx][1])
        else:
            cond = cond | ((xdf.min_id >= tradehrs[idx][0]) & (xdf.min_id < tradehrs[idx][1]))
    if asset in ['a', 'b', 'p', 'y', 'm', 'i', 'j', 'jm']:
        cond = cond | ((xdf.date < datetime.date(2015, 5, 12)) & (xdf.min_id >= 300) & (xdf.min_id < 830))
    if asset in ['a', 'b', 'm', 'm_Opt', 'y', 'p', 'i', 'j', 'jm']:
        cond = cond | (( xdf.date <= datetime.date(2019, 3, 29)) & (xdf.date >= datetime.date(2015, 5, 12)) & (xdf.min_id >= 300) & (xdf.min_id < 530))
    if asset in ['rb', 'hc', 'bu']:
        cond = cond | ((xdf.date < datetime.date(2016, 5, 1)) & (xdf.min_id >= 300) & (xdf.min_id < 700))
    if asset in ['IF', 'IH', 'IC', ]:
        cond = cond | ((xdf.index < datetime.datetime(2016, 1, 1, 15, 0, 0)) & (xdf.min_id >= 1515) & (xdf.min_id < 1530))
        cond = cond | ((xdf.index < datetime.datetime(2016, 1, 1, 15, 0, 0)) & (xdf.min_id >= 2100) & (xdf.min_id < 2115))
    elif asset in ['T', 'TF', 'TS']:
        cond = cond | ((xdf.index < datetime.datetime(2020, 7, 20, 0, 0, 0)) & (xdf.min_id >= 1515) & (xdf.min_id < 1530))        
    if asset in ['CF', 'CF_Opt', 'CY', 'SR', 'SR_Opt', 'RM', 'TA', 'MA', 'SA', 'OI', 'ZC', 'ZC_Opt', 'FG']:
        cond = cond | ((xdf.date < datetime.date(2019, 12, 12)) & (xdf.min_id >= 300) & (xdf.min_id < 530))
    if asset in product_code['DCE'] + product_code['SHFE'] + product_code['CZCE'] + product_code['INE']:
        cond = cond & (~((xdf.date == datetime.date(2019,12,26)) & (xdf.min_id < 430)))
    xdf = xdf[cond]
    #xdf = xdf[(xdf.close > 0) & (xdf.high > 0) & (xdf.open > 0) & (xdf.low > 0)]
    if skip_hl:
        xdf = xdf[xdf.high > xdf.low]
    flag = (xdf.date<datetime.date(2020,1,1))
    xdf.loc[flag, 'volume'] = xdf.loc[flag, 'volume']/2
    xdf.loc[flag, 'openInterest'] = xdf.loc[flag, 'openInterest'] / 2
    if index_col == None:
        xdf = xdf.reset_index()
    return xdf

