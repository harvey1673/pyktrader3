import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time

def snap_rate():
    url = "https://www.boc.cn/sourcedb/whpj/index_4.html"
    # Request
    r1 = requests.get(url)
    # We'll save in coverpage the cover page content
    coverpage = r1.content
    # Soup creation
    soup = BeautifulSoup(coverpage, 'html5lib')
    table = soup.find('table', attrs = {"cellpadding": "0", "align": "left", "cellspacing": "0",
                                        "width": "100%"})
    table_rows = table.find_all('tr')
    l = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [tr.text for tr in td]
        if len(row) == 8:
            l.append(row)

    df = pd.DataFrame(l, columns=["货币名称","现汇买入价",
                            "现钞买入价", "现汇卖出价",
                            "现钞卖出价","中行折算价",
                            "发布日期", "发布时间"])
    return df
