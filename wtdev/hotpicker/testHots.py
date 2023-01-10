'''
Descripttion: Automatically generated file comment
version: 
Author: Wesley
Date: 2021-08-23 09:38:05
LastEditors: Wesley
LastEditTime: 2021-08-23 15:06:29
'''
from wtpy.apps import WtHotPicker, WtCacheMonExchg, WtCacheMonSS, WtMailNotifier
import json
import datetime
import logging

logging.basicConfig(filename='hotsel.log', level=logging.INFO, filemode="w", 
    format='[%(asctime)s - %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 设置日志打印格式
formatter = logging.Formatter(fmt="[%(asctime)s - %(levelname)s] %(message)s", datefmt='%m-%d %H:%M:%S')
console.setFormatter(formatter)
# 将定义好的console日志handler添加到root logger
logging.getLogger('').addHandler(console)


def rebuild_hot_rules(start_date, end_date,
                      files=["hots.json", "seconds.json"],
                      snapshot_loc="C:/dev/wtdev/storage/his/snapshot/"):
    '''
    重构全部的主力合约切换规则
    '''
    if snapshot_loc:
        # 从datakit落地的行情快照直接读取
        cacher = WtCacheMonSS(snapshot_loc)
    else:
        # 从交易所官网拉取行情快照
        cacher = WtCacheMonExchg()

    picker = WtHotPicker(hotFile=files[0], secFile=files[1])
    picker.set_cacher(cacher)

    hotRules, secRules = picker.execute_rebuild(start_date, end_date,
                                                exchanges=["CFFEX", "SHFE", "CZCE", "DCE", "INE", "GFEX"],
                                                wait=False)
    output = open(files[0], 'w')
    output.write(json.dumps(hotRules, sort_keys=True, indent=4))
    output.close()
    output = open(files[1], 'w')
    output.write(json.dumps(secRules, sort_keys=True, indent=4))
    output.close()
    return hotRules, secRules


def daily_hot_rules(end_date=None,
                    files=["hots.json", "seconds.json"],
                    snapshot_loc="C:/dev/wtdev/storage/his/snapshot/"):
    # 增量更新主力合约切换规则
    if snapshot_loc:
        # 从datakit落地的行情快照直接读取
        cacher = WtCacheMonSS(snapshot_loc)
    else:
        # 从交易所官网拉取行情快照
        cacher = WtCacheMonExchg()

    picker = WtHotPicker(hotFile=files[0], secFile=files[1])
    picker.set_cacher(cacher)

    # notifier = WtMailNotifier(user="yourmailaddr", pwd="yourmailpwd", host="smtp.exmail.qq.com", port=465, isSSL=True)
    # notifier.add_receiver(name="receiver1", addr="receiver1@qq.com")
    # picker.set_mail_notifier(notifier)

    picker.execute_increment(end_date)


if __name__ == "__main__":
    rebuild_hot_rules()
    input("press enter key to exit\n")