from wtpy.apps import WtHotPicker, WtCacheMonExchg, WtCacheMonSS, WtMailNotifier
import os
from shutil import copyfile
import json
import datetime
import logging
from pycmqlib3.utility.sec_bits import EMAIL_HOTMAIL

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
                    files={'loc': './', 'hot': 'hots.json', 'sec': 'seconds.json', 'marker': 'marker.json'},
                    snapshot_loc="C:/dev/wtdev/storage/his/snapshot/",
                    notify=False):
    # 增量更新主力合约切换规则
    if snapshot_loc:
        # 从datakit落地的行情快照直接读取
        cacher = WtCacheMonSS(snapshot_loc)
    else:
        # 从交易所官网拉取行情快照
        cacher = WtCacheMonExchg()

    picker = WtHotPicker(files)
    picker.set_cacher(cacher)
    if notify:
        notifier = WtMailNotifier(user=EMAIL_HOTMAIL['user'],
                                  pwd=EMAIL_HOTMAIL['passwd'],
                                  host=EMAIL_HOTMAIL['host'],
                                  port=EMAIL_HOTMAIL['port'],
                                  isSSL=False)
        notifier.add_receiver(addr="harvey_wwu@yahoo.com")
        picker.set_mail_notifier(notifier)
    picker.execute_increment(end_date)


if __name__ == "__main__":
    files = {'loc': 'C:/dev/wtdev/hotpicker/', 'hot': 'hots.json', 'sec': 'seconds.json', 'marker': 'marker.json'}
    daily_hot_rules(files=files, notify=True)
    prod_loc = 'C:/dev/wtdev/common/'
    for file in ['hots', 'seconds']:
        try:
            os.rename(f'{prod_loc}{file}.json', f'{prod_loc}{file}_old.json')
        except WindowsError:
            os.remove(f'{prod_loc}{file}_old.json')
            os.rename(f'{prod_loc}{file}.json', f'{prod_loc}{file}_old.json')
        copyfile('%s%s.json' % (files['loc'], file), '%s%s.json' % (prod_loc, file))
    input("press enter key to exit\n")
