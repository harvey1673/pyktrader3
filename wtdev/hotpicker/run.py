from wtpy.apps import WtHotPicker, WtCacheMonExchg, WtCacheMonSS, WtMailNotifier
import os
from shutil import copyfile
import json
import datetime
import logging
from pycmqlib3.utility.sec_bits import EMAIL_QQ, EMAIL_NOTIFY, NOTIFIERS, \
    HOT_UPDATE_NUTSHARE, LOCAL_NUTSTORE_FOLDER
from pycmqlib3.utility import update_contract_roll
from pycmqlib3.utility.convert_hot_json import process_file_only_czce

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
                    notify=EMAIL_NOTIFY):
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
        notifier = WtMailNotifier(user=EMAIL_QQ['user'],
                                  pwd=EMAIL_QQ['passwd'],
                                  host=EMAIL_QQ['host'],
                                  port=EMAIL_QQ['port'],
                                  isSSL=True)
        for rec in NOTIFIERS:
            notifier.add_receiver(addr=rec)
        picker.set_mail_notifier(notifier)
    picker.execute_increment(end_date)


if __name__ == "__main__":
    files = {'loc': 'C:/dev/wtdev/deploy/hotpicker/', 'hot': 'hots.json', 'sec': 'seconds.json', 'marker': 'marker.json'}
    daily_hot_rules(files=files, notify=EMAIL_NOTIFY)
    
    prod_loc = 'C:/dev/wtdev/common/'
    hot_config_loc = 'C:/dev/wtdev/config/'
    nutstore_common_loc = f'{LOCAL_NUTSTORE_FOLDER}/common'
    nutstore_config_loc = f'{LOCAL_NUTSTORE_FOLDER}/config'

    file_map = {
        'hots': 'hot1',
        'seconds': 'hot2',
    }

    for file in ['hots', 'seconds', ]:
        try:
            os.rename(f'{prod_loc}{file}.json', f'{prod_loc}{file}_old.json')
        except WindowsError:
            os.remove(f'{prod_loc}{file}_old.json')
            os.rename(f'{prod_loc}{file}.json', f'{prod_loc}{file}_old.json')
        
        copyfile('%s%s.json' % (files['loc'], file), 
                 '%s%s.json' % (prod_loc, file))
        
        copyfile('%s%s.json' % (files['loc'], file), 
                 '%s%s.json' % (hot_config_loc, file_map[file]))
        
        if HOT_UPDATE_NUTSHARE:
            copyfile('%s%s.json' % (files['loc'], file), 
                     '%s%s.json' % (nutstore_common_loc, file))
            
            copyfile('%s%s.json' % (hot_config_loc, file_map[file]), 
                     '%s%s.json' % (nutstore_config_loc, file_map[file]))
        else:
            copyfile('%s%s.json' % (nutstore_common_loc, file),
                     '%s%s.json' % (files['loc'], file))
            
            copyfile('%s%s.json' % (nutstore_config_loc, file_map[file]),
                     '%s%s.json' % (hot_config_loc, file_map[file]))

    input("press enter key to exit\n")
