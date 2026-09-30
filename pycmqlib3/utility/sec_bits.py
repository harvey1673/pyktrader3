import platform
import json
from pathlib import Path

with (Path(__file__).resolve().parents[2] / "process" / "sec_bits.json").open(encoding="utf-8") as _config_file:
    _settings = json.load(_config_file)

LOCAL_PC_NAME = _settings['LOCAL_PC_NAME']
EMAIL_NOTIFY = _settings['EMAIL_NOTIFY']
HOT_UPDATE_NUTSHARE = _settings['HOT_UPDATE_NUTSHARE']
NOTIFIERS = _settings['NOTIFIERS']
PROXY_CREDENTIALS = _settings['PROXY_CREDENTIALS']
skype_user = _settings['skype_user']
LOCAL_NUTSTORE_FOLDER = _settings['LOCAL_NUTSTORE_FOLDER']
IFIND_XL_HOTKEYS = _settings['IFIND_XL_HOTKEYS']
MYSTEEL_XL_HOTKEYS = _settings['MYSTEEL_XL_HOTKEYS']
ifind_user = _settings['ifind_user']
dbconfig = _settings['dbconfig']
misc_dbconfig = _settings['misc_dbconfig']
hist_dbconfig = _settings['hist_dbconfig']
bktest_dbconfig = _settings['bktest_dbconfig']
EMAIL_HOTMAIL = _settings['EMAIL_HOTMAIL']
EMAIL_ALIYUN = _settings['EMAIL_ALIYUN']
EMAIL_QQ = _settings['EMAIL_QQ']

del _settings, _config_file


def get_prod_folder():
    folder = ''
    system = platform.system()
    if system == 'Linux':
        folder = '/home/dev/pycmqlib/'
    elif system == 'Windows':
        folder = 'C:\\dev\\pycmqlib\\'
    return folder
