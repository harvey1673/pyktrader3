import platform

LOCAL_PC_NAME = 'LAPTOP-ROG'
EMAIL_NOTIFY = False
NOTIFIERS = ['harveywu@hotmail.com']
PROXY_CREDENTIALS = {'user': 'xxxxxx', 'passwd': 'xxxxxxx'}
skype_user = {'user': 'wei.x.wu', 'pwd': 'HW@9619252y'}
LOCAL_NUTSTORE_FOLDER = 'C:/Users/harvey/Nutstore/1/Nutstore'
IFIND_XL_HOTKEYS = ['alt', 'y', '3', 'y', 'h', 'enter']
dbconfig = {
    'user': 'harvey',
    'password':'9619252y',
    'host':'localhost',
    'database': 'blueshale',
}

misc_dbconfig = {
    'user': 'harvey',
    'password':'9619252y',
    'host':'localhost',
    'database': 'blueshale',
}

hist_dbconfig = {
    'user': 'harvey',
    'password': '9619252y',
    'host': 'localhost',
    'database': 'hist_data',
}

bktest_dbconfig = {
    'user': 'harvey',
    'password': '9619252y',
    'host': 'localhost',
    'database': 'bktest_db',
}

EMAIL_HOTMAIL = {
    'host': 'smtp-mail.outlook.com',
    'port': 587,
    'user': 'harveywu@outlook.com',
    'passwd': 'HW@9619252y',
}

EMAIL_ALIYUN = {
    'host': 'smtp.aliyun.com',
    'port': 25,
    'user': 'harvey_wwu@aliyun.com',
    'passwd': 'HW@9619252y',
}


EMAIL_QQ = {
    'host': 'smtp.qq.com',
    'port': 465,
    'user': 'harvey_wwu@qq.com',
    'passwd': 'jmvkusyrenxjbebf',
}


def get_prod_folder():
    folder = ''
    system = platform.system()
    if system == 'Linux':
        folder = '/home/dev/pycmqlib/'
    elif system == 'Windows':
        folder = 'C:\\dev\\pycmqlib\\'
    return folder
