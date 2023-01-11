import platform

PROXY_CREDENTIALS = {'user': 'xxxxxx', 'passwd': 'xxxxxxx'}

skype_user = {'user': 'wei.x.wu', 'pwd': 'HW@9619252y'}

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
    'host': 'smtp.office365.com',
    'port': 587,
    'user': 'harvey_wwu@hotmail.com',
    'passwd': 'HW@9619252y',
}


def get_prod_folder():
    folder = ''
    system = platform.system()
    if system == 'Linux':
        folder = '/home/dev/pycmqlib/'
    elif system == 'Windows':
        folder = 'C:\\dev\\pycmqlib\\'
    return folder
