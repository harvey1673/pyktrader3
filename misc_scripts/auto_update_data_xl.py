import time
import datetime
import pyautogui
import win32com.client
import win32gui
import win32con
from pycmqlib3.utility.sec_bits import LOCAL_NUTSTORE_FOLDER, IFIND_XL_HOTKEYS
from pycmqlib3.utility.dbaccess import write_edb_by_xl_sheet, write_stock_data_by_xl


def update_ifind_xlsheet(filename='C:/Users/harvey/Nutstore/1/Nutstore/ifind_data.xlsx', 
                         wait_time=40, excluded=['hist']):
    xl = win32com.client.DispatchEx("Excel.Application")
    wb = xl.Workbooks.open(filename)
    xl.Visible = True
    wb.Activate()
    time.sleep(10)
    for s in range(len(wb.Sheets)):
        try:
            if wb.Sheets[s].name in excluded:
                continue
            wtime = wait_time
            is_daily = False
            if '_d' in wb.Sheets[s].name:
                is_daily=True
                wtime = wait_time + 10
            win32gui.ShowWindow(xl.Hwnd, win32con.SW_RESTORE)
            win32gui.EnableWindow(xl.Hwnd, True)
            win32gui.SetForegroundWindow(xl.Hwnd)
            wb.Sheets[s].Activate()
            pyautogui.typewrite(IFIND_XL_HOTKEYS, interval=0.5)
            time.sleep(3)
            pyautogui.hotkey("shift", "f9")
            time.sleep(wtime)
            if is_daily:
                pyautogui.typewrite(IFIND_XL_HOTKEYS, interval=0.5)
                time.sleep(3)
                pyautogui.hotkey("shift", "f9")
                time.sleep(3)
            xl.CalculateUntilAsyncQueriesDone()
        except Exception as e:
            print("error activating Excel window for update %s" % (wb.Sheets[s].name))
            continue
    wb.Close(SaveChanges=1)
    time.sleep(3)
    xl.Quit()


def update_data_from_xl(data_folder=LOCAL_NUTSTORE_FOLDER, lookback=30):
    file_setup = {
        # ('ifind_data.xlsx', 'hist'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
        #                                 'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'ferrous_w'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'base_w'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                           'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'const_d'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                       'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        # special format from here
        ('ifind_data.xlsx', 'base_d2'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 6, 7, 8],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'macro_m'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 6, 7, 8],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'warrant_d'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 6, 7, 8],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_daily.xlsx', 'petchem_d'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},                                        
        ('ifind_daily.xlsx', 'macro_d'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 6, 7, 8],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_daily.xlsx', 'ferrous_d'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 6, 7, 8],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_daily.xlsx', 'base_d'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                         'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
    }
    write_edb_by_xl_sheet(file_setup, data_folder=data_folder, lookback=lookback)
    write_stock_data_by_xl('ifind_stock.xlsx', data_folder=LOCAL_NUTSTORE_FOLDER, lookback=1000)



if __name__ == "__main__":
    data_folder = LOCAL_NUTSTORE_FOLDER 
    now = datetime.datetime.now()
    if now.time() > datetime.time(18, 0, 0):
        update_ifind_xlsheet(filename=f'{data_folder}/ifind_data.xlsx', wait_time=40, excluded=['hist'])
    update_ifind_xlsheet(filename=f'{data_folder}/ifind_daily.xlsx', wait_time=40, excluded=[])
    update_ifind_xlsheet(filename=f'{data_folder}/ifind_stock.xlsx', wait_time=10, excluded=['setup'])
