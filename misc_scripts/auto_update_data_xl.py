import time
import pyautogui
import win32com.client
import win32gui
import win32con
from pycmqlib3.utility.sec_bits import LOCAL_NUTSTORE_FOLDER, IFIND_XL_HOTKEYS
from pycmqlib3.utility.dbaccess import write_edb_by_xl_sheet


def update_ifind_xlsheet(filename='C:/Users/harvey/Nutstore/1/Nutstore/ifind_data.xlsx', wait_time=65):
    xl = win32com.client.DispatchEx("Excel.Application")
    wb = xl.Workbooks.open(filename)
    xl.Visible = True
    wb.Activate()
    time.sleep(10)
    for s in range(len(wb.Sheets)):
        try:
            if wb.Sheets[s].name in ['hist']:
                continue
            win32gui.ShowWindow(xl.Hwnd, win32con.SW_RESTORE)
            win32gui.EnableWindow(xl.Hwnd, True)
            win32gui.SetForegroundWindow(xl.Hwnd)
            wb.Sheets[s].Activate()
            pyautogui.typewrite(IFIND_XL_HOTKEYS, interval=0.5)
            time.sleep(3)
            pyautogui.hotkey("shift", "f9")
            time.sleep(wait_time)
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
        ('ifind_data.xlsx', 'const'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'daily'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                       'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'weekly'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'sector'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'base_daily'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                        'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
        ('ifind_data.xlsx', 'base_wkly'): {'header': [0, 1, 2, 3], 'skiprows': [0, 1, 2, 7, 8, 9],
                                            'source': 'ifind', 'reorder': [0, 1, 2, 3], 'drop_zero': False},
    }
    write_edb_by_xl_sheet(file_setup, data_folder=data_folder, lookback=lookback)


if __name__ == "__main__":
    data_folder = LOCAL_NUTSTORE_FOLDER
    filename = f'{data_folder}/ifind_data.xlsx'
    update_ifind_xlsheet(filename=filename)
