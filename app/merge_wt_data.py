import numpy as np
import math
import pandas as pd
from typing import Union, List
import datetime
from pycmqlib3.utility import misc
from pycmqlib3.utility.dbaccess import *
import pycmqlib3.analytics.data_handler as dh
import pathlib
from pycmqlib3.utility.process_wt_data import *
from wtpy.wrapper import WtDataHelper

base_folder = "C:/dev/wtdev/storage/his"
update_folder = "C:/dev/data/his"

src_folder = base_folder
dst_folder = update_folder
target_folder = base_folder
base_folder = "C:/dev/wtdev/storage/his"
update_folder = "C:/dev/data/his"

d_cutoff = 20241218
time_cutoff = 202412172115
dtHelper = WtDataHelper()
period_map = {'day': 'd', 'min1': 'm1', 'min5': 'm5'}
if src_folder.lower() == target_folder.lower():
    update_flag = True
else:
    update_flag = False
for period in ['day', 'min1', 'min5', ]:
    for exch in ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE', 'GFEX']:
        print(f'{period}-{exch}')
        src_path = '%s/%s/%s' % (src_folder, period, exch)
        file_list = [f for f in listdir(src_path) if isfile(join(src_path, f))]
        for file in file_list:
            cont = file.split('.')[0]
            src_df = dtHelper.read_dsb_bars(f'{src_path}/{file}')
            src_df = src_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})
            #src_df['time'] = src_df['time'] - 199000000000
            src_df = src_df[src_df['vol']>0]
            dst_path = '%s/%s/%s' % (dst_folder, period, exch)
            dst_df = dtHelper.read_dsb_bars(f'{dst_path}/{file}')
            if dst_df:
                dst_df = dst_df.to_df().rename(columns={'bartime': 'time', 'volume': 'vol'})                
                dst_df = dst_df[dst_df['vol'] > 0]
                if period == 'day':
                    src_df = src_df[src_df['date'] < d_cutoff]
                    dst_df = dst_df[dst_df['date'] >= d_cutoff]
                    dst_df = pd.concat([src_df, dst_df])
                else:
                    src_df = src_df[src_df['time'] < time_cutoff]
                    dst_df = dst_df[dst_df['time'] >= time_cutoff]
                    dst_df = pd.concat([src_df, dst_df])
            else:
                if update_flag:
                    continue
                else:
                    dst_df = src_df
            dst_df['time'] = dst_df['time'] - 199000000000
            dst_df['time'] = dst_df['time'].astype('int64')                        
            save_bars_to_dsb(dst_df, contract=cont, folder_loc=f'{target_folder}/{period}/{exch}',
                             period=period_map[period])
            
