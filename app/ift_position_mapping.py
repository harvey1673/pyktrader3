import sys
sys.path.append("C:/dev/pycmqlib3/")
sys.path.append("C:/dev/pycmqlib3/scripts/")
import datetime
import email_tool
import pandas as pd
import numpy as np

dest = "C:\\dev\\data\\"
origin_folder = "S:\\FERROUS\\International_Steel\\IFT Open Position\\Archive\\"

map_file = "C:\\dev\\pycmqlib3\\data\\IFT_curve_risk_map.xlsx"
map_sheet = "L3_L12"

def load_daily_position(tday = datetime.date.today(), origin_folder = origin_folder, pos_sheet = "TodaysPosition"):
    src_folder = origin_folder + datetime.datetime.strftime(tday, "%B %Y") + "\\"
    src_file = "IFT Position " + datetime.datetime.strftime(tday,"%m%d%y") + ".xlsm"
    src_path = src_folder + src_file
    df = pd.read_excel(src_path, pos_sheet)
    return df

def load_position_map(mfile = map_file, msheet = "L3_L12"):
    map_df = pd.read_excel(mfile, msheet).set_index(['l3_name'])
    map_df = map_df.fillna(0.0)
    return map_df

def position_mapping(df, map_df, tday = datetime.date.today()):
    xdf = df[['Contract Ref', 'P/S', 'Book', 'Product', 'Origin', 'Destination', \
              'Time Bucket', 'Position', 'Contract Term']]
    xdf = xdf[xdf['Position'].apply(np.isreal)]
    xdf['Time Bucket'] = xdf['Time Bucket'].apply(lambda x: x.date())
    xdf = xdf[xdf['Time Bucket'] >= tday]
    xdf['l3_curve'] = np.nan
    flag = (xdf['Product'] == 'Rebar') & (xdf['Destination'].isin(['Singapore', 'Hong Kong']))
    xdf.loc[flag, 'l3_curve'] = 'rebar_sea'
    flag = (xdf['Product'] == 'Billet') & (xdf['Destination'].isin(['Philippines', 'Indonesia']))
    xdf.loc[flag, 'l3_curve'] = 'billet_sea'
    flag = (xdf['Product'] == 'Billet') & (xdf['Origin'].isin(['Saudi Arabia']))
    xdf.loc[flag, 'l3_curve'] = 'billet_sea'
    flag = (xdf['Product'] == 'TS Billet')
    xdf.loc[flag, 'l3_curve'] = 'billet_ts'
    flag = (xdf['Product'] == 'SHFE Rebar')
    xdf.loc[flag, 'l3_curve'] = 'rebar_shfe'
    flag = (xdf['Product'] == 'LME Scrap') & (~(xdf['Contract Ref'].isnull()))
    xdf.loc[flag, 'l3_curve'] = 'scrap_lme'
    flag = (xdf['Origin'] == 'Bastug Option') & (~(xdf['Contract Term'].isnull()))
    xdf.loc[flag, 'l3_curve'] = 'scrap_lme'
    xdf.loc[flag, 'Product'] = 'LME Scrap'
    #adf = xdf[flag].copy()
    #adf.loc[:, 'l3_curve'] = 'billet_bs'
    #adf.loc[:, 'Position'] = xdf.loc[flag, 'Position'] / 0.43
    #adf.loc[:, "Product"] = 'Billet'
    #adf.loc[:, 'Origin'] = 'Turkey'
    #bdf = xdf[flag].copy()
    #bdf.loc[:, 'l3_curve'] = 'scrap_lme'
    #bdf.loc[:, 'Position'] = xdf.loc[flag, 'Position'] / 0.43
    #bdf.loc[:, "Product"] = 'LME Scrap'
    #bdf.loc[:, 'Origin'] = 'Turkey'
    #xdf = xdf[~flag]
    #xdf = xdf.append(adf)
    #xdf = xdf.append(bdf)

    flag = (xdf['Product'].isin(['Hot Rolled Coil', 'Hot Rolled Coil PO'])) & (
        xdf['Destination'].isin(['Thailand', 'Vietnam', 'Pakistan', 'Uae']))
    xdf.loc[flag, 'l3_curve'] = 'hrc_sea'
    flag = (xdf['Product'].isin(['Hot Rolled Coil', 'Hot Rolled Coil PO'])) & (
        xdf['Destination'].isin(['Italy', 'Spain']))
    xdf.loc[flag, 'l3_curve'] = 'hrc_eu'
    flag = (xdf['Product'].isin(['Hot Rolled Coil', 'Hot Rolled Coil PO'])) & (
        xdf['Origin'].isin(['MEPS Index Exposure']))
    xdf.loc[flag, 'l3_curve'] = 'hrc_eu'

    flag = (xdf['Product'] == 'Cold Rolled Coil')
    xdf.loc[flag, 'l3_curve'] = 'crc_eu'
    flag = (xdf['Product'] == 'SHFE HRC')
    xdf.loc[flag, 'l3_curve'] = 'hrc_shfe'

    flag = (xdf['Product'] == 'HRC - Swap')
    xdf.loc[flag, 'l3_curve'] = 'hrc_sea'

    flag = (xdf['Product'] == 'LME HRC China')
    xdf.loc[flag, 'l3_curve'] = 'hrc_sea'
    # xdf = xdf[xdf['Book']=='Int Longs']
    xdf_longs = xdf[xdf['Book'] == 'Int Longs']
    xdf_flats = xdf[xdf['Book'] == 'Int Flats']
    longs_table = pd.pivot_table(xdf_longs, values='Position', index=['l3_curve'], \
                                 columns='Time Bucket', aggfunc=np.sum, fill_value=0)
    #flats_table = pd.pivot_table(xdf_flats, values='Position', index=['l3_curve'], \
    #                             columns='Time Bucket', aggfunc=np.sum, fill_value=0)

    longs_map = map_df.loc[longs_table.index].fillna(0.0)
    longs_l2 = pd.DataFrame(longs_map.values.transpose().dot(longs_table.values), index=longs_map.columns,
                            columns=longs_table.columns)
    longs_table['sum'] = longs_table.sum(axis=1)
    longs_l2['sum'] = longs_l2.sum(axis=1)
    return {'pos_l3': longs_table.T, 'pos_l2': longs_l2.T}

def run_pos_mapping(tday = datetime.date.today(), \
                    xlfile = 'S:\\FERROUS\\International_Steel\\Trading Support\\structuring_data\\IFT_level_positions_', \
                    send_email =  ''):
    df = load_daily_position(tday)
    map_df = load_position_map()
    res = position_mapping(df, map_df, tday)
    res['pos_map'] = map_df
    if xlfile:
        writer = pd.ExcelWriter(xlfile + str(tday) + '.xlsx')
        for key in res:
            res[key].to_excel(writer, key)
        writer.save()

    if len(send_email) > 0:
        recepient = send_email
        subject = "IFT longs daily aggregated positions - %s" % str(tday)
        body_text = "File in the attachment"
        attach_files = [xlfile + str(tday) + '.xlsx']
        email_tool.send_email_by_outlook(recepient, subject, body_text, attach_files)
    return res

