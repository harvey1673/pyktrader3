import datetime
import ts_tool
import pandas as pd
import misc
import email_tool
import dbaccess as db

ffill_spot_mkts = ['Pellet']
name_map = {'fef': 'P62Swap', 'm65f': 'M65Swap', 'iolp': 'LPSwap', 'iac': 'CCSwap',\
        'plt_io62': 'P62', 'plt_io65': 'P65', 'plt_io58': 'P58', 'mb_io65': 'M65', 'plt_lp': 'LP', 'plt_pelletprem_cn': 'Pellet', \
        'plt_ckc_oz': 'AusCokingCoal', 'plt_frt_oz': 'FRT(AUS-CHN)', 'frt_baltic_c3': 'FRT(C3)', 'frt_baltic_c2': 'FRT(C2)', 'plt_si_45': 'Si 4.5-6.5%', 'plt_si_65': 'Si 6.5%', 'plt_al': 'Al 1-2.5%', \
        'plt_phos': 'Phos', 'plt_fe_viu': 'Fe VIU 1%', \
        'pbF_62_qd': 'PBF(QD)', 'nmF_63_qd': 'NMF(QD)', 'macF_61_qd': 'MAC(QD)', 'ydF_58_qd': 'Yandi(QD)', \
        'jbF_61_qd': 'JBF(QD)', 'ssF_56_qd': 'SSF(QD)', 'iocjF_65_qd': 'IOCJ(QD)', 'brbF_63_qd': 'BRBF(QD)', 'fbF_59_qd': 'FBF(QD)',\
        }
all_asset_list = {  'fut_daily': ['fef', 'm65f', 'iolp', 'iac'], \
                    'spot_daily': ['plt_io62', 'plt_io65', 'plt_io58', 'mb_io65', 'plt_lp', 'plt_pelletprem_cn', 'plt_ckc_oz', \
                                'plt_frt_oz', 'frt_baltic_c3', 'frt_baltic_c2', \
                                'plt_si_45', 'plt_si_65', 'plt_al', 'plt_phos', 'plt_fe_viu', \
                                'pbF_62_qd', 'nmF_63_qd', 'macF_61_qd', 'ydF_58_qd', 'jbF_61_qd', 'ssF_56_qd', \
                                'iocjF_65_qd', 'brbF_63_qd', 'fbF_59_qd', \
                                'pbF_62_plt', 'pbF_prem_plt', 'nmF_63_plt', 'nmF_prem_plt', 'jbF_61_plt', 'jbF_prem_plt', 'brbF_63_plt', 'brbF_prem_plt',]
                    #'cmvol_daily': ['fef'],\
                    #'fx_daily': ['USD/CNY',], \
                    #'ir_daily': ['USD3M'],\
                    }

def load_hist_data(ref_date = datetime.date.today(), ref_tenors = ['0b', '7b', '21b']):
    res = {}
    cutoff_tenor = misc.day_shift(ref_date, '18m')
    cnx = db.connect(**db.dbconfig)
    ref_dates = [misc.day_shift(ref_date, '-' + ten, misc.CHN_Holidays) for ten in ref_tenors]
    for tab_key in ['fut_daily', 'spot_daily']:
        if tab_key == 'fut_daily':
            df_list = []
            for idx, prod_code in enumerate(all_asset_list[tab_key]):
                for tenor, rdate in zip(ref_tenors, ref_dates):
                    df = db.load_fut_curve(cnx, prod_code, rdate.strftime("%Y-%m-%d"))
                    df['tenor'] = df['instID'].apply(lambda x: misc.inst2cont(x))
                    df.rename(columns={'close': (prod_code, str(rdate))}, inplace=True)
                    df = df[['tenor', (prod_code, str(rdate))]].set_index('tenor')
                    df_list.append(df)
            xdf = pd.concat(df_list, axis = 1)
            xdf.columns = pd.MultiIndex.from_product([[name_map[key] for key in all_asset_list[tab_key]], [d.strftime("%y%m%d") for d in ref_dates]])
            for col in xdf['M65Swap'].columns:
                xdf[('M65P62Spd', col)] = xdf[('M65Swap', col)] - xdf[('P62Swap', col)]
            xdf = xdf[xdf.index <= cutoff_tenor].dropna()
        elif tab_key == 'spot_daily':
            df_list = []
            for idx, prod_code in enumerate(all_asset_list[tab_key]):
                df = ts_tool.get_data(prod_code, misc.day_shift(ref_dates[-1], '-' + ref_tenors[-1]), ref_dates[0], spot_table = 'spot_daily', name = name_map[prod_code], index_col = 'date', fx_pair = None, field = 'spotID', args = None)
                df_list.append(df)
            xdf = pd.concat(df_list, axis = 1)
            xdf[ffill_spot_mkts] = xdf[ffill_spot_mkts].fillna(method = 'ffill')
            xdf = xdf[xdf.index.isin(ref_dates)].T
            xdf = xdf[reversed(xdf.columns)]
            xdf.columns = [d.strftime("%y%m%d") for d in ref_dates]
        res[tab_key] = xdf
    return res

def send_mkt_update(tday = datetime.date.today(), email_addr = ''):
    ref_date = misc.day_shift(misc.day_shift(tday, '1b', misc.CHN_Holidays), '-1b', misc.CHN_Holidays)
    res = load_hist_data(ref_date, ref_tenors = ['0b', '10b', '22b'])
    df = res['fut_daily']
    xdf = pd.DataFrame(columns=['qtr'] + [col for col in df.columns])
    for tenor in df.index:
        if (tenor > ref_date) and (tenor.month % 3 == 1):
            qtr_start = tenor
            qtr_end = misc.day_shift(tenor, '2m')
            if qtr_end > df.index[-1]:
                break
            qtr_name = str(tenor.year) + 'Q' + str(tenor.month // 3 + 1)
            xrow = [qtr_name]
            for col in res['fut_daily'].columns:
                xrow.append(df[col][(df.index >= qtr_start) & (df.index <= qtr_end)].mean())
            xdf.loc[len(xdf), :] = xrow
    xdf = xdf.set_index('qtr').dropna()
    xdf.columns = df.columns
    res['swap_grid'] = xdf.round(2)
    attach_files = []
    for mkt in ['P62Swap', 'CCSwap']:
        df = res['fut_daily'][mkt]
        df = df[df.index<=datetime.date(ref_date.year + 1, 12, 31)]
        fig = df.plot().get_figure()
        fig.suptitle('%s fwd move' % (mkt))
        fig_name = "C:\\dev\\data\\%s.png" % (mkt)
        fig.savefig(fig_name)
        attach_files.append([fig_name, mkt])
    recepient = email_addr
    subject = "ICG Market Update - %s" % str(ref_date)
    html_text = "<html><head>ICG Market Update</head><body><p>"
    html_text += "<br>Forward curve<br>{0}<br><div style=""width: 1000px; height: 600px;"">".format(
        res['fut_daily'].round(2).to_html())
    for afile in attach_files:
        html_text += "<br>{name} Fwd Change<br><img src=""cid:{id}"" width=""40%"" height=""40%""><br>".format( \
            name=afile[1], id=afile[1] + '.png')
    html_text += "</div><br>Quarterly Swap Grid<br>{tab}<br>".format(tab = res['swap_grid'].round(2).to_html())
    html_text += "</div><br>Spot market update<br>{tab}<br>".format(tab = res['spot_daily'].round(2).to_html())
    html_text += "</p></body></html>"
    email_tool.send_email_by_outlook(recepient, subject, attach_files = attach_files, html_text = html_text)
    return res

