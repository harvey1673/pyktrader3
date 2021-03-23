import datetime
import pandas as pd
import numpy as np
import dbaccess as db
import misc
import email_tool

Product_List = [('fef', 'plt_io62'), ('m65f', 'mb_io65'), ('iolp', 'plt_lp')]
ProdFolder = "S:\\CMSC Trade Execution\\BOF\\Catherine Shipments\\25. Baffinland\\TE - FRM Baffin File\\"
TestFolder = "P:\\data\\"
TestFilename = "P:\\data\\Baffinland shipment Spreadsheet.xlsx"
ProdFilename = "S:\\CMSC Trade Execution\\BOF\\Catherine Shipments\\25. Baffinland\\TE - FRM Baffin File\\Baffinland shipment Spreadsheet.xlsx"

def tenor_to_date(tenor):
    date_list = []
    if 'Q' in tenor:
        ten_split = tenor.split('Q')
        for i in range(1, 4):
            date_list.append(datetime.date(int(ten_split[0]), (int(ten_split[1])-1)*3+i, 1))
    elif 'Cal' in tenor:
        ten_split = tenor.split('Cal')
        for i in range(1, 13):
            date_list.append(datetime.date(int(ten_split[0]), i, 1))
    elif 'H' in tenor:
        ten_split = tenor.split('H')
        for i in range(1, 7):
            date_list.append(datetime.date(int(ten_split[0]), (int(ten_split[1])-1)*6+i, 1))
    else:
        date_list = [datetime.datetime.strptime(tenor, "%Y-%m-%d").date()]
    return date_list

def load_hist_data(vdate, tenor_list, date_list, prod_list = Product_List):
    all_tenors = []
    for tenor in tenor_list:
        all_tenors += tenor_to_date(tenor)
    all_tenors = sorted(list(set(all_tenors)))
    cnx = db.connect(**db.dbconfig)
    fixings = {}
    need_fixing = (misc.day_shift(misc.day_shift(all_tenors[0], '1m'), '-1d') < vdate)
    if need_fixing:
        fix_start = all_tenors[0]
        fix_end = vdate
        for prod_name, fix_name in prod_list:
            fix_df = db.load_daily_data_to_df(cnx, 'spot_daily', fix_name, fix_start, fix_end, index_col = 'date', field = 'spotID')
            fix_df.index = pd.to_datetime(fix_df.index)
            fixings[prod_name] = fix_df.resample('M').mean()
            fixings[prod_name] = fixings[prod_name].reset_index()
            fixings[prod_name] = fixings[prod_name][fixings[prod_name]['date'].apply(lambda d: d.date()) < vdate]
            fixings[prod_name]['date'] = fixings[prod_name]['date'].apply(lambda d: datetime.date(d.year, d.month, 1))
    res = {}
    fwd_curve = {}
    for d in date_list:
        for prod_name, fix_name in prod_list:
            fdf = db.load_fut_curve(cnx, prod_name, d)
            fdf['date'] = fdf['instID'].apply(lambda x: datetime.date(2000+int(x[-4:-2]), int(x[-2:]), 1))
            fdf = fdf[['date', 'close']]
            if need_fixing:
                flag = fixings[prod_name]['date'].apply(lambda x: misc.day_shift(misc.day_shift(x, '1m'), '-1d') < d)
                fix_df = fixings[prod_name][['date', 'close']][flag]
                fdf = fix_df.append(fdf[['date', 'close']])
            fwd_curve[(fix_name, d)] = fdf
            for tenor in tenor_list:
                tenor_dates = tenor_to_date(tenor)
                res[(prod_name, d, tenor)] = fdf[fdf['date'].isin(tenor_dates)]['close'].mean()
    return res, fwd_curve

def calc_curr_margin(vdate, filename = TestFilename, product_list = Product_List):
    df = pd.read_excel(filename, skiprows=[0, 2], sheet_name = "TE Master Spread sheet")
    df = df.dropna(subset = ['Purchase no.', 'Provisional QP Date'])
    # df['init_date'] = df['Provisional QP Date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y").date())
    df['init_date'] = df['Provisional QP Date'].apply(lambda x: x.date())
    flag = (df['Final Payment date'].isna()) & (df['init_date'] <= vdate)
    df = df[flag]
    df['BL Weight (DMT)'] = (df['BL Weight (WMT)'] * (1-df['Loadport Moisture'])).round(3)
    df['Last Margin Amount'] = df['Last Margin Amount'].fillna(0.0)
    date_list = list(df['init_date'].unique())
    tenor_list = list(df['QP (Purchase contract)'].unique())
    date_list.append(vdate)

    res, fwd_curve = load_hist_data(vdate, tenor_list, date_list, product_list)

    for prod_name, fix_name in product_list:
        df['curr_' + fix_name.split('_')[1]] = df.apply(lambda x: res[(prod_name, vdate, getattr(x, 'QP (Purchase contract)'))], axis=1).round(3)
        df['init_' + fix_name.split('_')[1]] = df.apply(lambda x: res[(prod_name, x['init_date'], x['QP (Purchase contract)'])], axis=1).round(3)
    df['LPP'] = df['init_io62'] * 0.95
    df['Payment Value'] = (df['LPP'] * df['BL Weight (DMT)']).round(2)
    for prefix in ['curr_', 'init_']:
        df[prefix + 'MTMPrice'] = (0.7 * (df[prefix + 'io62'] + df[prefix + 'lp'] * 62) \
                                  + 0.3 * (df[prefix + 'io62'] + (df[prefix + 'io65'] - df[prefix + 'io62']) / 2.0)).round(3)
        df[prefix + 'MTMValue'] = (df[prefix + 'MTMPrice'] * df['BL Weight (DMT)']).round(2)
        df[prefix + 'StrategicReserve'] = (df[prefix + 'MTMValue'] - df['Payment Value']).round(2)

    df['Report Date'] = str(vdate)
    df['curr_date'] = df['Provisional QP Date']
    df['QP'] = df['QP (Purchase contract)']

    out_cols = ['Report Date', 'Purchase no.', 'QP', 'BL Weight (DMT)', 'BL Weight (WMT)', 'Loadport Moisture']
    for prefix in ['init_', 'curr_']:
        out_cols += [prefix + field for field in
                     ['date', 'io62', 'io65', 'lp', 'MTMPrice', 'MTMValue', 'StrategicReserve']]
    out_cols += ['Payment Value']
    out_df = df[out_cols]

    xdf = pd.DataFrame()
    xdf['Contract'] = df['Purchase no.']
    xdf['QP'] = df['QP (Purchase contract)']
    xdf['Report Date'] = str(vdate)
    xdf['MTM Price'] = df['curr_MTMPrice'].round(3)
    xdf['LPP/Prov PP'] = df['LPP'].round(3)
    xdf['Weight (DMT)'] = df['BL Weight (DMT)'].round(3)
    xdf['Updated Cargo Value'] = (xdf['MTM Price'] * xdf['Weight (DMT)']).round(2)
    xdf['Current Strategic Reserve'] = df['curr_StrategicReserve'].round(2)
    xdf['Day 1 Cargo Value'] = df['init_MTMValue'].round(2)
    xdf['Prov Payment'] = df['Payment Value'].round(2)
    xdf['Day 1 Strategic Reserve'] = df['init_StrategicReserve'].round(2)

    xdf['Margin Contributed by Cargill'] = df['Last Margin Amount'].apply(lambda x: max(-x, 0)).round(2)
    xdf['Margin Contributed by Baffinland'] = df['Last Margin Amount'].apply(lambda x: max(x, 0)).round(2)
    xdf['Margin Current Balance'] = xdf['Margin Contributed by Baffinland'] - xdf['Margin Contributed by Cargill']
    xdf['Net Balance'] = xdf['Current Strategic Reserve'] + xdf['Margin Current Balance']

    out_cols = ['Report Date', 'QP', 'Contract', 'MTM Price', 'LPP/Prov PP', 'Weight (DMT)', 'Updated Cargo Value', \
                'Current Strategic Reserve', 'Day 1 Cargo Value', 'Prov Payment', 'Day 1 Strategic Reserve', \
                'Margin Contributed by Cargill', 'Margin Contributed by Baffinland', 'Margin Current Balance',
                'Net Balance']
    out_xdf = xdf[out_cols]
    out_xdf = out_xdf.set_index('Contract')
    out_xdf.loc['Total'] = out_xdf.sum(numeric_only=True, axis=0).round(2)
    out_xdf.loc['Total', 'MTM Price'] = np.nan
    out_xdf.loc['Total', 'LPP/Prov PP'] = np.nan
    out_xdf['Margin Move'] = 0.0
    out_text = "No margin movement"
    min_payment = 1.0
    if out_xdf.loc['Total', 'Current Strategic Reserve'] > out_xdf.loc['Total', 'Day 1 Strategic Reserve']:
        out_xdf['Margin Move'] = out_xdf['Day 1 Strategic Reserve'] - out_xdf['Current Strategic Reserve'] \
                        - out_xdf['Margin Current Balance']
    elif (out_xdf.loc['Total', 'Current Strategic Reserve'] > 0) and (out_xdf.loc['Total', 'Current Strategic Reserve'] \
                                                                        < out_xdf.loc['Total', 'Day 1 Strategic Reserve']):
        out_xdf.loc['Total', 'Margin Current Balance'] = - out_xdf['Margin Current Balance']
    else:
        if abs(out_xdf.loc['Total', 'Net Balance']) >= 5000000:
            out_xdf.loc['Total', 'Margin Current Balance'] = - out_xdf['Net Balance'].round(2)
    if out_xdf.loc['Total', 'Margin Move'] > min_payment:
        out_text = "Margin call on Baffinland to pay %s" % (abs(out_xdf.loc['Total', 'Margin Move']))
    elif out_xdf.loc['Total', 'Margin Move'] < -min_payment:
        out_text = "Margin call on Cargill to pay %s" % (abs(out_xdf.loc['Total', 'Margin Move']))
    output = {'sendout': out_xdf, 'working': out_df, 'out_message': out_text}
    return output

def run_margin_result(tday=datetime.date.today(), \
                    xlfile='Baffinland shipment Spreadsheet.xlsx', \
                    send_email='', input_loc = ProdFolder):
    filename = input_loc + xlfile
    vdate = misc.day_shift(tday, '-1b')
    output = calc_curr_margin(vdate, filename = filename)
    curr_time = datetime.datetime.now()
    orig_file = xlfile.split('.')[0]
    outxlfile = "%sdata\\%s_result_%s.xlsx" % (input_loc, orig_file, curr_time.strftime("%Y%m%d_%H%M%S"))
    with pd.ExcelWriter(outxlfile) as writer:
        output['sendout'].to_excel(writer, "sendout")
        output['working'].to_excel(writer, 'working')
        writer.save()
    if len(send_email) > 0:
        recepient = send_email
        subject = "Baffinland margin calculation - %s" % str(tday)
        body_text = output['out_message']
        attach_files = [outxlfile]
        attach_files = []
        email_tool.send_email_by_outlook(recepient, subject, body_text, attach_files)
    return output
