import warnings
import sys
import datetime
import email_tool
sys.path.append("C:/dev/pycmqlib3/")
sys.path.append("C:/dev/pycmqlib3/scripts/")
import pandas as pd
import numpy as np
import win32com.client
import time
import math

input_sheet_dict = {'var': "input-variables", 'fixed': "input-fixed"}

def load_sheet_data(filename, sheet, skiprows = None, usecols = None, header = 0):
    df = pd.read_excel(filename, sheet, skiprows = skiprows, usecols = usecols, header = header)
    df.columns = [col.strip() for col in df.columns]
    return df

def update_frt_file(tday = datetime.date(2020,3,10), filename = "C:\\dev\\data\\20200311_tacora_freight_spread.xlsx",
                    sheet_dict = input_sheet_dict ):
    xlapp = win32com.client.DispatchEx("Excel.Application")
    wb = xlapp.Workbooks.Open(filename)
    ws = wb.Worksheets(sheet_dict['var'])
    ws.Cells(1,1).Value= str(tday)
    wb.RefreshAll()
    time.sleep(20)
    wb.Save()
    xlapp.Quit()

def calc_frt_grid(filename = "C:\\dev\\data\\20200311_tacora_freight_spread.xlsx", sheet_dict = input_sheet_dict):
    var_df = load_sheet_data(filename, sheet_dict['var'], skiprows = [0], usecols = 'B:U', header = 0)
    fixed_df = load_sheet_data(filename, sheet_dict['fixed'])
    fixed_df.dropna(subset=['port'], inplace = True)
    ports = list(fixed_df['port'])
    sources = list(fixed_df['source'])
    ship_types = list(fixed_df['ship_type'])
    route_list = []
    for idx, (port, source, ship_type) in enumerate(zip(ports, sources, ship_types)):
        route_name = (source, port, ship_type)
        route_list.append(route_name)
        outright = False
        frt_name = ''
        if source == 'Saldanha Bay' and port == 'Qingdao' and ship_type == 'cape':
            outright = True
            frt_name = 'c17'
        elif source == 'Tubarao' and port == 'Qingdao' and ship_type == 'cape':
            outright = True
            frt_name = 'c3'
        elif source == 'Tubarao' and port == 'Rotterdam' and ship_type == 'cape':
            outright = True
            frt_name = 'c2'
        if outright:
            frt_crv = '_'.join(['TimeCharter', ship_type, frt_name])
            var_df[route_name] = var_df[frt_crv]
            continue
        constant = 0.0
        adder = 0.0
        breaching_crv = 'BreachingINL_' + ship_type
        ballast_crv = 'BallastBonus'
        position_pickup = var_df['PositionPickup_Kembla']
        if source == 'Nueva Palmira' and ship_type == 'smax':
            if 'qingdao' in port.lower():
                adder = -3
                frt_name = 's558'
            elif 'rotterdam' in port.lower():
                adder = -2
                frt_name = 's958'
        elif source in ['Sept-Iles', 'Mo i Rana']:
            if ('qingdao' in port.lower()) or ('dung quat' in port.lower()) or ('kashima' in port.lower()) \
                    or ('nagoya' in port.lower()) or ('kembla' in port.lower()) or ('pohang' in port.lower()) \
                    or ('kaohsiung' in port.lower()) or ('fangcheng' in port.lower()) or ('beilun' in port.lower())\
                    or ('caofeidian' in port.lower()) or ('tianjin' in port.lower()) or ('huanghua' in port.lower())\
                    or ('bayuquan' in port.lower()) or ('jiangyin' in port.lower()) or ('bahrain' in port.lower()) \
                    or ('jaigad' in port.lower()) or ('mangalore' in port.lower()) or ('paradip' in port.lower()):
                if ship_type == 'cape':
                    frt_name = 'bc5tc'
                elif ship_type == 'pmax':
                    frt_name = 'p2a82'
                    constant = 2000
            elif ('rotterdam' in port.lower()):
                if ship_type == 'cape':
                    frt_name = 'c8'
                    constant = 2500
                elif ship_type == 'pmax':
                    frt_name = 'p1a82'
                    constant = 1000
        elif (source in ['Saldanha Bay']) and ('rotterdam' in port.lower()) and ship_type == 'cape':
            frt_name = 'bc5tc'
            position_pickup = (var_df[ballast_crv] + var_df[ballast_crv].shift(-1).fillna(method='ffill')) * 0.5
        frt_crv = '_'.join(['TimeCharter', ship_type, frt_name])
        var_df[route_name] = ((var_df[frt_crv] + constant + var_df[breaching_crv] * fixed_df['UseBreaching'].iloc[idx]) * fixed_df['duration'].iloc[idx] \
                        + var_df[ballast_crv] * fixed_df[ballast_crv].iloc[idx]) * (1-0.0375) \
                        + fixed_df['fixed_costs'].iloc[idx] - position_pickup * fixed_df['PositionPickup_Kembla'].iloc[idx]
        bunker_loc = [b.strip() for b in fixed_df['bunkers'].iloc[idx].split(',')]
        num_loc = len(bunker_loc)
        for loc in bunker_loc:
            for fuel in ['IFO', 'LSGO']:
                fuel_crv = fuel + '_' + loc
                var_df[route_name] += var_df[fuel_crv] * fixed_df[fuel].iloc[idx]/float(num_loc)
        var_df[route_name] = var_df[route_name] /fixed_df['intake'].iloc[idx] + adder
    cost_grid = var_df[route_list].round(2)
    cost_grid['tenor'] = var_df['tenor']
    cost_grid.set_index('tenor', inplace=True)
    return cost_grid

def run_frt_cost(tday = datetime.date.today(), filename = "P:\\data\\20200311_tacora_freight_spread.xlsx",
                 sheet_dict = input_sheet_dict, xlfile = "C:\\dev\\data\\freight_grid_calc_%s.xlsx", send_email = ''):
    update_frt_file(tday, filename, input_sheet_dict)
    cost_grid = calc_frt_grid(filename, sheet_dict)
    xlfile = xlfile % str(tday)
    if xlfile:
        writer = pd.ExcelWriter(xlfile)
        cost_grid.to_excel(writer, "freight_grid")
        writer.save()
        if len(send_email) > 0:
            recepient = send_email
            subject = "Freight Cost Grid - %s" % (str(tday))
            html_text = cost_grid.to_html()
            attach_files = [xlfile]
            email_tool.send_email_by_outlook(recepient, subject, attach_files = attach_files, html_text = html_text)

if __name__ == "__main__":
    run_frt_cost()