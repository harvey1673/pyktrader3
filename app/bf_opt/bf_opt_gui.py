#-*- coding:utf-8 -*-
import tkinter as tk
import copy
import sys
import tkinter.ttk
import json
import pandas as pd
from gui_misc import *
import load_bf_data
import bf_optimizer

class BFOptGui(tk.Tk):
    def __init__(self, config, load_config = True):
        tk.Tk.__init__(self)
        self.title("Cargill Metal Blast Furnace Optimizer")
        self.geometry("1120x640")
        self.plant_file = config.get('plant_file', 'plant_config.json')
        self.material_file = config.get('material_file', 'material_config.json')
        self.price_map = config.get('price_map', 'PORT_PRICE_MAP')
        if load_config:
            self.load_material_setting(need_update = False)
            self.load_plant_setting(need_update=False)
        else:
            self.material_dict = load_bf_data.prepare_material_data()
            self.plant_config = load_bf_data.prepare_plant_data()

        self.pricing_start = datetime.date(2018,1,1)
        self.pricing_end = datetime.date.today()
        p_map = getattr(bf_optimizer, self.price_map)
        self.mkt_data, self.pricing_indices = bf_optimizer.get_material_prices(p_map, self.pricing_start, self.pricing_end)
        self.price_weights = {}
        self.weight_entries = {}

        for material in self.material_dict:
            if material in p_map:
                self.price_weights[material] = dict(p_map[material])
            else:
                self.price_weights[material] = {}
            for index in self.pricing_indices:
                if index not in self.price_weights[material]:
                    self.price_weights[material][index] = 0.0
                self.weight_entries[material + ':'+ index] = get_type_var('float')
        self.pricing_date = self.mkt_data.index[-1]
        self.pricing_entries = {}
        for key, d in zip(['start', 'end', 'curr'], [self.pricing_start, self.pricing_end, self.pricing_date]):
            self.pricing_entries[key] = get_type_var('str')
            self.pricing_entries[key].set(datetime.datetime.strftime(d, '%Y%m%d'))

        self.sorted_materials = sorted(list(self.material_dict.keys()), key=lambda s: s.lower())
        if ('input_list' in self.plant_config) and (len(self.plant_config['input_list']) > 0):
            self.input_list = self.plant_config['input_list']
        else:
            self.input_list = list(self.material_dict.keys())
            self.plant_config['input_list'] = self.input_list

        self.plant_config['coke_penalty'] = self.plant_config.get('coke_penalty', 0.0) > 0
        self.plant_config['chem_penalty'] = self.plant_config.get('chem_penalty', 0.0) > 0

        self.plant_entries = {}
        self.plant_variable_map = {}
        self.mat_entries = {}
        self.mat_selector = {}
        self.solution = {}
        self.alloc_w = dict([(input, 0.0) for input in list(self.material_dict.keys())])
        self.plant_output = {'prod_info': ['conc_ratio'] + \
                                          [ key2 + '_' + key1 for key1 in ['cost', 'volume', 'mean'] \
                                            for key2 in ['total', 'sinter', 'hotmetal', 'slag', 'material']], \
                             'mat_info': [ '_'.join([a, b]) for a in ['Sinfeed', 'Cencentrate', 'Lump', 'Pellet', 'Sinfuel', 'BFCoke', 'Additive'] \
                                          for b in ['volume', 'num', 'cost', 'mean']],
                             'sinter_ratio': ['Fe', 'Mn', 'CaO', 'SiO2', 'Al2O3', 'MgO', 'P', 'S', 'Zn', 'K2O', 'Na2O', 'Pb', 'Bas', 'RDI'], \
                             'bf_input': ['Zn', 'Pb', 'K2O', 'Na2O', 'Pellet', "Lump", 'Fe', 'Mn', 'P', 'Alkali'], \
                             'hm_ratio': ['P', 'Mn'], \
                             'slag_ratio': ['usage', 'Al2O3', 'MgO', 'CaO', 'SiO2', 'Bas', 'Tbas', 'Mn','Cement'],
                             }
        self.output_entries = {}
        for key in self.plant_output:
            if type(self.plant_output[key]).__name__ == 'list':
                self.output_entries[key] = {}
                for elem in self.plant_output[key]:
                    self.output_entries[key][elem] = get_type_var('float')
            else:
                self.output_entries[key] = get_type_var('float')

        for key in self.plant_config:
            if key in ['ref_input', 'input_list']:
                continue
            vtype = type(self.plant_config[key]).__name__
            if vtype == 'dict':
                for key2 in self.plant_config[key]:
                    vvtype = type(self.plant_config[key][key2]).__name__
                    self.plant_entries[key + ':' + key2] = get_type_var(vvtype)
                    self.plant_variable_map[key + ':' + key2] = vvtype
                    self.plant_entries[key + ':' + key2].set(self.plant_config[key][key2])
            else:
                self.plant_entries[key] = get_type_var(vtype)
                self.plant_variable_map[key] = vtype
                self.plant_entries[key].set(self.plant_config[key])

        self.plant_status = get_type_var('int')
        self.mat_entry_fields = bf_optimizer.CHEM_CONTENTS + bf_optimizer.MATERIAL_ALLOC \
                                + ['price', 'coke_coeff', 'lower', 'upper', 'ref_input']
        self.mat_output_fields = ['alloc_w', 'alloc_cost', 'alloc_fbed', 'alloc_cbed', 'alloc_bof', 'alloc_sieved',]
        self.mat_label_fields = ['name', 'ptype', ]

        for input in list(self.material_dict.keys()):
            self.mat_selector[input] = tk.BooleanVar()
            self.mat_entries[input] = {}
            for field in self.mat_entry_fields:
                self.mat_entries[input][field] = get_type_var('float')
            for field in self.mat_output_fields:
                self.mat_entries[input][field] = get_type_var('float')

        menu = tk.Menu(self)
        toolmenu = tk.Menu(menu, tearoff=0)
        toolmenu.add_command(label='Load Plant Setting', command=self.load_plant_setting)
        toolmenu.add_command(label='Load Material Setting', command=self.load_material_setting)
        toolmenu.add_command(label='Save Plant Setting', command=self.save_plant_setting)
        toolmenu.add_command(label='Save Material Setting', command=self.save_material_setting)
        menu.add_cascade(label="Tools", menu=toolmenu)
        menu.add_command(label="Exit", command=self.onExit)
        self.config(menu=menu)
        self.notebook = tkinter.ttk.Notebook(self)

        plant_input_win = tkinter.ttk.Frame(self.notebook)
        self.plant_input(plant_input_win)
        self.notebook.add(plant_input_win, text='Plant Input')

        chem_input_win = tkinter.ttk.Frame(self.notebook)
        self.chem_input(chem_input_win)
        self.notebook.add(chem_input_win, text='Chem Contents')

        material_input_win = tkinter.ttk.Frame(self.notebook)
        self.material_input(material_input_win)
        self.notebook.add(material_input_win, text='Material Input')

        pricing_input_win = tkinter.ttk.Frame(self.notebook)
        self.pricing_input(pricing_input_win)
        self.notebook.add(pricing_input_win, text='Pricing Input')

        solution_output_win = tkinter.ttk.Frame(self.notebook)
        self.solution_settings(solution_output_win)
        self.notebook.add(solution_output_win, text='Solution')

        calculated_output_win = tkinter.ttk.Frame(self.notebook)
        self.calculated_settings(calculated_output_win)
        self.notebook.add(calculated_output_win, text='Calculated')
        self.notebook.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        self.get_pricing_weight()
        self.refresh_material_prices()
        self.set_plant_params()
        self.set_material_params(self.mat_entry_fields)

    def refresh_material_prices(self):
        if self.pricing_date not in self.mkt_data.index:
            self.pricing_date = self.mkt_data.index[-1]
        mkt_dict = self.mkt_data.to_dict(orient='index')
        for input in self.material_dict:
            self.material_dict[input].price = mkt_dict[self.pricing_date].get(input, 0.0) #* (1 - self.material_dict[input].contents['H2O'])

    def set_pricing_weight(self):
        for key, attr in zip(['start', 'end', 'curr'], ['pricing_start', 'pricing_end', 'pricing_date']):
            value = self.pricing_entries[key].get()
            value = datetime.datetime.strptime(value, '%Y%m%d').date()
            setattr(self, attr, value)
        for input in self.material_dict:
            self.mkt_data[input] = 0.0
            for index in self.pricing_indices:
                value = self.weight_entries[input + ":" + index].get()
                value = str2type(value, 'float')
                self.price_weights[input][index] = value
                self.mkt_data[input] += value * self.mkt_data[index]
        self.refresh_material_prices()

    def get_pricing_weight(self):
        for key, d in zip(['start', 'end', 'curr'], [self.pricing_start, self.pricing_end, self.pricing_date]):
            self.pricing_entries[key].set(datetime.datetime.strftime(d, '%Y%m%d'))
        for input in self.material_dict:
            for index in self.pricing_indices:
                value = self.price_weights[input][index]
                self.weight_entries[input + ":" + index].set(keepdigit(value, 5))

    def load_pricing_data(self):
        self.mkt_data, self.pricing_indices = bf_optimizer.get_material_prices(self.price_weights, self.pricing_start, self.pricing_end)
        if (self.pricing_date < self.pricing_start):
            self.pricing_date = min(self.mkt_data.keys())
        elif (self.pricing_date > self.pricing_end):
            self.pricing_date = max(self.mkt_data.keys())
        self.refresh_material_prices()

    def load_plant_setting(self, need_update = True):
        with open(self.plant_file, 'r') as ofile:
            self.plant_config = json.load(ofile)
            self.input_list = self.plant_config['input_list']
            if need_update:
                for input in self.input_list:
                    self.mat_selector[input].set(True)
                else:
                    self.mat_selector[input].set(False)
        if need_update:
            self.get_plant_params()

    def load_material_setting(self, need_update = True):
        with open(self.material_file, 'r') as ofile:
            mat_dict = json.load(ofile)
            self.material_dict = {}
            for input in mat_dict:
                self.material_dict[input] = bf_optimizer.BOFMaterial(mat_dict[input])
            self.sorted_materials = sorted(list(self.material_dict.keys()), key=lambda s: s.lower())
        if need_update:
            self.get_material_params(self.mat_entry_fields)

    def save_plant_setting(self):
        with open(self.plant_file, 'w') as ofile:
            json.dump(self.plant_config, ofile)

    def save_material_setting(self):
        mat_dict = {}
        for input in self.material_dict:
            mat_dict[input] = self.material_dict[input].__dict__
        with open(self.material_file, 'w') as ofile:
            json.dump(mat_dict, ofile)

    def solve_solution(self):
        self.plant_status.set(0)
        print("solving the optiomization")
        self.plant_config['coke_penalty'] = self.plant_entries['coke_penalty'].get()
        self.plant_config['chem_penalty'] = self.plant_entries['chem_penalty'].get()
        self.solution = bf_optimizer.plant_optimizer(self.plant_config, self.input_list, material_store=self.material_dict)
        for input in self.alloc_w:
            self.alloc_w[input] = self.solution['w_opt'].get(input, 0.0) * self.solution['ind_opt'].get(input, 0.0)
        self.update_solution_variables()
        self.plant_status.set(self.solution['status'])

    def get_material_params(self, key_list):
        for input in list(self.material_dict.keys()):
            for field in key_list:
                if field in bf_optimizer.CHEM_CONTENTS:
                    value = self.material_dict[input].contents[field]*100.0
                elif field in bf_optimizer.MATERIAL_ALLOC:
                    value = self.material_dict[input].flow[field]*100
                elif field in ['lower', 'upper']:
                    value = self.material_dict[input].usage[field]
                elif field in ['ref_input']:
                    value = self.plant_config[field][input]
                elif field in ['price', 'coke_coeff']:
                    value = getattr(self.material_dict[input], field)
                self.mat_entries[input][field].set(keepdigit(value,5))
            if input in self.input_list:
                self.mat_selector[input].set(True)
            else:
                self.mat_selector[input].set(False)

    def set_material_params(self, key_list):
        for input in list(self.material_dict.keys()):
            for field in key_list:
                value = self.mat_entries[input][field].get()
                value = str2type(value, 'float')
                if field in bf_optimizer.CHEM_CONTENTS:
                    self.material_dict[input].contents[field] = value/100.0
                elif field in bf_optimizer.MATERIAL_ALLOC:
                    self.material_dict[input].flow[field] = value/100.0
                elif field in ['lower', 'upper']:
                    self.material_dict[input].usage[field] = value
                elif field in ['ref_input']:
                    self.plant_config[field][input] = value
                elif field in ['price', 'coke_coeff']:
                    setattr(self.material_dict[input], field, value)
        self.input_list = [ input for input in list(self.material_dict.keys()) if self.mat_selector[input].get()]
        self.plant_config['input_list'] = self.input_list
        print(self.input_list)

    def get_plant_params(self):
        for combokey in list(self.plant_entries.keys()):
            if ':' in combokey:
                keys = combokey.split(':')
                if keys[0] in ['sinter_lowbound', 'sinter_upbound', \
                               'sinter_leakage', 'slag_target', \
                               'hot_metal_ref', 'hot_metal_target', 'bf_dry',\
                               'slag_lowbound', 'slag_upbound', 'dust_target']:
                    value = keepdigit(self.plant_config[keys[0]][keys[1]] * 100.0,3)
                else:
                    value = self.plant_config[keys[0]][keys[1]]
            else:
                value = self.plant_config[combokey]
            if combokey in ['coke_penalty', 'chem_penalty']:
                self.plant_entries[combokey].set(str(bool(value)))
            else:
                self.plant_entries[combokey].set(value)

    def set_plant_params(self):
        for combokey in list(self.plant_entries.keys()):
            value = self.plant_entries[combokey].get()
            if combokey not in ['name']:
                value = str2type(value, 'float')
            if ':' in combokey:
                keys = combokey.split(':')
                if keys[0] in ['sinter_lowbound', 'sinter_upbound', \
                               'sinter_leakage', 'slag_target', \
                               'hot_metal_ref', 'hot_metal_target', 'bf_dry',\
                               'slag_lowbound', 'slag_upbound', \
                               'dust_target']:
                    self.plant_config[keys[0]][keys[1]] = value/100.0
                else:
                    self.plant_config[keys[0]][keys[1]] = value
            else:
                self.plant_config[combokey] = value

    def chem_input(self, root):
        field_list = bf_optimizer.CHEM_CONTENTS
        scr_frame = ScrolledFrame(root)
        row_idx = 0
        set_btn = tkinter.ttk.Button(scr_frame.frame, text='Set', command= lambda: self.set_material_params(field_list))
        set_btn.grid(row=row_idx, column=0, sticky="ew")
        refresh_btn = tkinter.ttk.Button(scr_frame.frame, text='Refresh', command=lambda: self.get_material_params(field_list))
        refresh_btn.grid(row=row_idx, column=1, sticky="ew")
        row_idx += 1
        all_fields = ['selected', 'name'] + field_list
        for col_idx, field in enumerate(all_fields):
            lab = tkinter.ttk.Label(scr_frame.frame, text=field, anchor='w')
            lab.grid(column=col_idx, row=row_idx, sticky="ew")
        row_idx += 1
        for input in self.sorted_materials:
            col_idx = 0
            tkinter.ttk.Checkbutton(scr_frame.frame, variable=self.mat_selector[input], \
                            onvalue=True, offvalue=False).grid(row=row_idx, column=col_idx)
            col_idx += 1
            for field in ['name']:
                tkinter.ttk.Label(scr_frame.frame, text = str(getattr(self.material_dict[input], field))).grid(row=row_idx, column=col_idx)
                col_idx +=1
            for field in field_list:
                tkinter.ttk.Entry(scr_frame.frame, width = 10, textvariable=self.mat_entries[input][field]).grid(row=row_idx, column=col_idx)
                col_idx += 1
            row_idx += 1
        self.get_material_params(field_list)

    def material_input(self, root):
        field_list = bf_optimizer.MATERIAL_ALLOC + ['price', 'coke_coeff', 'lower', 'upper', 'ref_input']
        scr_frame = ScrolledFrame(root)
        row_idx = 0
        set_btn = tkinter.ttk.Button(scr_frame.frame, text='Set', command= lambda: self.set_material_params(field_list))
        set_btn.grid(row=row_idx, column=0, sticky="ew")
        refresh_btn = tkinter.ttk.Button(scr_frame.frame, text='Refresh', command=lambda: self.get_material_params(field_list))
        refresh_btn.grid(row=row_idx, column=1, sticky="ew")
        row_idx += 1
        all_fields = ['selected'] + self.mat_label_fields + field_list
        for col_idx, field in enumerate(all_fields):
            lab = tkinter.ttk.Label(scr_frame.frame, text=field, anchor='w')
            lab.grid(column=col_idx, row=row_idx, sticky="ew")
        row_idx += 1
        for input in self.sorted_materials:
            col_idx = 0
            tkinter.ttk.Checkbutton(scr_frame.frame, variable=self.mat_selector[input], \
                            onvalue=True, offvalue=False).grid(row=row_idx, column=col_idx)
            col_idx += 1
            for field in self.mat_label_fields:
                tkinter.ttk.Label(scr_frame.frame, text = str(getattr(self.material_dict[input], field))).grid(row=row_idx, column=col_idx)
                col_idx +=1
            for field in field_list:
                tkinter.ttk.Entry(scr_frame.frame, width = 10, textvariable=self.mat_entries[input][field]).grid(row=row_idx, column=col_idx)
                col_idx += 1
            row_idx += 1
        self.get_material_params(field_list)

    def pricing_input(self, root):
        scr_frame = ScrolledFrame(root)
        row_idx = 0
        set_btn = tkinter.ttk.Button(scr_frame.frame, text='Set param', command=lambda: self.set_pricing_weight())
        set_btn.grid(row=row_idx, column=0, sticky="ew")
        refresh_btn = tkinter.ttk.Button(scr_frame.frame, text='Refresh param', command=lambda: self.get_pricing_weight())
        refresh_btn.grid(row=row_idx, column=1, sticky="ew")
        tkinter.ttk.Label(scr_frame.frame, text='Curr Date', anchor='w').grid(column=3, row=row_idx, sticky="ew")
        tkinter.ttk.Entry(scr_frame.frame, width=10,
                  textvariable=self.pricing_entries['curr']).grid(column=4, row=row_idx, sticky="ew")
        curr_btn = tkinter.ttk.Button(scr_frame.frame, text='Use curr data', command=lambda: self.refresh_material_prices())
        curr_btn.grid(row=row_idx, column=5, sticky="ew")
        tkinter.ttk.Label(scr_frame.frame, text='Data Start', anchor='w').grid(column=7, row=row_idx, sticky="ew")
        tkinter.ttk.Entry(scr_frame.frame, width = 10,
                  textvariable = self.pricing_entries['start']).grid(column=8, row=row_idx, sticky="ew")
        tkinter.ttk.Label(scr_frame.frame, text='Data End', anchor='w').grid(column=9, row=row_idx, sticky="ew")
        tkinter.ttk.Entry(scr_frame.frame, width = 10,
                  textvariable=self.pricing_entries['end']).grid(column=10, row=row_idx, sticky="ew")
        load_btn = tkinter.ttk.Button(scr_frame.frame, text='Load data', command=lambda: self.load_pricing_data())
        load_btn.grid(row=row_idx, column=11, sticky="ew")
        row_idx += 1
        all_fields = ['Input'] + self.pricing_indices
        for col_idx, field in enumerate(all_fields):
            lab = tkinter.ttk.Label(scr_frame.frame, text=field, anchor='w')
            lab.grid(column=col_idx, row=row_idx, sticky="ew")
        for input in self.sorted_materials:
            row_idx += 1
            tkinter.ttk.Label(scr_frame.frame, text = input).grid(row=row_idx, column=0)
            for idy, index in enumerate(self.pricing_indices):
                tkinter.ttk.Entry(scr_frame.frame, width = 10, \
                          textvariable = self.weight_entries[input + ':' + index]).grid(row=row_idx, column=idy+1)

    def plant_input(self, root):
        scr_frame = ScrolledFrame(root)
        row_idx = 0
        col_idx = 0
        lbf = tk.LabelFrame(scr_frame.frame, text="Control", padx=0, pady=0, width=8, height=8)
        lbf.grid(row=row_idx, column=col_idx)
        set_btn = tkinter.ttk.Button(lbf, text='Set', command= self.set_plant_params)
        set_btn.grid(columnspan = 2, row=row_idx, column=0, sticky="ew")
        refresh_btn = tkinter.ttk.Button(lbf, text='Refresh', command=self.get_plant_params)
        refresh_btn.grid(columnspan = 2, row=row_idx, column=2, sticky="ew")
        solve_btn = tkinter.ttk.Button(lbf, text='Solve', command=self.solve_solution)
        solve_btn.grid(columnspan = 2, row=row_idx, column=4, sticky="ew")
        tk.Label(lbf, width=10, text='Coke Penalty', anchor='w').grid(column=6, \
                                                                      columnspan=2, row=row_idx, sticky="ew")
        tkinter.ttk.Checkbutton(lbf, variable=self.plant_entries['coke_penalty'], \
                        onvalue=True, offvalue=False).grid(row=row_idx, column=8)
        tk.Label(lbf, width=10, text='Chem Penalty', anchor='w').grid(column=10, \
                                            columnspan=2, row=row_idx, sticky="ew")
        tkinter.ttk.Checkbutton(lbf, variable=self.plant_entries['chem_penalty'], \
                        onvalue=True, offvalue=False).grid(row=row_idx, column=12)
        tk.Label(lbf, width=10, text='Status:', anchor='w').grid(column=14, \
                        columnspan=2, row=row_idx, sticky="ew")
        tk.Label(lbf, width=10, textvariable = self.plant_status, anchor='w').grid(column=16, \
                        columnspan=2, row=row_idx, sticky="ew")

        row_idx += 2
        lbf = tk.LabelFrame(scr_frame.frame, text= "General", padx=0, pady=0, width=8, height=8)
        lbf.grid(row = row_idx, column = col_idx)
        col_idx = 0
        field_list = [('name', 'Plant name'), \
                      ('target_production', 'Target Prod'), \
                      ('ref_production', 'Ref Prod'), \
                      ('burnt_lime_ratio', 'Burnt Lime Usage'), \
                      ('RDI:min', 'RDI min'), \
                      ('RDI:max', 'RDI max'), ]
        for idx, field in enumerate(field_list):
            tk.Label(lbf, width = 23, text = field[1], anchor='w').grid(column=(idx % 2) * 8, \
                                                columnspan = 4, row=row_idx + int(idx/2), sticky="ew")
            tk.Entry(lbf, width = 20, textvariable=self.plant_entries[field[0]]).grid(column=(idx % 2) * 8 + 4, \
                                                columnspan = 4, row=row_idx + int(idx/2), sticky="ew")
        row_idx += int(len(field_list)/2)+1
        lbf = tk.LabelFrame(scr_frame.frame, text="Iron Ores", padx=0, pady=0, width=100, height=30)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = ['max_num', 'low_bound', 'up_bound']
        product_list = ['pellet', 'lump', 'sinter',]
        for fname in ['Product'] + product_list:
            tk.Label(lbf, text = fname, width = 20, anchor='w').grid(column=col_idx, row=row_idx, columnspan = 4,sticky="ew")
            col_idx += 4
        row_idx += 2
        for field in field_list:
            col_idx = 0
            tk.Label(lbf, text= field.replace('_', ' ').title(), width = 20, anchor='w').grid(columnspan = 4,\
                                                                column=col_idx, row=row_idx, sticky="ew")
            for prod in product_list:
                col_idx += 4
                cbfield = prod + ":" + field
                tk.Entry(lbf, width = 18, textvariable=self.plant_entries[cbfield]).grid(columnspan = 4,\
                                                                column=col_idx, row=row_idx, sticky="ew")
            row_idx += 2
        col_idx = 0
        lbf = tk.LabelFrame(scr_frame.frame, text="Coke Coeff", padx=0, pady=0, width=100, height=20)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = [('ref_rate', 'Coke rate per HM', 'ton/ton HM'), \
                      ('ref_slag', 'Slag vol per HM', 'ton/ton HM'), \
                      ('PCI_rate', 'PCI usage per HM', 'ton/ton HM'), \
                      ('slag', 'coke coeff for slag', 'ton/ton slag'),\
                      ('PCI', 'coke coeff for PCI', 'ton/ton PCI'), ]
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text= field[1], width = 21, anchor='w').grid(columnspan = 4, \
                column=(idx % 2) * 6, row = row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Entry(lbf, width = 10, textvariable=self.plant_entries['coke_coeff:'+field[0]]).grid(\
                column= (idx % 2) * 6 + 4, row=row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Label(lbf, text=field[2], width=10, anchor='w').grid(\
                column= (idx % 2) * 8 + 5, row=row_idx + int(idx / 2) * 2, sticky="ew")
        row_idx += int(len(field_list)/2)*2 + 2
        lbf = tk.LabelFrame(scr_frame.frame, text="Sinter", padx=0, pady=0, width=100, height=20)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = [('price', 'Price', '$/ton'), \
                      ('cms', 'CMS sinter fuel usage', 'ton/ton'), \
                      ('FeO_target', 'FeO target ratio', ''), \
                      ('yield', 'Sinter yield', ''), \
                      ('work_rate', 'Work rate', ''), \
                      ('sieved', 'Sieving ratio', ''), \
                      ('conc_max_rate', 'Concenrate usage', ''),]
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text= field[1], width = 20, anchor='w').grid(columnspan = 4,\
                column=(idx % 2) * 8, row = row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Entry(lbf, width = 10, textvariable=self.plant_entries['sinter:'+field[0]]).grid(columnspan = 2, \
                column=(idx % 2) * 8 + 4, row=row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Label(lbf, text=field[2], width=10, anchor='w').grid(columnspan = 2, \
                column=(idx % 2) * 8 + 6, row=row_idx + int(idx / 2) * 2, sticky="ew")
        row_idx += int(len(field_list) / 2) * 2 + 2
        field_list = ['S', 'Zn', 'Pb', 'Alkali']
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text= field + " esacpe", width = 20, anchor='w').grid(columnspan = 4, \
                column=(idx % 2) * 8, row=row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Entry(lbf, width = 10, textvariable=self.plant_entries['sinter_leakage:'+field]).grid(columnspan = 2, \
                column=(idx % 2) * 8 + 4, row=row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Label(lbf, text='%', width=10, anchor='w').grid( columnspan = 2,\
                column=(idx % 2) * 8 + 6, row=row_idx + int(idx / 2) * 2, sticky="ew")
        row_idx += int(len(field_list) / 2) * 2 + 2
        field_list = self.plant_output['sinter_ratio'][:-1]
        for idx, field in enumerate(['(%)', 'Min', 'Max',]):
            tk.Label(lbf, text=field, anchor='w').grid(column= 0, row = row_idx + idx, sticky="ew")
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text=field, anchor='w').grid(column= idx + 1, row=row_idx, sticky="ew")
            tk.Entry(lbf, width=5, textvariable=self.plant_entries['sinter_lowbound:' + field]).grid( \
                column= idx + 1, row= row_idx + 1, sticky="ew")
            tk.Entry(lbf, width=5, textvariable=self.plant_entries['sinter_upbound:' + field]).grid( \
                column= idx + 1, row=row_idx + 2, sticky="ew")
        row_idx += 3
        lbf = tk.LabelFrame(scr_frame.frame, text="Blast furnace input", padx=0, pady=0, width=100, height=20)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = ['K2O', 'Na2O', 'Pb', 'Zn']
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text=field, width = 9, anchor='w').grid(column= idx * 4, row=row_idx, sticky="ew")
            tk.Entry(lbf, width = 12, textvariable=self.plant_entries['bf_dry:' + field]).grid( \
                column= idx * 4 + 2, row= row_idx, sticky="ew")
        row_idx += 1
        lbf = tk.LabelFrame(scr_frame.frame, text="Hot metal constraints", padx=0, pady=0, width=100, height=20)
        lbf.grid(row=row_idx, column=col_idx)
        row_list = ['P', 'Mn', 'S', 'Alkali', 'C']
        field_list = [('hot_metal_target', 'Target (%)'), ('hot_metal_ref', 'Ref (%)'), ('hot_metal_penalty', 'Penalty')]
        for idx, field in enumerate([('','Chemicals')] + field_list):
            tk.Label(lbf, text=field[1], width = 20, anchor='w').grid(column= idx * 4, row=row_idx, sticky="ew")
        for r_idx, row_field in enumerate(row_list):
            tk.Label(lbf, text=row_field, width=20, anchor='w').grid(column = 0, row=row_idx + r_idx + 1, sticky="ew")
            for idx, field in enumerate(field_list):
                key = field[0] + ':' + row_field
                if key in self.plant_entries:
                    tk.Entry(lbf, width=20, textvariable=self.plant_entries[key]).grid( \
                                column = (idx + 1) * 4 , row=row_idx + r_idx + 1, sticky="ew")
        row_idx += 6
        lbf = tk.LabelFrame(scr_frame.frame, text="Slag constraints", padx=0, pady=0, width=100, height=20)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = [('slag:price', 'Price', '$/ton'), \
                      ('slag:fraction', 'Slag Fraction', ''), \
                      ('slag:Mn_yield', 'Mn yield', ''), \
                      ('slag:cement_idx', 'Cement Index', ''), \
                      ('slag_target:Fe', 'Fe pct', '%'), \
                      ('slag_target:P', 'P pct', '%'), ]
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text=field[1], width=21, anchor='w').grid(columnspan=4, \
                                column=(idx % 2) * 8, row=row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Entry(lbf, width=10, textvariable=self.plant_entries[field[0]]).grid(columnspan=2, \
                                column=(idx % 2) * 8 + 4, row=row_idx + int(idx / 2) * 2, sticky="ew")
            tk.Label(lbf, text=field[2], width=10, anchor='w').grid(columnspan=2, \
                                column=(idx % 2) * 8 + 6, row=row_idx + int(idx / 2) * 2, sticky="ew")
        row_idx += int(len(field_list) / 2) * 2 + 2
        field_list = ['usage', 'Al2O3', 'MgO', 'Bas', 'Tbas']
        for idx, field in enumerate(['(%)', 'Min', 'Max',]):
            tk.Label(lbf, text=field, anchor='w').grid(column= 0, row = row_idx + idx, sticky="ew")
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text=field, anchor='w').grid(column= (idx + 1)*3, row=row_idx, sticky="ew")
            tk.Entry(lbf, width=5, textvariable=self.plant_entries['slag_lowbound:' + field]).grid( \
                column= (idx + 1)*3, row= row_idx + 1, sticky="ew")
            tk.Entry(lbf, width=5, textvariable=self.plant_entries['slag_upbound:' + field]).grid( \
                column= (idx + 1)*3, row=row_idx + 2, sticky="ew")
        row_idx += 3
        lbf = tk.LabelFrame(scr_frame.frame, text="Dust", padx=0, pady=0, width=100, height=20)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = ['usage', 'Fe', 'Mn', 'CaO', 'SiO2', 'Al2O3', 'MgO', 'P', 'Zn', 'K2O', 'Na2O', 'C', 'Pb']
        for idx, field in enumerate(['(%)', 'Target',]):
            tk.Label(lbf, text=field,  width=6, anchor='w').grid(column= 0, row = row_idx + idx, sticky="ew")
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text=field,  width=5, anchor='w').grid(column= (idx + 2), row=row_idx, sticky="ew")
            tk.Entry(lbf, width=5, textvariable=self.plant_entries['dust_target:' + field]).grid( \
                column= (idx + 2), row= row_idx + 1, sticky="ew")
        self.get_plant_params()

    def solution_settings(self, root):
        field_list = ['name'] + self.mat_output_fields
        scr_frame = ScrolledFrame(root)
        row_idx = 0
        for col_idx, field in enumerate(field_list):
            lab = tkinter.ttk.Label(scr_frame.frame, text=field, anchor='w')
            lab.grid(column=col_idx, row=row_idx, sticky="ew")
        for input in self.sorted_materials:
            col_idx = 0
            row_idx += 1
            for field in field_list:
                if field in ['name']:
                    vargs = {'text': input}
                else:
                    vargs = {'textvariable': self.mat_entries[input][field]}
                tkinter.ttk.Label(scr_frame.frame, **vargs).grid(row=row_idx, column=col_idx)
                col_idx += 1

    def update_solution_variables(self):
        for input in self.material_dict:
            for field in self.mat_output_fields:
                field_sp = field.split('_')
                if field_sp[1] in ['w']:
                    self.mat_entries[input][field].set(keepdigit(self.alloc_w[input], 4))
                elif field_sp[1] in ['cost']:
                    self.mat_entries[input][field].set(keepdigit(self.alloc_w[input]\
                                                                 *self.material_dict[input].price,1))
                elif field_sp[1] in ['fbed', 'cbed', 'bof', 'sieved', ]:
                    self.mat_entries[input][field].set(keepdigit(self.alloc_w[input] \
                                                                 * self.material_dict[input].flow[field_sp[1]], 4))
                else:
                    print('unknown field')
        self.optimal_results = {}
        plant = self.solution['plant']
        w = self.solution['w_opt']
        ind = self.solution['ind_opt']
        self.optimal_results['prod_info'] = {\
            'slag_volume': plant.slag_prod(w), \
            'slag_cost': -plant.slag_cost(w),  \
            'slag_mean': -plant.slag_cost(w)/plant.target_prod,\
            'sinter_volume': plant.sinter_prod(w), \
            'sinter_cost': plant.sinter_cost(w), \
            'sinter_mean': plant.sinter_cost(w)/plant.target_prod, \
            'material_volume': sum([w[input] for input in plant.material_dict]), \
            'material_cost': sum([w[input] * plant.material_dict[input].price for input in plant.material_dict]), \
            'material_mean': sum([w[input] * plant.material_dict[input].price for input in plant.material_dict])/plant.target_prod, \
            'hotmetal_volume': '', \
            'hotmetal_cost': plant.hm_penalty_cost(w), \
            'hotmetal_mean': plant.hm_penalty_cost(w)/plant.target_prod, \
            'total_cost': plant.total_cost(w),\
            'total_volume': '',\
            'total_mean': plant.total_cost(w) / plant.target_prod, \
            'conc_ratio': plant.concentrate_vol(w)/(plant.concentrate_vol(w) + plant.sinterfeed_vol(w)),}
        self.optimal_results['mat_info'] = {}
        for idx, prod in enumerate(['Sinfeed', 'Cencentrate', 'Lump', 'Pellet', 'Sinfuel', 'BFCoke', 'Additive']):
            self.optimal_results['mat_info']['%s_num' % (prod)] = sum([ind[input] for input in plant.input_list \
                                        if (plant.material_dict[input].ptype == idx) & (input in ind)])
            self.optimal_results['mat_info']['%s_volume' % (prod)] = sum([w[input] for input in plant.input_list \
                                        if (plant.material_dict[input].ptype == idx) & (input in w)])
            self.optimal_results['mat_info']['%s_cost' % (prod)] = sum([w[input] * self.material_dict[input].price \
                                        for input in plant.input_list if (plant.material_dict[input].ptype == idx) & (input in w)])
            self.optimal_results['mat_info']['%s_mean' % (prod)] = self.optimal_results['mat_info']['%s_cost' % (prod)]/plant.target_prod
        self.optimal_results['sinter_ratio'] = {}
        for attr in self.plant_output['sinter_ratio'][:-2]:
            self.optimal_results['sinter_ratio'][attr] = plant.sinter_chem_vol(w, attr)/plant.sinter_prod(w) * 100.0
        self.optimal_results['sinter_ratio']['Bas'] = plant.sinter_chem_vol(w, 'CaO')/plant.sinter_chem_vol(w, 'SiO2') * 100.0
        self.optimal_results['sinter_ratio']['RDI'] = plant.RDIB(w)/plant.sinter_prod(w) + bf_optimizer.RDI_global['coeff']
        self.optimal_results['slag_ratio'] = {}
        for attr in ['Al2O3', 'MgO', 'CaO', 'SiO2', 'Mn']:
            self.optimal_results['slag_ratio'][attr] = plant.slag_chemical(w, attr)/plant.slag_prod(w) * 100.0
        self.optimal_results['slag_ratio']['Bas'] = plant.slag_chemical(w, 'CaO')/plant.slag_chemical(w, 'SiO2') * 100.0
        self.optimal_results['slag_ratio']['Tbas'] = (plant.slag_chemical(w, 'CaO') + plant.slag_chemical(w, 'MgO'))/ \
                                                     (plant.slag_chemical(w, 'SiO2') + plant.slag_chemical(w, 'Al2O3')) * 100.0
        self.optimal_results['slag_ratio']['usage'] = plant.slag_prod(w) / plant.target_prod * 100.0
        self.optimal_results['slag_ratio']['Cement'] = (plant.slag_chemical(w, 'CaO') + plant.slag_chemical(w, 'Al2O3') * 0.56 \
                        + plant.slag_chemical(w, 'MgO') * 1.4)/plant.slag_chemical(w, 'SiO2')

        self.optimal_results['bf_input'] = {}
        for attr in ['Zn', 'Pb', 'K2O', 'Na2O']:
            self.optimal_results['bf_input'][attr] = plant.bf_chem_content(w, attr)/plant.target_prod* 100.0
        self.optimal_results['bf_input']['Fe'] = plant.hot_metal_Fe(w) / plant.target_prod* 100.0
        self.optimal_results['bf_input']['P'] = plant.hot_metal_P(w) / plant.target_prod* 100.0
        self.optimal_results['bf_input']['Mn'] = plant.hot_metal_Mn(w) / plant.target_prod* 100.0
        self.optimal_results['bf_input']['Alkali'] = plant.hot_metal_Alkali(w)/plant.target_prod* 100.0

        for key1 in self.optimal_results:
            if type(self.optimal_results[key1]).__name__ == 'dict':
                for key2 in self.optimal_results[key1]:
                    self.output_entries[key1][key2].set(keepdigit(self.optimal_results[key1][key2], 3))
            else:
                self.output_entries[key1].set(keepdigit(self.optimal_results[key1]), 3)


    def calculated_settings(self, root):
        scr_frame = ScrolledFrame(root)
        row_idx = 0
        col_idx = 0
        lbf = tk.LabelFrame(scr_frame.frame, text="Production Summary", padx=0, pady=0, width=8, height=8)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = ['cost', 'volume', 'mean']
        item_list = ['total', 'material', 'sinter', 'hotmetal', 'slag']
        for fname in ['Item'] + item_list:
            tk.Label(lbf, text=fname, width=8, anchor='w').grid(column=col_idx, row=row_idx, columnspan=1, sticky="ew")
            col_idx += 1
        row_idx += 1
        for field in field_list:
            col_idx = 0
            tk.Label(lbf, text=field, width=6, anchor='w').grid(columnspan=1, column=col_idx, row=row_idx,
                                                                sticky="ew")
            for idx, prod in enumerate(item_list):
                col_idx += 1
                cbfield = prod + "_" + field
                tk.Label(lbf, width=10, textvariable=self.output_entries['prod_info'][cbfield]).grid(columnspan=1, \
                                                                                       column=col_idx, row=row_idx,
                                                                                       sticky="ew")
            row_idx += 1
        col_idx = 0
        lbf = tk.LabelFrame(scr_frame.frame, text="Material Summary", padx=0, pady=0, width=8, height=8)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = ['num', 'volume', 'cost', 'mean']
        product_list = ['Sinfeed', 'Cencentrate', 'Lump', 'Pellet', 'Sinfuel', 'BFCoke', 'Additive']
        for fname in ['Input'] + product_list:
            tk.Label(lbf, text=fname, width=8, anchor='w').grid(column=col_idx, row=row_idx, columnspan=1, sticky="ew")
            col_idx += 1
        row_idx += 1
        for field in field_list:
            col_idx = 0
            tk.Label(lbf, text=field, width=6, anchor='w').grid(columnspan=1, column=col_idx, row=row_idx,
                                                                sticky="ew")
            for idx, prod in enumerate(product_list):
                col_idx += 1
                cbfield = prod + "_" + field
                tk.Label(lbf, width=10, textvariable=self.output_entries['mat_info'][cbfield]).grid(columnspan=1, \
                                                                                       column=col_idx, row=row_idx,
                                                                                       sticky="ew")
            row_idx += 1
        col_idx = 0
        lbf = tk.LabelFrame(scr_frame.frame, text="Sinter Constraints", padx=0, pady=0, width=8, height=8)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = self.plant_output['sinter_ratio']
        for idx, field in enumerate(['(%)', 'Min', 'Curr', 'Max']):
            tk.Label(lbf, text=field, anchor='w').grid(column=0, row=row_idx + idx, sticky="ew")
        for idx, field in enumerate(field_list):
            tk.Label(lbf, text=field, anchor='w').grid(column=idx + 1, row=row_idx, sticky="ew")
            if field == 'RDI':
                tk.Label(lbf, width=5, textvariable=self.plant_entries['RDI:min']).grid( column=idx + 1, row=row_idx + 1, sticky="ew")
            else:
                tk.Label(lbf, width=5, textvariable=self.plant_entries['sinter_lowbound:' + field]).grid( \
                        column=idx + 1, row=row_idx + 1, sticky="ew")
            tk.Label(lbf, width=5, textvariable=self.output_entries['sinter_ratio'][field]).grid( \
                        column=idx + 1, row=row_idx + 2, sticky="ew")
            if field == 'RDI':
                tk.Label(lbf, width=5, textvariable=self.plant_entries['RDI:max']).grid(column=idx + 1, row=row_idx + 3, sticky="ew")
            else:
                tk.Label(lbf, width=5, textvariable=self.plant_entries['sinter_upbound:' + field]).grid( \
                        column=idx + 1, row=row_idx + 3, sticky="ew")
        row_idx += 4

        lbf = tk.LabelFrame(scr_frame.frame, text="Slag Constraints", padx=0, pady=0, width=8, height=8)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = ['usage', 'Bas', 'Tbas', 'Al2O3', 'MgO', 'CaO', 'SiO2', 'Mn', 'Cement']
        for idx, field in enumerate(['Curr', 'Min', 'Max']):
            tk.Label(lbf, text=field, width=6, anchor='w').grid(column=0, row=row_idx + idx + 1, sticky="ew")
        min_list = [self.plant_entries['slag_lowbound:' + key] for key in ['usage', 'Bas', 'Tbas', 'Al2O3', 'MgO']] \
                   + [None, None, None, self.plant_entries['slag:cement_idx']]
        max_list = [self.plant_entries['slag_upbound:' + key] for key in ['usage', 'Bas', 'Tbas', 'Al2O3', 'MgO']] \
                   + [None, None, None, None]
        for idx, (field, min_entry, max_entry) in enumerate(zip(field_list, min_list, max_list)):
            tk.Label(lbf, width=8, text=field, anchor='w').grid(column=idx + 1, row=row_idx, sticky="ew")
            tk.Label(lbf, width=8, textvariable= self.output_entries['slag_ratio'][field]).grid(column=idx + 1, row=row_idx + 1, sticky="ew")
            tk.Label(lbf, width=8, textvariable= min_entry).grid(column=idx + 1, row=row_idx + 2, sticky="ew")
            tk.Label(lbf, width=8, textvariable= max_entry).grid(column=idx + 1, row=row_idx + 3, sticky="ew")
        row_idx += 4

        lbf = tk.LabelFrame(scr_frame.frame, text="Furnace Constraints", padx=0, pady=0, width=8, height=8)
        lbf.grid(row=row_idx, column=col_idx)
        field_list = ['Zn', 'Pb', 'K2O', 'Na2O', '', '', 'Fe', 'Mn', 'P', 'Alkali']
        for idx, field in enumerate(['Curr', 'Max']):
            tk.Label(lbf, text=field, width=7, anchor='w').grid(column=0, row=row_idx + idx + 1, sticky="ew")
        max_list = [self.plant_entries['bf_dry:' + key] for key in ['Zn', 'Pb', 'K2O', 'Na2O']] \
                    + [None, None, None] \
                    + [self.plant_entries['hot_metal_target:' + key] for key in ['Mn', 'P']] + [None]
        for idx, (field, max_entry) in enumerate(zip(field_list, max_list)):
            tk.Label(lbf, width=7, text=field, anchor='w').grid(column=idx + 1, row=row_idx, sticky="ew")
            if len(field)<=0:
                continue
            tk.Label(lbf, width=7, textvariable = self.output_entries['bf_input'][field]).grid(column=idx + 1, \
                                        row = row_idx + 1, sticky="ew")
            tk.Label(lbf, width=7, textvariable=max_entry).grid(column=idx + 1, row=row_idx + 2, sticky="ew")
        row_idx += 4

    def onExit(self):
        self.destroy()


def run_gui(conf):
    myGui = BFOptGui(conf, load_config = True)
    myGui.iconbitmap(r'data\\cargill.ico')
    myGui.mainloop()

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 2:
        material_file = args[1]
    else:
        material_file = 'data\\material_config.json'
    if len(args) >= 1:
        plant_file = args[0]
    else:
        plant_file = 'data\\plant_config.json'

    conf = {'plant_file': plant_file, \
            'material_file': material_file, \
            'price_map': 'SEABORNE_PRICE_MAP',}
    run_gui(conf)