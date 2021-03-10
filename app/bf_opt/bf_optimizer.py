# -*- coding: utf-8 -*-
import pulp
import os
import ts_tool

CHEM_CONTENTS = ['H2O', 'Fe', 'Mn', 'CaO', 'SiO2', 'Al2O3','MgO', 'P', 'S', 'Zn', 'K2O', \
                 'Na2O', 'VV', 'C', 'Pb', 'FeO']

MATERIAL_ALLOC = ['fbed', 'cbed', 'bof', 'sieved']

class BFProductType:
    SinterFeed, Concentrate, Lump, Pellet, SinterFuel, BFCoke, Additive = list(range(7))

RDI_global = {'min': 0.0, 'max': 0.0, 'coeff': 38.3, 'Al2O3': 24.3, 'FeO': -0.9, 'Gangue': -1.9, }

class BOFMaterial(object):
    common_params = {'contents': dict([ (x, 0.0) for x in CHEM_CONTENTS]), \
                     'name': 'Unknown', 'price': 0.0, 'tcost': 0.0, 'ptype': 0, \
                     'flow': {'fbed': 0.0, 'cbed': 1.0, 'bof': 0.0, 'sieved': 0.0}, \
                     'coke_coeff': 0.0, # KBFCkLumpI\
                     'usage': {'upper': 1000000.0, 'lower': 0.0,}, \
                     }
    def __init__(self, spec_dict = {}):
        self.load_config(spec_dict)

    def load_config(self, spec_dict):
        d = self.__dict__
        for key in self.common_params:
            if type(self.common_params[key]).__name__ == 'dict':
                data_dict = spec_dict.get(key, self.common_params[key])
                d[key] = {}
                for elem in data_dict:
                    d[key][elem] = data_dict.get(elem, 0.0)
            else:
                d[key] = spec_dict.get(key, self.common_params[key])


class BOFPlant(object):
    def __init__(self, config):
        self.load_config(config)

    def load_config(self, config):
        self.name = config.get('name', 'dummy')
        self.input_list = config.get('input_list', [])
        self.ref_usage = config.get('ref_input', {})
        self.material_dict = {}
        self.ore_list = []
        self.target_prod = config.get('target_production', 100.0)
        self.ref_prod = config.get('ref_production', 100.0)
        self.coke_coeff = config.get('coke_coeff', {'slag': 0.0, 'PCI': 0.0, 'ref_PCI': 0.2, 'PCI_rate': 0.2,\
                                                    'ref_rate': 0.312, 'ref_slag': 0.265,})
        self.burnt_lime_ratio = config.get('burnt_lime_ratio', 0.0)
        self.pellet = config.get('pellet', {'max_num': 5, 'up_bound': 2.0, 'low_bound': 0.0})
        self.lump = config.get('lump', {'max_num': 5, 'up_bound': 2.0, 'low_bound': 0.0})
        self.sinter = config.get('sinter', {'price': 5.0, 'cms': 0.05, 'FeO_target': 0.075,\
                                            'yield': 0.0, 'work_rate': 0.965, \
                                            'sieved': 0.145, 'up_bound': 1.7, 'low_bound': 0.0,\
                                            'max_num': 10, 'conc_max_rate': 1.0, })
        self.sinter_leakage = config.get('sinter_leakage', {'S': 0.85, 'Zn': 0.0, 'Pb': 0.0, 'Alkali': 0.0})
        self.sinter_lowbound = config.get('sinter_lowbound', \
                                          {'Fe': 0.55, 'Mn': 0.0, 'CaO': 0.0, 'SiO2': 0.0, \
                                           'Al2O3': 0.0, 'MgO': 0.0, 'P': 0.0, 'S': 0.0, \
                                           'Zn': 0.0, 'K2O': 0.0, 'Na2O': 0.0, 'Pb': 0.0, \
                                           'Bas': 1.0})
        self.sinter_upbound = config.get('sinter_upbound', \
                                          {'Fe': 0.6, 'Mn': 1.0, 'CaO': 1.0, 'SiO2': 1.0, \
                                           'Al2O3': 1.0, 'MgO': 1.0, 'P': 1.0, 'S': 1.0, \
                                           'Zn': 1.0, 'K2O': 1.0, 'Na2O': 1.0, 'Pb': 1.0, \
                                           'Bas': 4.0})
        self.bf_dry = config.get('bf_dry', {'K2O': 0.00133, 'Na2O': 0.00066, \
                                            'Pb': 0.0001, 'Zn': 0.00018, })
        self.hm_ref = config.get('hot_metal_ref', {'P': 0.00092, 'Mn': 0.0028, 'S': 0.00018, \
                                                         'Alkali': 0.005,})
        self.hm_target = config.get('hot_metal_target', {'P': 0.0011, 'Mn': 0.0035, 'S': 0.00015, \
                                                         'C': 0.048, 'Si': 0.00385, })
        self.hm_penalty = config.get('hot_metal_penalty',{'S': 0.192, 'P': 0.047, 'Mn': 0.006, 'Alkali': 0.0})
        self.slag = config.get('slag', {'fraction': 0.979, 'price': 15.36, \
                                        'Mn_yield': 0.77, 'cement_idx': 1.74,})
        self.slag_lowbound = config.get('slag_lowbound', \
                                       {'usage': 0.2, 'Al2O3': 0.085, 'MgO': 0.0995, 'Bas': 1.1, 'Tbas': 1.0, })
        self.slag_upbound = config.get('slag_upbound', \
                                       {'usage': 0.3, 'Al2O3': 0.145, 'MgO': 0.11, 'Bas': 1.3, 'Tbas': 1.18, })
        self.slag_target = config.get('slag_target', {'Fe': 0.005, 'P': 0.0003})
        self.dust_target = config.get('dust_target', {'Fe': 0.44, 'Mn': 0.001, 'CaO': 0.026, 'SiO2': 0.058, \
                                        'Al2O3': 0.024, 'MgO': 0.006, 'P': 0.0014,  \
                                        'Zn': 0.006, 'K2O': 0.005, 'Na2O': 0.0032, 'C': 0.0, 'Pb': 0.0, \
                                        'usage': 0.009, })
        self.RDI = config.get('RDI', {'min': 0.0, 'max': 35.0})
        self.chem_penalty = config.get('chem_penalty', False)
        self.coke_penalty = config.get('coke_penalty', False)

    def load_materials(self, material_dict):
        for input in self.input_list:
            self.material_dict[input] = material_dict[input]
            if self.material_dict[input].ptype <= 3:
                self.ore_list.append(self.material_dict[input])

    def setup_varirables(self):
        self.w = pulp.LpVariable.dicts('w', self.input_list, lowBound = 0, upBound = self.target_prod * 3.0)
        self.ind = pulp.LpVariable.dicts('ind', self.input_list, cat = 'Binary')

    # A(st,GRONDSTOFFEN)
    def sinter_prod_coeff(self, bf_input):
        mobj = self.material_dict[bf_input]
        if mobj.ptype == BFProductType.BFCoke:
            return 0
        else:
            return (1 - mobj.contents['H2O']) * (1 - self.sinter_leakage['S'] * mobj.contents['S'] - mobj.contents['VV'] \
                                     - self.sinter_leakage['Zn'] * mobj.contents['Zn'] \
                                     - self.sinter_leakage['Pb'] * mobj.contents['Pb'] \
                                     - self.sinter_leakage['Alkali'] * (mobj.contents['K2O'] + mobj.contents['Na2O']) \
                                     + mobj.contents['FeO']/7.0)/( 1 + self.sinter['FeO_target']/7.0)

    # A_CHEMICAL(st, j, i)
    def input_chemical(self, bf_input, attr):
        mobj = self.material_dict[bf_input]
        if attr == 'C':
            return (1 - mobj.contents['H2O']) * (1 - mobj.contents['C'])
        elif attr in ['Pb', 'Zn', 'S', 'K2O', 'Na2O']:
            xattr = attr
            if xattr in ['K2O', 'Na2O']:
                xattr = 'Alkali'
            return (1 - mobj.contents['H2O']) * (1 - self.sinter_leakage[xattr])
        else:
            return (1 - mobj.contents['H2O'])

    # PERCSINBT
    def sinter_chem_vol(self, w, attr):
        if attr == 'Fe':
            return (self.sinter_prod(w) \
                    - ( sum([w[mobj.name] * mobj.flow['fbed'] \
                        * (1-mobj.contents['H2O']) \
                        * (mobj.contents['Mn'] * (54.938+15.999)/54.938 \
                           + mobj.contents['CaO'] \
                           + mobj.contents['SiO2'] \
                           + mobj.contents['MgO'] \
                           + mobj.contents['Al2O3'] \
                           + mobj.contents['S'] * ( 1- self.sinter_leakage['S']) \
                           + (mobj.contents['Na2O'] \
                              + mobj.contents['K2O']) * (1-self.sinter_leakage['Alkali'])\
                           + mobj.contents['Zn'] * (1-self.sinter_leakage['Zn']) \
                                            * (65.37+15.999)/65.37 \
                           + mobj.contents['Pb'] * (1-self.sinter_leakage['Pb']) \
                                            * (207.19+15.999)/207.19 \
                           + mobj.contents['P'] * (2*30.973+5*15.999)/(2*30.973))
                        for mobj in list(self.material_dict.values()) if mobj.ptype != BFProductType.BFCoke]) \
                    + self.sinter_prod(w) * self.sinter['FeO_target'] * 1.28))*2*55.847/(2*55.847+3*15.999) \
                + self.sinter_prod(w) * self.sinter['FeO_target']*1.28*55.847/(55.847+15.999) # \
                # - self.sinter_prod(w) * 0.001
        elif attr == 'C':
            return 0.0
        elif attr in CHEM_CONTENTS: #['S', 'Zn', 'Pb', 'Na2O', 'K2O']:
            return sum([w[mobj.name] * mobj.flow['fbed'] * self.input_chemical(mobj.name, attr) * mobj.contents[attr] \
                        for mobj in list(self.material_dict.values()) if mobj.ptype != BFProductType.BFCoke])
        #elif attr in CHEM_CONTENTS:
        #    return sum([w[mobj.name] * mobj.flow['fbed'] * self.input_chemical(input, attr) * mobj.contents[attr] \
        #                for mobj in self.material_dict.values() if mobj.ptype != BFProductType.BFCoke])
        else:
            print("Unknown element=%s" % attr)
            return 0.0

    # PARSINB
    def sinter_prod(self, w):
        return sum([ self.sinter_prod_coeff(mobj.name) * w[mobj.name] *  mobj.flow['fbed'] \
                    for mobj in list(self.material_dict.values()) if mobj.ptype != BFProductType.BFCoke])

    def sinter_cost(self, w):
        return self.sinter_prod(w) * self.sinter['price'] /( 1- self.sinter['sieved'])

    # RDIB(st)
    def RDIB(self, w):
        return 100 * (RDI_global['Al2O3'] * self.sinter_chem_vol(w, 'Al2O3') \
                      + RDI_global['FeO'] * self.sinter_chem_vol(w, 'FeO') \
                      + RDI_global['Gangue'] * (self.sinter_chem_vol(w, 'SiO2')  \
                                                + self.sinter_chem_vol(w, 'CaO') \
                                                + self.sinter_chem_vol(w, 'MgO')))

    # SumPellet(st)
    def sum_pellet(self, w):
        return sum([w[input] for input in self.material_dict if self.material_dict[input].ptype == BFProductType.Pellet])

    def sum_lump(self, w):
        return sum([w[input] for input in self.material_dict if self.material_dict[input].ptype == BFProductType.Lump])


    def dust_alkali(self):
        return self.dust_target['CaO'] + self.dust_target['SiO2'] + self.dust_target['Al2O3'] + self.dust_target['MgO']

    def bf_chem_content(self, w, key):
        if key == 'H2O':
            return sum([mobj.contents[key] * w[mobj.name] for mobj in list(self.material_dict.values())])
        else:
            return sum([mobj.contents[key] * w[mobj.name] * \
                        (1 - mobj.contents['H2O']) for mobj in list(self.material_dict.values())])

    def hot_metal_Alkali(self, w):
        return self.bf_chem_content(w, 'Na2O') + self.bf_chem_content(w, 'K2O')

    # PCMNRYB(st)
    def hot_metal_Mn(self, w):
        return self.slag['Mn_yield'] * sum([mobj.contents['Mn'] \
                    * ( 1 - mobj.contents['H2O']) * w[mobj.name] \
                                            for mobj in list(self.material_dict.values())])

    # PCPRYB(st)
    def hot_metal_P(self, w):
        return sum([self.material_dict[input].contents['P'] * w[input]\
                    * ( 1-self.material_dict[input].contents['H2O']) for input in self.material_dict]) \
                    - self.slag_prod(w) * self.slag_target['P'] \
                    - self.dust_target['usage'] * self.target_prod * self.dust_target['P']

    # PCFERYB(st)
    def hot_metal_Fe(self, w):
        return self.sinter_chem_vol(w, 'Fe') \
                + sum([mobj.contents['Fe'] * w[mobj.name] * (1 - mobj.contents['H2O']) \
                * (mobj.flow['cbed'] * (1-mobj.flow['sieved']) + mobj.flow['bof']) \
                for mobj in list(self.material_dict.values()) if mobj.ptype != BFProductType.SinterFuel]) \
                - self.slag_prod(w) * self.slag_target['Fe'] \
                - self.dust_target['usage'] * self.target_prod * self.dust_target['Fe']

    # PenPICompB(st)
    def hm_penalty_cost(self, w):
        if self.chem_penalty:
            return (self.hm_target['S'] - self.hm_ref['S']) * self.hm_penalty['S'] * self.target_prod \
                +(self.hot_metal_Mn(w) - self.hm_ref['Mn'] * self.target_prod) * self.hm_penalty['Mn'] \
                +(self.hot_metal_P(w) - self.hm_ref['P'] * self.target_prod) * self.hm_penalty['P'] \
                +(self.hot_metal_Alkali(w) - self.hm_ref['Alkali'] * self.target_prod) * self.hm_penalty['Alkali']
        else:
            return 0.0

    # PARSLB(st, PRODKOST)
    def slag_cost(self, w):
        return self.slag_prod(w) * self.slag['price']

    # PARSLB(st, VERBR)
    def slag_prod(self, w):
        return (sum([ w[input] * ( 1-self.material_dict[input].contents['H2O']) * \
                     (self.material_dict[input].contents['CaO'] \
                      + self.material_dict[input].contents['SiO2'] \
                      + self.material_dict[input].contents['Al2O3'] \
                      + self.material_dict[input].contents['MgO']) \
                     for input in self.material_dict]) \
               - self.dust_alkali() * self.dust_target['usage'] * self.target_prod \
               - 2.14 * self.hm_target['Si'] * self.target_prod)/self.slag['fraction']

    def slag_chemical(self, w, attr):
        if attr == 'Mn':
            ratio = 1.0 - self.slag['Mn_yield']
        else:
            ratio = 1.0
        if attr == 'SiO2':
            adder = -2.14 * self.target_prod * self.hm_target['Si']
        else:
            adder = 0.0
        return ratio * sum([ w[input] * ( 1-self.material_dict[input].contents['H2O']) \
                            * self.material_dict[input].contents[attr] for input in self.material_dict]) \
                    - self.dust_target[attr] * self.dust_target['usage'] * self.target_prod \
                    + adder

    def concentrate_vol(self, w):
        return sum([w[mobj.name] * mobj.flow['fbed'] for mobj in list(self.material_dict.values()) if \
                    mobj.ptype == BFProductType.Concentrate])

    def sinterfeed_vol(self, w):
        return sum([w[mobj.name] * mobj.flow['fbed'] for mobj in list(self.material_dict.values()) if \
                    mobj.ptype == BFProductType.SinterFeed])

    def sinter_coke_vol(self, w):
        return sum([w[input] for input in self.material_dict if self.material_dict[input].ptype == BFProductType.SinterFuel])

    # # USAGE_KOOKS(st, BFCoke, TOTVERBR)
    def bf_coke_usage(self, w):
        if self.coke_penalty:
            ref_coke = sum([mobj.coke_coeff * self.ref_usage[mobj.name]/self.ref_prod \
                            for mobj in self.ore_list])
            cur_coke = sum([mobj.coke_coeff * w[mobj.name] for mobj in self.ore_list])
            return (self.coke_coeff['ref_rate'] - (ref_coke + self.coke_coeff['slag'] * self.coke_coeff['ref_slag']) \
                        + self.coke_coeff['PCI'] * self.coke_coeff['ref_PCI']) * self.target_prod \
                        + cur_coke + self.coke_coeff['slag'] * self.slag_prod(w) \
                        - w['PCI'] * self.coke_coeff['PCI']
        else:
            return self.coke_coeff['ref_rate'] * self.target_prod

    # PARB(st, BKCoke)
    def bf_coke_vol(self, w):
        return sum([w[input] for input in self.material_dict if self.material_dict[input].ptype == BFProductType.BFCoke])

    def total_cost(self, w):
        return sum([w[input] * self.material_dict[input].price for input in self.material_dict]) \
               + self.sinter_cost(w) + self.hm_penalty_cost(w) - self.slag_cost(w)

def plant_optimizer(plant_config, material_list, material_store = None):
    RDI_constraint = True
    if material_store:
        material_dict = material_store
    else:
        material_dict = {}
        for input in material_list:
            material_dict[input] = BOFMaterial()
    plant_config['input_list'] = material_list
    plant = BOFPlant(plant_config)
    plant.load_materials(material_dict)

    # The Goalfunction =  production cost of the ores
    #      + sintering production cost
    #      + production cost coarse cokes
    #      -   profit slag
    #	 + pig iron composition penalty
    #	 + Solid Fuel usage penalty
    # ---------------------------------------------------------------
    prob = pulp.LpProblem("PlantOptimizer", pulp.LpMinimize)
    plant.setup_varirables()
    prob += plant.total_cost(plant.w)

    # material usage constraints
    for input in plant.input_list:
        prob += plant.w[input] - plant.material_dict[input].usage['lower'] >= 0
        prob += plant.w[input] - plant.ind[input] * plant.material_dict[input].usage['upper'] <= 0
    # num of pellet/lump/sinter constraints
    prob += sum([plant.ind[input] for input in plant.input_list \
                 if plant.material_dict[input].ptype == BFProductType.Pellet]) - plant.pellet['max_num'] <= 0
    prob += sum([plant.ind[input] for input in plant.input_list \
                 if plant.material_dict[input].ptype == BFProductType.Lump]) - plant.lump['max_num'] <= 0
    prob += sum([plant.ind[input] for input in plant.input_list \
                 if plant.material_dict[input].ptype in \
                 [BFProductType.SinterFeed, BFProductType.Concentrate]]) - plant.sinter['max_num'] <= 0

    # burnt_lime, PCI, Coke constraints
    prob += plant.w['Burnt_lime'] - plant.burnt_lime_ratio * plant.sinter_prod(plant.w)/( 1- plant.sinter['sieved']) == 0
    prob += plant.w['PCI'] - plant.coke_coeff['PCI_rate'] * plant.target_prod == 0
    prob += plant.sinter_coke_vol(plant.w) - plant.sinter['cms'] * plant.sinter_prod(plant.w) == 0
    prob += plant.bf_coke_vol(plant.w) - plant.bf_coke_usage(plant.w) == 0

    #sinter prod constraints
    prob += plant.sinter_prod(plant.w) - plant.sinter['up_bound'] * plant.target_prod <= 0
    prob += plant.sinter_prod(plant.w) - plant.sinter['low_bound'] * plant.target_prod >= 0
    for attr in ['Fe', 'MgO', 'SiO2', 'Al2O3']:
        prob += plant.sinter_chem_vol(plant.w, attr) - plant.sinter_prod(plant.w) * plant.sinter_lowbound[attr] >= 0
        prob += plant.sinter_chem_vol(plant.w, attr) - plant.sinter_prod(plant.w) * plant.sinter_upbound[attr] <= 0
    prob += plant.sinter_chem_vol(plant.w, 'CaO') \
            - plant.sinter_chem_vol(plant.w, 'SiO2') * plant.sinter_lowbound['Bas'] >= 0
    prob += plant.sinter_chem_vol(plant.w, 'CaO') \
            - plant.sinter_chem_vol(plant.w, 'SiO2') * plant.sinter_upbound['Bas'] <= 0

    if RDI_constraint:
        prob += plant.RDIB(plant.w) - (plant.RDI['max'] - RDI_global['coeff']) \
                                      * plant.sinter_prod(plant.w) <= 0
        prob += plant.RDIB(plant.w) - (plant.RDI['min'] - RDI_global['coeff']) \
                                      * plant.sinter_prod(plant.w) >= 0
    prob += ( 1 - plant.sinter['conc_max_rate']) * plant.concentrate_vol(plant.w) \
                    - plant.sinter['conc_max_rate'] * plant.sinterfeed_vol(plant.w) <= 0

    # constraint for BF input
    for attr in ['Zn', 'Pb', 'K2O', 'Na2O']:
        prob += plant.bf_chem_content(plant.w, attr) - plant.bf_dry[attr] * plant.target_prod <= 0
    prob += plant.sum_pellet(plant.w) - plant.pellet['up_bound'] * plant.target_prod <= 0
    prob += plant.sum_pellet(plant.w) - plant.pellet['low_bound'] * plant.target_prod >= 0
    prob += plant.sum_lump(plant.w) - plant.lump['up_bound'] * plant.target_prod <= 0
    prob += plant.sum_lump(plant.w) - plant.lump['low_bound'] * plant.target_prod >= 0

    # constraint for hot metal
    prob += plant.hot_metal_P(plant.w) - plant.hm_target['P'] * plant.target_prod <= 0
    prob += plant.hot_metal_Mn(plant.w) - plant.hm_target['Mn'] * plant.target_prod <= 0

    prob += (1 - plant.hm_target['C'] - plant.hm_target['Si'] - plant.hm_target['S']) * plant.target_prod \
            + plant.dust_target['P'] * plant.dust_target['usage'] * plant.target_prod \
            + plant.slag_target['P'] * plant.slag_prod(plant.w) - plant.hot_metal_Fe(plant.w) \
            - sum([ plant.w[mobj.name] * mobj.contents['H2O'] * (mobj.contents['P'] \
            + mobj.contents['Mn'] * plant.slag['Mn_yield']) for mobj in list(plant.material_dict.values())]) == 0

    for attr in ['Al2O3', 'MgO']:
        prob += plant.slag_chemical(plant.w, attr) - plant.slag_lowbound[attr] * plant.slag_prod(plant.w) >= 0
        prob += plant.slag_chemical(plant.w, attr) - plant.slag_upbound[attr] * plant.slag_prod(plant.w) <= 0

    prob += plant.slag_chemical(plant.w, 'CaO') - plant.slag_lowbound['Bas'] * plant.slag_chemical(plant.w, 'SiO2') >= 0
    prob += plant.slag_chemical(plant.w, 'CaO') - plant.slag_upbound['Bas'] * plant.slag_chemical(plant.w, 'SiO2') <= 0
    prob += plant.slag_chemical(plant.w, 'CaO') + plant.slag_chemical(plant.w, 'MgO') \
            - plant.slag_lowbound['Tbas'] * (plant.slag_chemical(plant.w, 'SiO2') + plant.slag_chemical(plant.w, 'Al2O3')) >= 0
    prob += plant.slag_chemical(plant.w, 'CaO') + plant.slag_chemical(plant.w, 'MgO') \
            - plant.slag_upbound['Tbas'] * (plant.slag_chemical(plant.w, 'SiO2') + plant.slag_chemical(plant.w, 'Al2O3')) <= 0

    prob += plant.slag_chemical(plant.w, 'CaO') \
            + plant.slag_chemical(plant.w, 'Al2O3') * 0.56 \
            + plant.slag_chemical(plant.w, 'MgO') * 1.4 - plant.slag_chemical(plant.w, 'SiO2') * plant.slag['cement_idx'] >= 0
    prob += plant.slag_prod(plant.w) - plant.slag_lowbound['usage'] * plant.target_prod >= 0
    prob += plant.slag_prod(plant.w) - plant.slag_upbound['usage'] * plant.target_prod <= 0
    prob.solve()
    #cwd = os.getcwd()
    #solverdir = 'cbc-2.7.1\\bin\\cbc.exe'  # extracted and renamed the binary zip.
    #solverdir = os.path.join(cwd, solverdir)
    #solver = pulp.COIN_CMD(path=solverdir)  # I am importing pulp using from pulp import *
    #prob.solve(solver)
    w_opt = {}
    ind_opt = {}
    for v in prob.variables():
        if v.name[:2] == 'w_':
            w_opt[v.name[2:]] = v.varValue
        else:
            ind_opt[v.name[4:]] = v.varValue
    solution = {'w_opt': w_opt, 'ind_opt': ind_opt, 'status': prob.status, \
                'prob': prob, 'plant': plant, }
    return solution

SEABORNE_PRICE_MAP = {
    'BF_coke': dict([('coke_ts', 1.0)]),
    'BSA_Pellet': dict([('plt_io62', 1.0), ('plt_pelletprem_cn', 1.0)]),
    'BHP_JBF': dict([('jbF_61_sb', 1.0)]),
    'BHP_MACF': dict([('macF_61_sb', 1.0)]),
    'BHP_YDF': dict([('ydF_58_sb', 1.0)]),
    'BHP_NMF': dict([('nmF_63_sb', 1.0)]),
    'BHP_NML': dict([('nmF_63_sb', 1.0), ('plt_lp', 63.1)]),
    'FMG_KF': dict([('fbF_59_sb', 1.0)]),
    'FMG_FBF': dict([('fbF_59_sb', 1.0)]),
    'FMG_SSF': dict([('ssF_56_sb', 1.0)]),
    'FMG_WF': dict([('ssF_56_sb', 1.0)]),
    'RH_Fines': dict([('rhF_61_rz', 1.0)]),
    'RH_Lump': dict([('rhL_62_rz', 1.0)]),
    'Rio_PBF': dict([('pbF_62_sb', 1.0)]),
    'Rio_PBL': dict([('pbL_63_sb', 1.0)]),
    'Rio_RVF': dict([('rrF_57_sb', 1.0)]),
    'Rio_RVL': dict([('rrF_57_sb', 1.0), ('plt_lp', 57.3)]),
    'Rio_YDF': dict([('ydF_58_sb', 1.0), ('prem', 1.0)]),
    'CSN_IOC6': dict([('brbF_63_sb', 1.0), ('prem', 0.0)]),
    'CSN_IOCP': dict([('brbF_63_sb', 1.0), ('prem', 2.0)]),
    'Vale_BRBF': dict([('brbF_63_sb', 1.0)]),
    'Vale_BRBF_MYR': dict([('brbF_63_sb', 1.0), ('prem', 1.0)]),
    'Vale_IOCJ': dict([('iocjF_65_sb', 1.0)]),
    'Vale_SFLA': dict([('plt_io62', 1.0), ('prem', 13.0)]),
    'Vale_SSFG': dict([('ssfgF_62_qd', 1.0)]),
    'Vale_SFHG': dict([('sfhtF_60_qd', 1.0)]),
    'Snim_TZFC': dict([('plt_io62', 1.0), ('prem', 3.0)]),
    'Tacora': dict([('plt_io65', 1.0), ('prem', 0.0)]),
    'SINTER_FUEL': dict([('coke_ts', 0.7)]),
    'PCI': dict([('coke_ts', 0.7)]),
    'Bauxite': dict([('prem', 40.0)]),
    'Burnt_lime': dict([('prem', 60.0)]),
    'dolomite': dict([('prem', 40.0)]),
    'dolomite_lump': dict([('prem', 50.0)]),
    'dunite': dict([('prem', 40.0)]),
    'dunite_lump': dict([('prem', 50.0)]),
    'gravel_silex': dict([('prem', 40.0)]),
    'limestone': dict([('prem', 40.0)]),
    'Limestone_BF': dict([('prem', 40.0)]),
    'olivine': dict([('prem', 40.0)]),
    'total_recycling_materials': dict([('prem', 40.0)]),
    }

PORT_PRICE_MAP = { \
    'BF_coke': dict([('coke_ts', 1.0)]),
    'BSA_Pellet': dict([('plt_io62', 1.0), ('plt_pelletprem_cn', 1.0)]),
    'BHP_JBF': dict([('jbF_61_qd', 1.0)]),
    'BHP_MACF': dict([('macF_61_qd', 1.0)]),
    'BHP_YDF': dict([('ydF_58_qd', 1.0)]),
    'BHP_NMF': dict([('nmF_63_qd', 1.0)]),
    'BHP_NML': dict([('nmL_64_qd', 1.0)]),
    'FMG_KF': dict([('mixF_59_qd', 1.0)]),
    'FMG_FBF': dict([('mixF_59_qd', 1.0)]),
    'FMG_SSF': dict([('ssF_56_qd', 1.0)]),
    'FMG_WF': dict([('ssF_56_qd', 1.0)]),
    'RH_Fines': dict([('rhF_61_rz', 1.0)]),
    'RH_Lump': dict([('rhL_62_rz', 1.0)]),
    'Rio_PBF': dict([('pbF_62_qd', 1.0)]),
    'Rio_PBL': dict([('pbL_63_qd', 1.0)]),
    'Rio_RVF': dict([('rrF_57_rz', 1.0)]),
    'Rio_RVL': dict([('rrL_58_rz', 1.0)]),
    'Rio_YDF': dict([('ydF_58_qd', 1.0), ('prem', 1.0)]),
    'CSN_IOC6': dict([('brbF_63_qd', 1.0), ('prem', 0.0)]),
    'CSN_IOCP': dict([('brbF_63_qd', 1.0), ('prem', 2.0)]),
    'Vale_BRBF': dict([('brbF_63_qd', 1.0)]),
    'Vale_BRBF_MYR': dict([('brbF_63_qd', 1.0), ('prem', 1.0)]),
    'Vale_IOCJ': dict([('iocjF_65_qd', 1.0)]),
    'Vale_SFLA': dict([('pbF_62_qd', 1.0), ('prem', 13.0)]),
    'Vale_SSFG': dict([('ssfgF_62_qd', 1.0)]),
    'Vale_SFHG': dict([('sfhtF_60_qd', 1.0)]),
    'Snim_TZFC': dict([('plt_io62', 1.0), ('prem', 3.0)]),
    'Tacora': dict([('plt_io65', 1.0), ('prem', 0.0)]),
    'SINTER_FUEL': dict([('coke_ts', 0.7)]),
    'PCI': dict([('coke_ts', 0.7)]),
    'Bauxite': dict([('prem', 40.0)]),
    'Burnt_lime': dict([('prem', 60.0)]),
    'dolomite': dict([('prem', 40.0)]),
    'dolomite_lump': dict([('prem', 50.0)]),
    'dunite': dict([('prem', 40.0)]),
    'dunite_lump': dict([('prem', 50.0)]),
    'gravel_silex': dict([('prem', 40.0)]),
    'limestone': dict([('prem', 40.0)]),
    'Limestone_BF': dict([('prem', 40.0)]),
    'olivine': dict([('prem', 40.0)]),
    'total_recycling_materials': dict([('prem', 40.0)]),
    }

def get_material_prices(price_map, start, end):
    curve_set = set()
    for input in price_map:
        curve_set = curve_set.union(set([elem for elem in list(price_map[input].keys())]))
    data_list = []
    fx_conv = []
    for spot_id in curve_set:
        if spot_id in ['prem']:
            continue
        data = ts_tool.get_data(spot_id, start, end, name = spot_id)
        if ('_qd' in spot_id) or ('_cfd' in spot_id) or ('_rz' in spot_id) or ('_jt' in spot_id) \
                or ('_jy' in spot_id) or ('_ly' in spot_id) or ('_ls' in spot_id) or ('_tj' in spot_id) \
                or ('_tc' in spot_id):
            data = data - 30.0
            fx_conv.append(spot_id)
            if 'pbF' in spot_id:
                data = ts_tool.apply_vat(data)/0.915
            elif 'pbL' in spot_id:
                data = ts_tool.apply_vat(data)/0.96
            elif 'nmF' in spot_id:
                data = ts_tool.apply_vat(data)/0.93
            elif 'nmL' in spot_id:
                data = ts_tool.apply_vat(data)/0.96
            elif 'jbF' in spot_id:
                data = ts_tool.apply_vat(data)/0.929
            elif 'macF' in spot_id:
                data = ts_tool.apply_vat(data)/0.922
            elif 'macL' in spot_id:
                data = ts_tool.apply_vat(data)/0.96
            elif 'ydF' in spot_id:
                data = ts_tool.apply_vat(data)/0.905
            elif 'iocjF' in spot_id:
                data = ts_tool.apply_vat(data)/0.915
            elif 'rrF' in spot_id:
                data = ts_tool.apply_vat(data)/0.91
            elif 'rrL' in spot_id:
                data = ts_tool.apply_vat(data)/0.96
            elif 'brbF' in spot_id:
                data = ts_tool.apply_vat(data)/0.91
            elif 'sfhtF' in spot_id:
                data = ts_tool.apply_vat(data)/0.91
            elif 'ssfgF' in spot_id:
                data = ts_tool.apply_vat(data)/0.92
            elif 'fbF' in spot_id:
                data = ts_tool.apply_vat(data)/0.925
            elif 'ssF' in spot_id:
                data = ts_tool.apply_vat(data)/0.91
            else:
                data = ts_tool.apply_vat(data)/0.91
        elif ('_ts' in spot_id):
            fx_conv.append(spot_id)
        data_list.append(data)
    fx = ts_tool.get_data('USD/CNY', start, end, spot_table = 'fx_daily', name = 'fx', field = 'ccy')
    data_list.append(fx)
    merged = ts_tool.merge_df(data_list).fillna(method = 'ffill').fillna(method = 'bfill')
    merged['prem'] = 1.0
    for spotID in fx_conv:
        merged[spotID] = merged[spotID]/merged['fx']
    for input in price_map:
        if input not in merged.columns:
            merged[input] = 0.0
        for elem in list(price_map[input].keys()):
            merged[input] = merged[input] + price_map[input][elem] * merged[elem]
    return merged, list(curve_set)








