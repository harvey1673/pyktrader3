# -*- coding: utf-8 -*-
from bf_optimizer import *

#folder_loc = "C:\\dev\\data\\plant_data\\"
folder_loc = "C:\\Users\H464717\Documents\AMIOS - Emperical approach July 2010\\"

def load_lingo_list(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
    data_list = []
    for line in lines:
        product = line.strip().replace(',', '')
        data_list.append(product)
    return data_list

def load_lingo_data(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
    data = {}
    pri_key = None
    sec_key = None
    for line in lines:
        if line[0] == '~':
            pri_key = None
            sec_key = None
        elif line[0] == '!':
            new_key = line[1:].replace(';', '').strip()
            if pri_key == None:
                pri_key = new_key
                data[pri_key] = []
            else:
                sec_key = new_key
                if type(data[pri_key]).__name__ != 'dict':
                    data[pri_key] = {}
                data[pri_key][sec_key] = []
        else:
            if ' ' in line.strip():
                new_data = [float(elem) for elem in line.strip().split(' ') if len(elem)>0]
            else:
                new_data = float(line.strip())
            if sec_key != None:
                data[pri_key][sec_key].append(new_data)
            elif pri_key != None:
                data[pri_key].append(new_data)
    return data

def prepare_plant_data():
    invoer = load_lingo_data(folder_loc + "invoer.dat")
    invoer2 = load_lingo_data(folder_loc + "invoer2.dat")
    invoer_ref = load_lingo_data(folder_loc + "invoerRefPoint.dat")
    plant_spec = {}
    plant_spec['name'] = 'dummy'
    plant_spec['input_list'] = load_lingo_list(folder_loc + "aGrst_Nm.dat")
    plant_spec['ref_input'] = dict([(key, invoer_ref['AMOUNTREFI'][key][0]) for key in invoer_ref['AMOUNTREFI']])
    plant_spec['target_production'] = invoer['GLOBAL_PIG_IRON'][0]
    plant_spec['ref_production'] = invoer2['PigIronAmountRefI'][0]
    plant_spec['coke_coeff'] = {'ref_rate': invoer['PCGKRYI'][0], \
                                'ref_slag': invoer2['SlagUseRefI'][0], \
                                'slag': invoer2['KBFCkSlagI'][0], \
                                'ref_PCI': invoer2['PCIRefI'][0], \
                                'PCI': invoer2['PCIEqCoeffI'][0], \
                                'PCI_rate': invoer2['PCIFixedUsage'][0]/1000.0, }
    plant_spec['burnt_lime_ratio'] = invoer2['BURNTLIMEI'][0]/1000.0
    plant_spec['pellet'] = {'max_num': invoer2['NumMaxPellets'][0], \
                            'up_bound': invoer2['MAXPELLET'][0]/1000.0, \
                            'low_bound': invoer2['MINPELLET'][0]/1000.0,}
    plant_spec['lump'] = {'max_num': invoer2['NumMaxLumps'][0], \
                            'up_bound': invoer2['MAXLUMP'][0] / 1000.0, \
                            'low_bound': invoer2['MINLUMP'][0] / 1000.0, }
    plant_spec['sinter'] = {'price': invoer['PARSINI'][2]*1000.0, \
                            'cms': invoer['CMSFKI'][0], \
                            'FeO_target': invoer2['FeOTargetValue'][0],\
                            'yield': invoer2['SINTER_YIELD_CURRENT'][0], \
                            'work_rate': invoer2['WORKING_RATE_CURRENT'][0], \
                            'sieved': invoer2['GROSSSNTSIEVEDI'][0], \
                            'up_bound': invoer['PARSINI'][1], \
                            'low_bound': invoer['PARSINI'][0],\
                            'max_num': invoer2['NumMaxSinterOres'][0], \
                            'conc_max_rate': invoer2['CONCRATEMAXI'][0], }
    plant_spec['sinter_leakage'] = {'S': invoer2['SULPHUR_LEAK'][0], \
                                    'Zn': invoer2['ZINC_LEAK'][0], \
                                    'Pb': invoer2['PB_LEAK'][0], \
                                    'Alkali': invoer2['ALKALI_LEAK'][0], }
    plant_spec['sinter_lowbound'] = dict([(elem, val) for elem, val in zip(CHEM_CONTENTS, invoer['PERCSINOGI'])])
    plant_spec['sinter_lowbound']['Bas'] = invoer['VBBASIOGI'][0]
    plant_spec['sinter_upbound'] = dict([(elem, val) for elem, val in zip(CHEM_CONTENTS, invoer['PERCSINBGI'])])
    plant_spec['sinter_upbound']['Bas'] = invoer['VBBASIBGI'][0]
    plant_spec['bf_dry'] = {'K2O': invoer['VBK2OHLBGI'][0], \
                            'Na2O': invoer['VBNA2OHLBGI'][0], \
                            'Pb': invoer['VBPBHLBGI'][0], \
                            'Zn': invoer['VBZNHLBGI'][0], }
    plant_spec['hot_metal_ref'] = {'P': invoer2['PIGIRONPREFI'][0], \
                                'Mn': invoer2['PIGIRONMNREFI'][0], \
                                'S': invoer2['PIGIRONSREFI'][0], \
                                'Alkali': invoer2['PIGIRONALKALIREFI'][0],}
    plant_spec['hot_metal_target'] = {'P': invoer['PCPRYBGI'][0], \
                                    'Mn': invoer['PCMNRYOGI'][0], \
                                    'S': invoer['PCSRYSOLI'][0], \
                                    'C': invoer['PCCRYSOLI'][0], \
                                    'Si': invoer['PCSIRYSOLI'][0], }
    plant_spec['hot_metal_penalty'] = {'S': invoer2['FACTORSI'][0], \
                                       'P': invoer2['FACTORPI'][0],\
                                       'Mn': invoer2['FACTORMNI'][0], \
                                       'Alkali': invoer2['FACTORALKALII'][0], }
    plant_spec['slag'] = {'fraction': invoer['SLAKFRAKI'][0], \
                        'price': invoer['PARSLI'][2] * 1000.0, \
                        'Mn_yield': invoer['MNRENDI'][0],\
                        'cement_idx': invoer['CEMINDSLOGI'][0],}
    plant_spec['slag_lowbound'] = {'usage': invoer['PARSLI'][0], \
                                   'Al2O3': invoer['PCALOSLOGI'][0], \
                                   'MgO': invoer['PCMGOSLOGI'][0], \
                                   'Bas': invoer['VBBASLOGI'][0], \
                                   'Tbas': invoer['VBTBASLOGI'][0], }
    plant_spec['slag_upbound'] = {'usage': invoer['PARSLI'][1], \
                                   'Al2O3': invoer['PCALOSLBGI'][0], \
                                   'MgO': invoer['PCMGOSLBGI'][0], \
                                   'Bas': invoer['VBBASLBGI'][0], \
                                   'Tbas': invoer['VBTBASLBGI'][0], }
    plant_spec['slag_target'] = {'Fe': invoer['PCFESLSOLI'][0], 'P': invoer['PCPSLSOLI'][0], }
    plant_spec['dust_target'] = dict([(elem, val) for elem, val in zip(CHEM_CONTENTS, invoer['PERCSTI'])])
    plant_spec['dust_target']['usage'] = invoer['RELVBSTI'][0]
    plant_spec['RDI'] = {'min': invoer2['RDIMINI'][0], 'max': invoer2['RDIMAXI'][0]}
    return plant_spec

def prepare_material_data():
    invoer = load_lingo_data(folder_loc + "invoer.dat")
    invoer2 = load_lingo_data(folder_loc + "invoer2.dat")
    invoer_ref = load_lingo_data(folder_loc + "invoerRefPoint.dat")
    material_list = load_lingo_list(folder_loc + "aGrst_Nm.dat")
    coke_list = load_lingo_list(folder_loc + "cokes_Nm.dat")
    additive_list = load_lingo_list(folder_loc + "toesl_Nm.dat")
    ore_list = load_lingo_list(folder_loc + "erts_Nm.dat")
    material_dict = {}
    for input in material_list:
        spec_dict = {}
        spec_dict['name'] = input
        spec_dict['contents'] = dict([(elem, x)for elem, x in zip(CHEM_CONTENTS, invoer['PERCI'][input])])
        if input in ore_list:
            spec_dict['ptype'] = int(invoer2['TYPE'][input][0]) - 1
        elif input in additive_list:
            spec_dict['ptype'] = 6
        elif input in coke_list:
            if input.upper() == 'BF_COKE':
                spec_dict['ptype'] = 5
                spec_dict['flow'] = {'fbed': 0.0, 'cbed': 0.0, 'bof': 1.0, 'sieved': 0.0,}
            elif input.upper() == 'SINTER_FUEL':
                spec_dict['ptype'] = 4
                spec_dict['flow'] = {'fbed': 1.0, 'cbed': 0.0, 'bof': 0.0, 'sieved': 0.0, }
            else:
                print("unknown input for coke %s" % input)
        else:
            print('unknow product type')
        if input in invoer['MATSTRI']:
            spec_dict['flow'] = { 'fbed': invoer['MATSTRI'][input][0],\
                                'cbed': invoer['MATSTRI'][input][1], \
                                'sieved': invoer['MATSTRI'][input][2], \
                                'bof': invoer['MATSTRI'][input][3], }
        spec_dict['usage'] =  {'lower': invoer['PARI'][input][0], \
                                'upper': invoer['PARI'][input][1], }
        spec_dict['price'] = invoer['PARI'][input][2] * 1000
        if input in invoer2['KBFCkLumpI']:
            spec_dict['coke_coeff'] = invoer2['KBFCkLumpI'][input][0]
        else:
            spec_dict['coke_coeff'] = 0.0
        mobj = BOFMaterial(spec_dict)
        material_dict[input] = mobj
    return material_dict

def calc_bf_mix():
    material_dict = prepare_material_data()
    plant_config = prepare_plant_data()
    plant_config['input_list'] = ['BF_coke', 'BSA_Pellet', 'Burnt_lime',\
                                  'dolomite', 'dunite', 'FMG_FBF', 'FMG_SSF', 'gravel_silex',\
                                  'limestone', 'Limestone_BF', 'olivine', 'PCI', 'Rio_PBL', 'Rio_Yandi', \
                                  'SINTER_FUEL', 'total_recycling_materials', 'Vale_SFLA']
    plant_config['coke_penalty'] = False
    plant_config['chem_penalty'] = False
    material_list = list(material_dict.keys())
    solution = plant_optimizer(plant_config, material_list, material_store=material_dict)
    return solution






