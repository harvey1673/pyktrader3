import json
from pycmqlib3.utility.sec_bits import LOCAL_NUTSTORE_FOLDER
from pycmqlib3.utility.misc import inst2product
from misc_scripts.factor_data_update import port_pos_config

hotmap_file = f"{LOCAL_NUTSTORE_FOLDER}/hotmap_prod.json"


def update_roll_cont():
    with open(hotmap_file, 'r') as fp:
        hotmap = json.load(fp)

    cont_map = {}
    for exch in hotmap:
        cont_map.update(hotmap[exch])

    strat_list = []
    for key in port_pos_config:
        pos_loc = port_pos_config[key]['pos_loc']
        strat_by_pos = [f'{pos_loc}/settings/{strname}' for (strname, w, roll) in port_pos_config[key]['strat_list']]
        strat_list += strat_by_pos
    strat_list = list(set(strat_list))
    for strat_file in strat_list:
        with open(strat_file, 'r') as fp:
            strat_conf = json.load(fp)
        strat_args = strat_conf['config']
        assets = strat_args['assets']
        for asset_conf in assets:
            underlying = asset_conf['underliers'][0]
            prod = inst2product(underlying)
            if cont_map[prod] != underlying:
                asset_conf["prev_underliers"] = asset_conf['underliers']
                asset_conf['underliers'] = [cont_map[prod]]
                print(strat_file, asset_conf)

        with open(strat_file, 'w') as f:
            json.dump(strat_conf, f, indent=4)


if __name__ == "__main__":
    update_roll_cont()
