import json
from pycmqlib3.utility.sec_bits import LOCAL_NUTSTORE_FOLDER
from pycmqlib3.utility.misc import inst2product

hotmap_file = f"{LOCAL_NUTSTORE_FOLDER}/hotmap_prod.json"
strat_list = [
    "C:/dev/pyktrader3/process/pt_test3/settings/PT_FACTPORT3.json",
    "C:/dev/pyktrader3/process/pt_test3/settings/PT_FACTPORT_HCRB.json",
    "C:/dev/pyktrader3/process/pt_test3/settings/PT_FACTPORT_LEADLAG1.json",
    "C:/dev/pyktrader3/process/pt_test3/settings/PT_FUNDA_FERROUS.json",
    "C:/dev/pyktrader3/process/pt_test3/settings/PT_FUNDA_BASE.json",

    "C:/dev/pyktrader3/process/pt_test1/settings/PTSIM1_FACTPORT.json",
    "C:/dev/pyktrader3/process/pt_test1/settings/PTSIM1_HRCRB.json",
    "C:/dev/pyktrader3/process/pt_test1/settings/PTSIM1_LL.json",
    "C:/dev/pyktrader3/process/pt_test1/settings/PTSIM1_FUNFER.json",
    "C:/dev/pyktrader3/process/pt_test1/settings/PTSIM1_FUNBASE.json",
]


def update_roll_cont():
    with open(hotmap_file, 'r') as fp:
        hotmap = json.load(fp)

    cont_map = {}
    for exch in hotmap:
        cont_map.update(hotmap[exch])

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
