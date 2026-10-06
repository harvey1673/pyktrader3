"""Research-only shortlist of spot inputs for China futures physical carry.

This module deliberately describes a small representative spot basket.  It is
not a production price map and it does not imply that a vendor label is an
exchange-deliverable quote without the adjustments recorded below.
"""

from __future__ import annotations


SPOT_CATALOG = {
    # Ferrous and industrial materials
    "rb": dict(preferred="上海 HRB400E 20mm 螺纹钢市场价（含税）", provider="mysteel", fallback="杭州同规格市场价", local_alias="rebar_sh", adjustment="统一理计/过磅、品牌、仓库和含税口径"),
    "hc": dict(preferred="上海 Q235B 4.75mm 热轧卷板市场价（含税）", provider="mysteel", fallback="天津同规格市场价", local_alias="hrc_sh", adjustment="统一厚度、宽度、品牌、仓库和含税口径"),
    "i": dict(preferred="青岛港 PB 粉现货成交/最低可成交价（湿吨、含税）", provider="mysteel", fallback="青岛港 BRBF/IOC6/超特粉小篮子", local_alias="io_ctd_spot", adjustment="按合约期执行品牌、Fe、SiO2、Al2O3、P、S、粒度和湿吨折干吨调整"),
    "j": dict(preferred="日照港准一级湿熄冶金焦现汇含税出库价", provider="mysteel", fallback="吕梁准一级干熄冶金焦出厂现汇含税价", local_alias="coke_sub_a_rz_outstock", adjustment="不要与平仓/FOB价取最小值；统一水分后按合约期质量升贴水并计入到库运费"),
    "jm": dict(preferred="唐山自提蒙5精煤 ID01892939", provider="mysteel", fallback="介休中硫主焦煤；2016回溯可用孝义低硫主焦煤", local_alias="ckc_mongol5_ts", adjustment="沙河驿序列2024后停更，不作主腿；按灰、硫、G、CSR、镜质体反射率及交割地运费调整"),
    "SM": dict(preferred="天津港/天津市场 FeMn65Si17 锰硅现货价（含税）", provider="mysteel", fallback="乌兰察布主产区出厂价", local_alias="SM_65s17_tj", adjustment="核对粒度、锰硅含量、现金/承兑和交割地运费"),
    "SF": dict(preferred="天津市场 72# 合格块硅铁现货价（含税）", provider="mysteel", fallback="宁夏中卫72#出厂价", local_alias="SF_72_tj", adjustment="天津价优先；核对粒度、Al/Ca等杂质、现金/承兑及仓库运费"),
    "SA": dict(preferred="沙河重质纯碱市场/送到价（含税）", provider="mysteel", fallback="山东重质纯碱出厂价", local_alias="SA_heavy_shahe", adjustment="仅用重碱；统一厂提/送到、包装、品牌、仓库和合约期质量标准"),
    "FG": dict(preferred="沙河 5mm 大板浮法玻璃市场/出厂价（含税）", provider="mysteel", fallback="湖北5mm大板浮法玻璃价", local_alias="FG_5mm_shahe", adjustment="统一规格、计重方式、厂库升贴水和运输费用"),
    "v": dict(preferred="华东电石法 SG-5 PVC 市场价（含税）", provider="mysteel", fallback="华北电石法SG-5市场价", local_alias="pvc_cac2_sh", adjustment="只选可交割牌号/厂家；统一自提地点、含税和品牌升贴水"),
    "SH": dict(preferred="山东32%液碱出厂价÷0.32", provider="mysteel", fallback="山东50%液碱出厂价÷0.50", local_alias="SH_ctd_spot", adjustment="折100%干基后执行合约期浓度、品质、地区和仓储升贴水"),

    # Metals and precious metals
    "cu": dict(preferred="上海 SMM 1# 电解铜现货均价（含税）", provider="ifind", fallback="长江有色1#铜现货均价", local_alias="cu_smm1_spot", adjustment="如做严格CTD再叠加可交割品牌升贴水与仓库费用"),
    "al": dict(preferred="上海 SMM A00 铝现货均价（含税）", provider="ifind", fallback="长江有色A00铝现货均价", local_alias="al_smm0_spot", adjustment="统一地区、品牌、仓库和含税口径"),
    "zn": dict(preferred="上海 SMM 0# 锌现货均价（含税）", provider="ifind", fallback="上海1#锌现货均价", local_alias="zn_smm0_spot", adjustment="0#为核心；按品牌和交割仓库调整"),
    "ni": dict(preferred="上海 SMM 1# 电解镍现货均价（含税）", provider="ifind", fallback="金川镍/俄镍可交割品牌报价", local_alias="ni_smm1_spot", adjustment="品牌溢价在压力期会主导CTD，保留品牌字段"),
    "pb": dict(preferred="上海 SMM 1# 铅现货均价（含税）", provider="ifind", fallback="华东99.994%铅市场价", local_alias="pb_994_shmet_east", adjustment="统一原生/再生、品牌、仓库和含税口径"),
    "sn": dict(preferred="上海 SMM 1# 锡现货均价（含税）", provider="ifind", fallback="长江有色1#锡现货均价", local_alias="sn_smm1_spot", adjustment="核对可交割品牌和仓库费用"),
    "ss": dict(preferred="无锡宏旺 304/2B 2*1240*C 冷轧不锈钢卷市场价", provider="mysteel", fallback="无锡304/2B同规格可交割品牌最低价", local_alias="ss_304_2b_hongwang_wuxi", adjustment="1240mm报价先作为市场对照；转换到可交割宽度后再按合约期毛/切边升贴水"),
    "ao": dict(preferred="山东 AO-1/可交割氧化铝现货均价（含税）", provider="mysteel", fallback="山西或河南同品级现货价", local_alias="alumina_spot_qd", adjustment="仅纳入注册品牌；按交割地、包装和合约期地区升贴水调整"),
    "au": dict(preferred="上金所 Au99.99 现货收盘/加权均价", provider="ifind", fallback="上金所Au99.95现货价", local_alias="au_9999_sge_close", adjustment="统一元/克与期货元/克；核对交割成色和金锭规格"),
    "ag": dict(preferred="上海 SMM 1# 银现货均价", provider="ifind", fallback="上金所Ag99.99实物现货价", local_alias=None, adjustment="避免用Ag(T+D)充当纯物理现货；统一元/千克、品牌和仓库"),

    # New-energy materials
    "si": dict(preferred="四川553不通氧工业硅现货价", provider="mysteel", fallback="四川421工业硅现货价", local_alias="si_ctd_spot", adjustment="按合约期牌号升贴水；553与421分别调整后再取最低可交割成本"),
    "lc": dict(preferred="国产电池级碳酸锂现货均价", provider="ifind", fallback="国产工业级碳酸锂现货均价", local_alias="lc_ctd_spot", adjustment="工业级按合约期升贴水转换；统一含税、包装、品牌和交割地"),
    "ps": dict(preferred="N型致密料/混合块料现货均价", provider="ifind", fallback="N型颗粒硅现货均价", local_alias=None, adjustment="多晶硅交割标准仍在快速演变；必须按具体合约核对基准品切换与+2000等升贴水后再取CTD"),

    # Rubber, pulp and fertilizer
    "ru": dict(preferred="上海/江苏可交割全乳胶 SCR WF 现货价", provider="mysteel", fallback="昆明全乳胶SCR WF现货价", local_alias="ru_scrwf_kunming", adjustment="优先华东以减少地区调整；昆明价需加云南交割地贴水/运费并检查生产年份"),
    "UR": dict(preferred="山东小颗粒尿素市场/出厂价（含税）", provider="mysteel", fallback="河南小颗粒尿素市场价", local_alias="UR_shandong_spot", adjustment="统一粒度、氮含量、出厂/送到、包装及厂库升贴水"),
    "sp": dict(preferred="山东市场进口针叶浆银星牌现货价", provider="mysteel", fallback="江苏/上海银星牌现货价", local_alias=None, adjustment="保留品牌和产地；仅使用交易所认可品牌并计入仓库/地区费用"),
    "nr": dict(preferred="青岛保税区 STR20/SIR20 美元现货价", provider="mysteel", fallback="青岛非保税20号胶人民币现货价", local_alias=None, adjustment="美元价需按当日汇率、关税/增值税、融资和保税仓库费用转换；按注册产地品牌筛选"),
    "br": dict(preferred="山东市场 BR9000 可交割品牌现货价", provider="mysteel", fallback="华东BR9000可交割品牌现货价", local_alias=None, adjustment="只纳入交易所认证品牌/牌号，统一含税和仓库费用"),

    # Petrochemicals and energy
    "l": dict(preferred="天津/华北 LLDPE 7042 可交割厂家市场价", provider="mysteel", fallback="华东LLDPE 7042市场价", local_alias="l_7042_tj", adjustment="保留厂家和牌号，统一含税、自提地及仓库费用"),
    "pp": dict(preferred="华东 PP 拉丝 T30S 可交割厂家市场价", provider="ifind", fallback="大庆炼化T30S出厂/华东销售价", local_alias="pp_t30s_shaoxing_hz", adjustment="用牌号/厂家报价替代泛化价格指数；统一含税和自提地"),
    "TA": dict(preferred="华东 PTA 现货自提价", provider="mysteel", fallback="主港可交割品牌仓单/现货价", local_alias="TA_east_spot", adjustment="统一现款/承兑、品牌、仓库和含税口径"),
    "PX": dict(preferred="华东国产 PX 现货出库价", provider="mysteel", fallback="CFR中国/台湾PX美元现货价", local_alias=None, adjustment="进口价按汇率、关税、增值税、港杂及融资转成人民币完税到岸价"),
    "eg": dict(preferred="华东主港 MEG 罐区现货价", provider="mysteel", fallback="张家港MEG现货价", local_alias="eg_east_spot", adjustment="统一现款/承兑、罐区、含税及可交割品质"),
    "MA": dict(preferred="太仓/江苏甲醇现货出罐价", provider="mysteel", fallback="鲁南甲醇出厂价", local_alias="MA_spot_jiangsu", adjustment="港口价优先；内地价需叠加运费并统一含税和现款口径"),
    "eb": dict(preferred="华东主港苯乙烯现货出罐价", provider="mysteel", fallback="江苏苯乙烯现货价", local_alias="eb_east_spot", adjustment="统一交货月份、罐区、含税和可交割品质"),
    "sc": dict(preferred="可交割中质含硫原油到岸成本篮子（Oman/Dubai等）", provider="ifind", fallback="INE仓单/保税现货成交价", local_alias=None, adjustment="不能直接拿Brent；按船期、升贴水、汇率、桶吨转换、运费、保税仓储及可交割油种筛选"),
    "lu": dict(preferred="新加坡0.5%低硫燃料油货物价", provider="ifind", fallback="舟山保税低硫燃料油供船/仓单价", local_alias=None, adjustment="货物价优先于船供零售价；按汇率、运费、贴水、吨桶转换和保税仓储转换"),
    "bu": dict(preferred="山东70#重交道路沥青可交割品牌出厂价", provider="mysteel", fallback="华东70#重交道路沥青市场价", local_alias="bu_heavy_shandong", adjustment="仅纳入注册品牌；统一牌号、含税、厂提/仓库和地区费用"),
    "fu": dict(preferred="舟山/新加坡380cst高硫燃料油货物价", provider="ifind", fallback="新加坡FOB 380cst", local_alias="fo_380cst_zhoushan", adjustment="按380cst、硫含量、汇率、运费和保税仓储转换"),
    "pg": dict(preferred="华南进口丙烷CFR人民币含税成本代理", provider="ifind", fallback="华东/台湾进口丙烷CFR", local_alias="propane_cfr_south_cny_vat", adjustment="已含汇率和13%增值税；仍需丙烷/丁烷组分、关税、运费和港杂调整"),

    # Oils, meals and grains
    "m": dict(preferred="日照/华东43%蛋白豆粕现货价", provider="ifind", fallback="东莞43%蛋白豆粕现货价", local_alias=None, adjustment="保留油厂/品牌和基差月份；统一含税、现款及仓库费用"),
    "RM": dict(preferred="东莞/广西36%蛋白菜粕现货价", provider="ifind", fallback="南通36%蛋白菜粕现货价", local_alias=None, adjustment="统一蛋白、水分、含税和交割地；注意进口菜籽压榨区域差异"),
    "y": dict(preferred="张家港一级豆油现货价", provider="ifind", fallback="天津一级豆油现货价", local_alias=None, adjustment="统一一级品质、散装、含税和交割库费用"),
    "p": dict(preferred="广州24度棕榈油现货价", provider="ifind", fallback="张家港/天津24度棕榈油现货价", local_alias=None, adjustment="仅用24度分提棕榈油；统一含税、港口和交割库费用"),
    "OI": dict(preferred="华东三级菜籽油现货价", provider="ifind", fallback="广西三级菜籽油现货价", local_alias=None, adjustment="核对合约期基准品、酸价和进口/国产来源；统一含税及仓库费用"),
    "a": dict(preferred="黑龙江国产非转基因三等大豆净粮价", provider="ifind", fallback="北安/哈尔滨同等级大豆收购价", local_alias=None, adjustment="统一蛋白、水分、完整粒、杂质、净粮/毛粮及交割库运费"),
    "b": dict(preferred="日照/青岛港进口转基因大豆分销价", provider="ifind", fallback="南通港进口大豆分销价", local_alias=None, adjustment="按黄大豆2号交割质量、含税、港杂、检疫及仓库费用调整"),
    "c": dict(preferred="锦州港二等玉米平舱/收购价", provider="ifind", fallback="鲅鱼圈港二等玉米价", local_alias=None, adjustment="统一14.5%水分、容重、霉变、毒素、平舱/库内及年度规则"),
    "cs": dict(preferred="山东一级玉米淀粉出厂价", provider="ifind", fallback="河北一级玉米淀粉出厂价", local_alias=None, adjustment="统一水分、酸度、蛋白、斑点、包装、含税和交割地运费"),

    # Soft commodities and livestock
    "CJ": dict(preferred="新疆产一级灰枣现货价（沧州销区或交割库）", provider="ifind", fallback="阿克苏/若羌一级灰枣产地价", local_alias=None, adjustment="按个数/公斤、含水率、总糖、不完善果和交割年度质量升贴水转换"),
    "CF": dict(preferred="中国棉花价格指数 CC Index 3128B", provider="ifind", fallback="新疆库3128B机采棉现货价", local_alias=None, adjustment="按颜色级、长度、马克隆值、断裂比强度、轧工质量及仓库升贴水转换"),
    "jd": dict(preferred="山东德州大码鲜鸡蛋产区价", provider="ifind", fallback="全国主产区鸡蛋均价", local_alias=None, adjustment="由元/斤转换为元/500kg；质量、包装、车板与交割库成本使其只能作为可解释代理"),
    "AP": dict(preferred="山东栖霞80#一二级纸袋富士库存果现货价", provider="ifind", fallback="陕西洛川同等级富士价", local_alias=None, adjustment="按果径、着色、硬度、糖度、果锈、冷库和交割年度标准转换"),
    "lh": dict(preferred="河南标准体重外三元生猪出栏价", provider="ifind", fallback="全国外三元生猪均价", local_alias=None, adjustment="统一110kg左右体重、瘦肉率、出栏/到场和区域；季节性体重价差需显式处理"),
    "SR": dict(preferred="南宁一级白砂糖现货价", provider="ifind", fallback="柳州一级白砂糖现货价", local_alias=None, adjustment="统一一级品、含税、仓库、包装和产销区运费"),
    "PK": dict(preferred="河南油料花生仁现货价", provider="ifind", fallback="山东油料花生仁现货价", local_alias=None, adjustment="避免食用白沙米报价；按含油率、水分、酸价、杂质、包装和交割年度规则转换"),
}


EXPECTED_CODES = (
    "rb", "hc", "i", "j", "jm", "SM", "SF", "SA", "FG", "v", "SH",
    "cu", "al", "zn", "ni", "pb", "sn", "ss", "ao", "au", "ag",
    "si", "lc", "ps", "ru", "UR", "sp", "nr", "br", "l", "pp",
    "TA", "PX", "eg", "MA", "eb", "sc", "lu", "bu", "fu", "pg",
    "m", "RM", "y", "p", "OI", "a", "b", "c", "cs", "CJ", "CF",
    "jd", "AP", "lh", "SR", "PK",
)


def get_spot_spec(code: str) -> dict:
    """Return a copy of the research spot specification for *code*."""

    try:
        return dict(SPOT_CATALOG[code])
    except KeyError as exc:
        raise KeyError(f"No CTD spot research specification for {code!r}") from exc
