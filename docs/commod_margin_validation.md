# commod margin formulas: external validation notes

Date: 2026-03-30
Source file reviewed: `docs/commod margin.csv`

## 1) Scope and method
This note validates representative formulas in `commod margin.csv` against
public references and first-principles stoichiometry.

Validation labels:
- `validated`: coefficient/order of magnitude matches public references or
  direct stoichiometric calculation.
- `plausible`: formula is directionally reasonable, but exact coefficient is
  plant/config dependent and cannot be uniquely fixed from public sources.
- `needs-internal-benchmark`: no reliable public fixed coefficient; should be
  calibrated with your internal plant/run-rate data.

Primary references used:
- Crush spread yield convention (meal/oil):
  https://en.wikipedia.org/wiki/Crush_spread
- Soybean meal bushel yield example:
  https://en.wikipedia.org/wiki/Soybean_meal
- Hall-Heroult process and electricity intensity:
  https://en.wikipedia.org/wiki/Hall%E2%80%93H%C3%A9roult_process
- Aluminum smelting overview (energy-intensive process):
  https://en.wikipedia.org/wiki/Aluminium_smelting
- TPA/PTA process background (PX oxidation route):
  https://en.wikipedia.org/wiki/Terephthalic_acid

## 2) Chemical chain (TA/MA/PF/PL/L/PP/V/EG/EB/PX/LU/FU)

### 2.1 TA (PTA)
- CSV formula: `TA = 0.655 * PX + 加工费`
- Result: `validated (engineering approximation)`
- Reason: PTA from PX has theoretical PX mass ratio about 0.639 by molecular
  weight (`106.16 / 166.13`). The practical 0.655 includes process losses and
  side consumption, which is a common industry treatment.

### 2.2 PF (polyester staple)
- CSV formula: `PF = 0.855 * PTA + 0.335 * EG + 加工费`
- Result: `plausible`
- Reason: PET/polyester chain commonly uses PTA + MEG core feedstock and
  coefficients near this range are widely used in trading cost models.

### 2.3 MA (methanol)
- Coal route: `MA = 1.6 * 煤价 + 综合费用`
- Gas route: `MA = 气耗 * 气价 + 加工费`
- Result: `plausible`
- Reason: route split and variable cost structure are correct; exact unit
  consumptions vary by coal quality, gas composition, and plant efficiency.

### 2.4 PL, L, PP, V, EG, EB, BZ, PX, LU, FU
- Result: `plausible` to `needs-internal-benchmark`
- Reason: route logic is correct (naphtha/PDH/MTO/electricity-intensive
  chlorine alkali chains), but fixed coefficients and processing fees are
  strongly plant-, fuel-, and region-dependent.

Recommendation:
- Keep formula structure.
- Move coefficients and processing fees into configurable parameters by route
  and region (for example: CN coastal, CN inland coal, US ethane).

## 3) Black and ferroalloy (j/rb/SM/SF)

### 3.1 Coke j
- CSV formula: `j = 1.35 * 焦煤 + 加工费`
- Result: `validated (common approximation)`
- Reason: coking coal-to-coke mass conversion around this level is a common
  market proxy.

### 3.2 Rebar rb (long/short process)
- Long route and EAF route decomposition is directionally correct.
- Result: `plausible`
- Reason: burden mix and power scrap ratio vary over time.

### 3.3 SM/SF
- Ore + reductant + electricity framework is correct.
- Result: `plausible`
- Reason: electricity and reductant quality drive large variance.

## 4) Non-ferrous (al/ao/pb/cu/zn/ni/ss/sn)

### 4.1 Aluminum al
- CSV formula: `al = 1.93 * 氧化铝 + 电价 * 13500度 + 辅料`
- Result: `validated (strong)`
- Reason:
  - Hall-Heroult common electricity consumption is often around 14 to 16 MWh/t,
    and practical modern value can sit near 13.5 MWh/t depending on technology.
  - Stoichiometric alumina need is about 1.889 t alumina/t Al; using 1.93 is a
    practical industrial coefficient with losses.

### 4.2 AO (alumina)
- CSV structure is correct (bauxite + caustic + energy + processing).
- Result: `plausible`

### 4.3 Pb/Cu/Zn/Ni/SS/Sn
- Route framework and cost decomposition are correct.
- Result: `needs-internal-benchmark`
- Reason: concentrate grade, TC/RC, recovery rate, and by-product credit change
  materially by smelter and period.

## 5) Agri and oils (m/y/b/p/OI/RM/lh/c/cs)

### 5.1 Soy crush (m/y)
- CSV formula: `压榨利润 = 0.185 * y + 0.785 * m - 大豆进口成本 - 200`
- Result: `validated (strong)`
- Reason:
  - Public crush-spread convention uses approximately 18.3% oil and 80% meal.
  - Your coefficients 18.5% and 78.5% are close and internally consistent.
  - Processing fee as fixed add-on is standard in practical margin modeling.

### 5.2 Soybean import b and palm import p
- Formula structure (FOB/CBOT + basis + freight + tariff + VAT + port fees) is
  correct.
- Result: `validated (structure)`
- Reason: exact tax and fee handling should follow latest customs/tax policy.

### 5.3 OI/RM, lh, c, cs
- Result: `plausible`
- Reason: route logic is right; coefficients depend on regional feed formulas,
  weather, and operation.

## 6) New energy and silicon chain (si/lc/ps)
- CSV framework is directionally correct:
  - `si`: ore + reductant + very high electricity
  - `lc`: ore/brine + chemical reagents + heavy processing fee
  - `ps`: silicon + very high electricity + purification process fee
- Result: `needs-internal-benchmark`
- Reason: wide dispersion across technology routes and plant generations.

## 7) Deep processing and import templates
- Cottonseed crush, starch, ethanol, F55, and import templates are structurally
  sound as trading heuristics.
- Result: `plausible`
- Reason: by-product credit and utility costs are cyclical and location-specific.

## 8) Suggested maintenance rules for this CSV
1. Keep current formula templates, but separate all key coefficients into a
   parameter table (per route, region, and update date).
2. Add columns: `coefficient_source`, `last_verified_date`, `confidence_level`.
3. Maintain three confidence levels exactly as used in this note:
   `validated`, `plausible`, `needs-internal-benchmark`.
4. For `validated` formulas, include at least one public reference link.
5. For `needs-internal-benchmark`, bind coefficients to internal cost curves
   (plant-level if available).

## 9) Quick summary
- Strongly validated now: `TA` (as practical stoichiometric proxy), `al`, `m/y`
  crush coefficients, and import-cost formula structure.
- Most other entries are structurally correct but require internal calibration
  for production use.

## 10) Product-wise formula lines (by subsection)
This section places each CSV formula line under its relevant product subsection.

### 化工
#### TA (精对苯二甲酸)
- 石化法: `TA = 0.655×PX + 加工费`

#### MA (甲醇)
- 煤制: `MA = 1.6×煤价 + 综合费用`
- 天然气制: `MA = 气耗×气价 + 加工费`

#### PF (短纤)
- 聚酯路线: `PF = 0.855×PTA + 0.335×EG + 加工费`

#### PL (丙烯)
- 油制: `PL = 0.9×石脑油 + 1200`
- 煤/甲醇制: `PL = 3×甲醇 + 2000`
- PDH制: `PL = 1.2×丙烷 + 1500`

#### L (聚乙烯)
- 油制: `L ≈ 1.5×石脑油 + 加工费(L)`
- 煤/甲醇制: `L = 3×甲醇 + 1500`
- 乙烷制: `L = 乙烷价格×单耗 + 加工费`

#### PP (聚丙烯)
- 油制: `PP ≈ 1.5×石脑油 + 加工费(PP)`
- 煤/甲醇制: `PP = 3×甲醇 + 1500`
- PDH制: `PP = 1.2×丙烷 + 1500`
- 外采丙烯制: `PP = 丙烯价格 + 加工费`

#### V (聚氯乙烯)
- 电石法: `V = 1.5×电石 + 氯气成本 + 电力成本 + 其他`
- 乙烯法: `V = 0.5×乙烯 + 氯气 + 加工费`

#### EG (乙二醇)
- 石脑油法: `EG = 0.81×石脑油 + 150美元`
- 乙烯法: `EG = 0.605×乙烯×汇率 + 1000`
- 煤制法: `EG = 煤成本 + 综合费用`

#### EB (苯乙烯)
- 乙苯脱氢法: `EB = 0.265×乙烯 + 0.738×纯苯 + 加工费`
- PO/SM联产法: `成本分摊复杂`

#### BZ (纯苯)
- 石油苯: `BZ ≈ 1.4×石脑油 + 1000`

#### PX (对二甲苯)
- 石脑油法: `PX ≈ 1.3×石脑油 + 加工费`

#### LU (低硫燃料油)
- 加氢脱硫: `LU = 燃料油成本 + 氢气成本 + 加工费`

#### FU (燃料油)
- 常减压: `FU = 0.95×原油 + 加工费`

### 建材/盐化工
#### SA (纯碱)
- 氨碱法: `SA = 原盐成本 + 石灰石成本 + 氨耗 + 综合费用`
- 联碱法: `SA = 原盐+氨耗 – 氯化铵收益 + 综合费用`

#### SH (烧碱)
- 离子膜法: `SH = 1.5×原盐 + 电价×2300度 + 加工费`

#### FG (玻璃)
- 浮法玻璃: `FG = 0.2×SA + 0.7×硅砂 + 燃料成本 + 加工费`

### 黑色
#### SM (锰硅)
- 矿热炉: `SM = 2.0×锰矿 + 0.55×焦炭 + 电价×4000度 + 辅料`

#### SF (硅铁)
- 矿热炉: `SF = 1.75×硅石 + 1.0×兰炭 + 电价×8200度 + 加工费`

#### j (焦炭)
- 炼焦: `j = 1.35×焦煤 + 加工费`

#### rb (螺纹钢)
- 长流程: `rb = 1.6×铁矿 + 0.45×焦炭 + 辅料 + 加工费`
- 短流程: `rb = 1.1×废钢 + 电价×450度 + 加工费`

### 有色
#### al (铝)
- 电解铝: `al = 1.93×氧化铝 + 电价×13500度 + 辅料`

#### ao (氧化铝)
- 拜耳法: `ao = 铝土矿成本 + 烧碱成本 + 能源成本 + 加工费`

#### pb (铅)
- 原生铅: `pb = 1.6×铅精矿 + 加工费`
- 再生铅: `pb = 废电池成本 + 拆解冶炼加工费`

#### cu (铜)
- 铜冶炼: `cu = (铜精矿价格×品味×回收率) + 冶炼加工费(TC/RC)`

#### zn (锌)
- 湿法冶炼: `zn = 1.3×锌精矿 + 加工费`

#### ni (镍)
- 火法(镍铁): `ni = 镍矿成本 + 焦炭成本 + 电力 + 加工费`
- 湿法(HPAL): `ni = 镍矿成本 + 酸耗成本 + 加工费`

#### ss (不锈钢)
- 304系: `ss = 0.9×镍铁 + 0.2×铬铁 + 铁水成本 + 加工费`

#### sn (锡)
- 锡冶炼: `sn = 1.3×锡精矿 + 加工费`

### 农产品原料
#### m (豆粕)
- 压榨利润: `压榨利润 = 0.185×y + 0.785×m – 大豆进口成本 – 200`

#### y (豆油)
- 同上(与豆粕压榨利润公式联动): `压榨利润 = 0.185×y + 0.785×m – 大豆进口成本 – 200`

#### b (豆二)
- 进口大豆: `b = CBOT期价 + 升贴水 + 海运费 + 关税+增值税`

#### p (棕榈油)
- 进口成本: `p = FOB + 海运费 + 关税+增值税 + 港杂费`

#### OI (菜籽油)
- 压榨利润: `见下方菜籽压榨`

#### RM (菜籽粕)
- 同上: `见下方菜籽压榨`

#### lh (生猪)
- 养殖成本: `lh = 仔猪成本 + 饲料成本 + 其他`

#### c (玉米)
- 种植/进口: `c = 种植成本(地租、种子、化肥等)或进口完税成本`

#### cs (玉米淀粉)
- 玉米加工: `cs = 1.4×玉米 + 加工费`
- 深加工湿法利润: `利润 = 0.68×淀粉 + 0.065×胚芽 + 0.045×蛋白粉 + 0.10×纤维 – 玉米价 – 350`

### 广州期货
#### si (工业硅)
- 矿热炉: `si = 硅石成本 + 还原剂成本 + 电价×12000度 + 加工费`

#### lc (碳酸锂)
- 锂辉石提锂: `lc = 8×锂辉石价格 + 加工费`
- 盐湖提锂: `lc = 卤水成本 + 加工费`

#### ps (多晶硅)
- 改良西门子法: `ps = 1.2×工业硅 + 电价×50000度 + 加工费`

### 油脂油料深加工/模板
#### Code-less formulas (no交易代码 in CSV)
- 棉籽压榨: `利润 = 0.16×棉油 + 0.49×棉粕 + 0.09×棉短绒 – 棉籽价 – 400`
- 玉米酒精: `利润 = 0.32×酒精 + 0.31×DDGS + 0.08×CO₂ – 玉米价 – 550`
- 果葡糖浆F55: `利润 = F55价格 – 1.05×玉米淀粉 – 900`
- 一体化F55: `利润 = F55价格 – 0.714×玉米价 – 1268`
- 大豆进口成本模板: `成本 = (CBOT+升贴水)×汇率×(1+9%关税)×(1+9%增值税) + 港杂费`
- 菜籽进口成本模板: `成本 = CNF×汇率×(1+9%关税)×(1+9%增值税) + 港杂费`

## 11) Full CSV formula appendix (raw)
The following section preserves the full raw rows from
`docs/commod margin.csv`.

```csv
板块,品种,代码,工艺路线,核心原料/单耗,简化成本/利润公式 (元/吨),备注
化工,精对苯二甲酸,TA,石化法,"PX (0.655 t)","TA = 0.655×PX + 加工费",加工费参考500-700元/吨，含能源、折旧、人工等。
化工,甲醇,MA,煤制,"原料煤 (1.6 t)","MA = 1.6×煤价 + 综合费用",综合费用包括燃料煤、电、水、人工等，约600-800元/吨。
化工,甲醇,MA,天然气制,天然气,"MA = 气耗×气价 + 加工费",天然气单耗约1000立方米/吨，加工费约200-300元/吨。
化工,短纤,PF,聚酯路线,"PTA (0.855 t) + EG (0.335 t)","PF = 0.855×PTA + 0.335×EG + 加工费",加工费参考800-1200元/吨，含聚合、纺丝、包装等。
化工,丙烯,PL,油制,石脑油,"PL = 0.9×石脑油 + 1200",加工费包含蒸汽裂解及分离成本。
化工,丙烯,PL,煤/甲醇制,"甲醇 (3 t)","PL = 3×甲醇 + 2000",甲醇单耗3吨，加工费含MTO/MTP装置费用。
化工,丙烯,PL,PDH制,"丙烷 (1.18 t)","PL = 1.2×丙烷 + 1500",丙烷单耗1.18吨，加工费含脱氢及分离。
化工,聚乙烯,L,油制,石脑油,"L ≈ 1.5×石脑油 + 加工费(L)",加工费(L)通常高于PP，约2000-2500元/吨。
化工,聚乙烯,L,煤/甲醇制,"甲醇 (3 t)","L = 3×甲醇 + 1500",甲醇制烯烃路线，加工费参考1500元/吨。
化工,聚乙烯,L,乙烷制,乙烷,"L = 乙烷价格×单耗 + 加工费",单耗约1.2吨乙烷/吨PE，加工费较低（美国优势）。
化工,聚丙烯,PP,油制,石脑油,"PP ≈ 1.5×石脑油 + 加工费(PP)",加工费(PP)约1800-2200元/吨。
化工,聚丙烯,PP,煤/甲醇制,"甲醇 (3 t)","PP = 3×甲醇 + 1500",与L类似，MTO工艺。
化工,聚丙烯,PP,PDH制,"丙烷 (1.18 t)","PP = 1.2×丙烷 + 1500",丙烷脱氢制丙烯再聚合，综合加工费。
化工,聚丙烯,PP,外采丙烯制,丙烯,"PP = 丙烯价格 + 加工费",加工费约1000-1500元/吨，含聚合、造粒等。
化工,聚氯乙烯,V,电石法,"电石 (1.5 t) + 氯气 + 电力","V = 1.5×电石 + 氯气成本 + 电力成本 + 其他",电石单耗1.5吨，电力约3000度/吨，氯气约0.8吨。
化工,聚氯乙烯,V,乙烯法,"乙烯 (0.5 t) + 氯气","V = 0.5×乙烯 + 氯气 + 加工费",加工费约1500-2000元/吨，含氧氯化、聚合等。
化工,乙二醇,EG,石脑油法,"石脑油 (0.81 t)","EG = 0.81×石脑油 + 150美元",150美元/吨加工费，需换算人民币。
化工,乙二醇,EG,乙烯法,"乙烯 (0.605 t)","EG = 0.605×乙烯×汇率 + 1000",加工费1000元/吨（含氧气、催化剂等）。
化工,乙二醇,EG,煤制法,煤炭,"EG = 煤成本 + 综合费用",综合费用约3500-4500元/吨（含电、蒸汽、折旧）。
化工,苯乙烯,EB,乙苯脱氢法,"乙烯 (0.265 t) + 纯苯 (0.738 t)","EB = 0.265×乙烯 + 0.738×纯苯 + 加工费",加工费800-1200元/吨，含乙苯合成、脱氢。
化工,苯乙烯,EB,PO/SM联产法,"乙烯、丙烯、纯苯",成本分摊复杂,副产品环氧丙烷，成本需联产分摊。
化工,纯苯,BZ,石油苯,石脑油,"BZ ≈ 1.4×石脑油 + 1000",催化重整/裂解汽油抽提，加工费1000元/吨。
化工,对二甲苯,PX,石脑油法,石脑油,"PX ≈ 1.3×石脑油 + 加工费",加工费1500-2000元/吨，含重整、吸附分离等。
化工,低硫燃料油,LU,加氢脱硫,"原油 + 氢气","LU = 燃料油成本 + 氢气成本 + 加工费",加氢成本占15%左右，加工费约300-500元/吨。
化工,燃料油,FU,常减压,原油,"FU = 0.95×原油 + 加工费",直馏燃料油，加工费约200-300元/吨。
建材/盐化工,纯碱,SA,氨碱法,"原盐、石灰石、氨","SA = 原盐成本 + 石灰石成本 + 氨耗 + 综合费用",综合费用含蒸汽、电、折旧等，约1000-1500元/吨。
建材/盐化工,纯碱,SA,联碱法,"原盐、氨、CO₂","SA = 原盐+氨耗 – 氯化铵收益 + 综合费用",联碱法副产氯化铵，成本低于氨碱法。
建材/盐化工,烧碱,SH,离子膜法,"原盐 (1.5 t) + 电力","SH = 1.5×原盐 + 电价×2300度 + 加工费",电力占成本60%以上，加工费约500-800元/吨。
建材/盐化工,玻璃,FG,浮法玻璃,"纯碱(0.2 t) + 硅砂(0.7 t) + 燃料","FG = 0.2×SA + 0.7×硅砂 + 燃料成本 + 加工费",燃料成本占30%，加工费含电、人工、折旧等。
铁合金,锰硅,SM,矿热炉,"锰矿 (2.0 t) + 焦炭 (0.55 t) + 电力","SM = 2.0×锰矿 + 0.55×焦炭 + 电价×4000度 + 辅料",电力单耗约3800-4200度/吨，辅料约200元/吨。
铁合金,硅铁,SF,矿热炉,"硅石(1.75 t) + 兰炭(1.0 t) + 电力","SF = 1.75×硅石 + 1.0×兰炭 + 电价×8200度 + 加工费",加工费含电极、人工等，约500元/吨。
黑色,焦炭,j,炼焦,"焦煤 (1.35 t)","j = 1.35×焦煤 + 加工费",加工费200-300元/吨，含熄焦、筛分等。
黑色,螺纹钢,rb,长流程,"铁矿(1.6 t) + 焦炭(0.45 t) + 辅料","rb = 1.6×铁矿 + 0.45×焦炭 + 辅料 + 加工费",辅料约200元/吨，加工费1000-1500元/吨（含冶炼、轧制）。
黑色,螺纹钢,rb,短流程,"废钢(1.1 t) + 电力","rb = 1.1×废钢 + 电价×450度 + 加工费",加工费约800-1000元/吨，含电炉冶炼、连铸、轧制。
有色,铝,al,电解铝,"氧化铝 (1.93 t) + 电力","al = 1.93×氧化铝 + 电价×13500度 + 辅料",辅料含冰晶石、氟化铝等，约300元/吨。
有色,氧化铝,ao,拜耳法,"铝土矿 (2.7 t) + 烧碱 (0.14 t) + 能源","ao = 铝土矿成本 + 烧碱成本 + 能源成本 + 加工费",加工费约300-500元/吨，能源主要为蒸汽。
有色,铅,pb,原生铅,"铅精矿 (1.6 t)","pb = 1.6×铅精矿 + 加工费",加工费2000-3000元/吨，含冶炼、精炼。
有色,铅,pb,再生铅,废铅蓄电池,"pb = 废电池成本 + 拆解冶炼加工费",加工费约1500-2000元/吨。
有色,铜,cu,铜冶炼,"铜精矿 (1.2-1.5 t)","cu = (铜精矿价格×品味×回收率) + 冶炼加工费(TC/RC)",加工费以TC/RC形式体现，浮动。
有色,锌,zn,湿法冶炼,"锌精矿 (1.3 t)","zn = 1.3×锌精矿 + 加工费",加工费5000-6000元/吨，含焙烧、浸出、电积。
有色,镍,ni,火法（镍铁）,"红土镍矿 (4-5 t) + 焦炭","ni = 镍矿成本 + 焦炭成本 + 电力 + 加工费",镍铁成本取决于品位，电耗高。
有色,镍,ni,湿法（HPAL）,"红土镍矿 + 硫酸","ni = 镍矿成本 + 酸耗成本 + 加工费",加工费含高压酸浸、中和等，约8000-10000元/吨镍。
有色,不锈钢,ss,304系,"镍铁(0.9 t) + 铬铁(0.2 t) + 铁水","ss = 0.9×镍铁 + 0.2×铬铁 + 铁水成本 + 加工费",加工费约1500-2000元/吨，含AOD精炼、连铸、热轧。
有色,锡,sn,锡冶炼,"锡精矿 (1.3 t)","sn = 1.3×锡精矿 + 加工费",加工费15000-20000元/吨，含冶炼、精炼。
农产品原料,豆粕,m,压榨利润,进口大豆,"压榨利润 = 0.185×y + 0.785×m – 大豆进口成本 – 200",出油率18.5%，出粕率78.5%，加工费200元/吨。
农产品原料,豆油,y,同上,同上,同上,同上
农产品原料,豆二,b,进口大豆,大豆到岸成本,"b = CBOT期价 + 升贴水 + 海运费 + 关税+增值税",进口成本公式，不含压榨利润。
农产品原料,棕榈油,p,进口成本,马来/印尼FOB,"p = FOB + 海运费 + 关税+增值税 + 港杂费",关税9%，增值税9%，港杂费约100元/吨。
农产品原料,菜籽油,OI,压榨利润,进口/国产菜籽,见下方菜籽压榨,分进口浸出和国产小榨两种工艺。
农产品原料,菜籽粕,RM,同上,同上,同上,同上
农产品原料,生猪,lh,养殖成本,"仔猪 + 饲料（玉米、豆粕）","lh = 仔猪成本 + 饲料成本 + 其他",其他含人工、水电、动保、折旧等，单位：元/公斤。
农产品原料,玉米,c,种植/进口,种植成本/进口完税,"c = 种植成本（地租、种子、化肥等）或进口完税成本",进口成本含关税、增值税。
农产品原料,玉米淀粉,cs,玉米加工,"玉米 (1.4 t)","cs = 1.4×玉米 + 加工费",加工费300-500元/吨，含浸泡、破碎、分离、干燥。
广州期货,工业硅,si,矿热炉,"硅石(2.8 t) + 碳质还原剂(2.0 t) + 电力","si = 硅石成本 + 还原剂成本 + 电价×12000度 + 加工费",加工费约2000-3000元/吨，含电极、人工等。
广州期货,碳酸锂,lc,锂辉石提锂,"锂辉石 (8 t) + 硫酸 + 纯碱","lc = 8×锂辉石价格 + 加工费",加工费约30000-40000元/吨，含焙烧、酸化、提纯。
广州期货,碳酸锂,lc,盐湖提锂,卤水,"lc = 卤水成本 + 加工费",卤水成本低，加工费约20000-30000元/吨。
广州期货,多晶硅,ps,改良西门子法,"工业硅 (1.2 t) + 电力","ps = 1.2×工业硅 + 电价×50000度 + 加工费",加工费约20000-30000元/吨，含三氯氢硅合成、还原、尾气处理。
油脂油料深加工,棉籽压榨,,预榨-浸出,棉籽,"利润 = 0.16×棉油 + 0.49×棉粕 + 0.09×棉短绒 – 棉籽价 – 400",出油率16%，出粕49%，短绒9%，加工费400元/吨。
油脂油料深加工,玉米淀粉,cs,湿法加工,玉米,"利润 = 0.68×淀粉 + 0.065×胚芽 + 0.045×蛋白粉 + 0.10×纤维 – 玉米价 – 350",得率及副产品收益，加工费350元/吨。
油脂油料深加工,玉米酒精,,发酵蒸馏,玉米,"利润 = 0.32×酒精 + 0.31×DDGS + 0.08×CO₂ – 玉米价 – 550",加工费550元/吨，部分装置有玉米油收益。
油脂油料深加工,果葡糖浆F55,,淀粉糖,"玉米淀粉 (1.05 t)","利润 = F55价格 – 1.05×玉米淀粉 – 900",加工费900元/吨（含酶制剂、能源）。
油脂油料深加工,一体化F55,,玉米→糖,玉米,"利润 = F55价格 – 0.714×玉米价 – 1268",将玉米制淀粉成本折算后的一体化公式。
进口成本公式,大豆进口,,,CBOT+升贴水,"成本 = (CBOT+升贴水)×汇率×(1+9%关税)×(1+9%增值税) + 港杂费",港杂费约50-80元/吨。
进口成本公式,菜籽进口,,,CNF,"成本 = CNF×汇率×(1+9%关税)×(1+9%增值税) + 港杂费",同上。
```
