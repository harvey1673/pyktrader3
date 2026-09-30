# CTD spot-price requirements

This is the acquisition shortlist behind `ctd_spot_requirements.py`. A time
series is usable only if the accompanying metadata proves the quote can be
converted to an eligible exchange delivery basis.

## Highest-priority gaps

| Product | Required candidate prices | Existing coverage | Main missing metadata/data |
|---|---|---|---|
| J | Rizhao wet-quench sub-grade-1; Jinzhong wet-quench; Lvliang dry-quench; Tianjin/port alternatives | Good regional and port coverage | Time-varying A/S/M40/M10/CRI/CSR/Mt/Mf assays; freight and loading |
| JM | Mongolian No. 5; Shanxi mid-sulfur; Shanxi low-sulfur; premium Australian hard coking coal | Ganqimaodu stock/outstock, Lvliang/Tangshan assay-labelled quotes, Australian Jingtang and CFR/FOB quotes are mapped | Verify exact grade/origin and raw/washed status; complete assay, port/factory basis and freight |
| SS | Registered-brand 304/2B coil in Wuxi and Foshan by specification | Aggregate Wuxi mill-edge and basis series | Brand, registered status, thickness, width and edge |
| SM | 6517 at Tianjin, Ulanqab, Shizuishan, Qinzhou, Rizhao and Yingkou | Tianjin, Inner Mongolia, Guangxi, Gansu, national average | Exact city and freight to each delivery point |
| SF | Grade 72 at Tianjin, Zhongwei, Inner Mongolia, Gansu and consumer areas | Ningxia, Inner Mongolia, Gansu, national 72/75 | Direct Tianjin/consumer prices and freight |

## Petrochemicals and rubber

| Product | Recommended CTD candidate panel | Essential metadata |
|---|---|---|
| L / PP | Registered producer-brand spot in East, North and South China | Producer, brand code, grade test, warehouse/factory pickup |
| V | Calcium-carbide and ethylene-route registered brands by region | Process, brand, grade, location and freight |
| EG / EB | East-China tank/warehouse spot, North/South spot and CFR imports | Tank/warehouse, incoterm, tax and port charges |
| TA | Registered and inspection-exempt PTA brands in East China | Brand status and pickup location |
| PX | East-China ex-works plus Taiwan CFR and Korea FOB | Grade, currency, tax, freight and delivery location |
| MA | Jiangsu/Zhejiang, Shandong and Inner Mongolia by factory-pickup point | Grade, pickup location, tax and freight |
| UR | Henan/Shandong/North China by small/large-granule grade | Grade, dynamic location group and freight |
| SC | Oman, Dubai, ESPO and every named deliverable crude grade | Exchange grade premium, FX, tax, freight, insurance and tank fees |
| FU / LU | Singapore cargo/bunker quotes and East-China delivered prices | Grade/sulfur, FX, tax, freight, storage and quote convention |
| BU | Registered heavy-asphalt brands in Shandong/North/East China | Brand, grade, registered status and pickup point |
| RU | Registered SCR WF brands in Yunnan, Hainan, Jiangsu and Zhejiang | Brand, production year, region and warehouse |
| NR | Registered TSR20 and TSR10 by origin | Origin, brand, production year, grade and cross-border status |
| BR | Registered BR9000 brands by warehouse/factory pickup | Producer, brand, production batch and pickup guidance |

## Quote metadata schema

At minimum, store these fields alongside each price column:

- `product`, `price_col`, `quote_location`, `quote_basis`, `tax_status`,
  `currency`, `unit`, and `observation_time`;
- `brand`, `registered_status`, `grade/spec`, and all product-specific assay
  fields;
- delivery point, freight/loading/storage/finance conversion, and the source of
  each conversion;
- effective start/end contract or date for eligibility and exchange premiums.

The CTD engine returns an audit table containing raw price, cash conversion,
quality/location/brand adjustments, weight multiplier, eligibility and final
futures-equivalent value for every candidate.
