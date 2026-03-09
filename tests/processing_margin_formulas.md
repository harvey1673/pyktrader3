# Processing Margin / Crack Relationship Formulas

## Conversion conventions

- Use all prices in **CNY/ton** where possible.
- `SC` (INE crude) is in **CNY/bbl**; convert to CNY/ton with:
  - `SC_ton = 7.33 * SC`
- If a yield is involved, convert feedstock cost by:
  - `Feed cost per ton product = SC_ton / yield`

---

## 1) Petrochemical chain

### a) `SC -> PX -> PTA`

- **PX from crude (proxy margin):**
  - `M_PX = PX - alpha_PX * SC`
  - where `alpha_PX ~= 7.33 / y_PX` (if `y_PX=0.20`, `alpha_PX~=36.7`)

- **PTA from PX (proxy margin):**
  - `M_PTA = PTA - 0.655 * PX - proc_fee_PTA`

### b) `PTA + EG -> PF`

- **PF polymerization margin (proxy):**
  - `M_PF = PF - (0.86 * PTA + 0.34 * EG) - poly_fee`

---

## 2) Ferrous / metals / chemicals

### a) `i + j -> rb / hc` (blast-furnace steel)

- `M_RB = RB - (1.6 * I + 0.55 * J) - BF_fee`
- `M_HC = HC - (1.6 * I + 0.55 * J) - BF_fee`

### b) `jm -> j` (coking)

- `M_J = J - 1.35 * JM - coking_fee`

### c) `SA -> FG` (glass)

- `M_FG = FG - 0.20 * SA - energy_fee`

### d) `ni -> ss` (stainless)

- `M_SS = SS - 0.08 * NI - other_alloy_energy_fee`

### e) `MA -> pp` (MTO-to-PP proxy)

- `M_PP = PP - 3.0 * MA - MTO_fee`

---

## 3) Cracks vs crude (`SC`)

- **Fuel oil crack:**
  - `C_FU = FU - beta_FU * SC`, with `beta_FU ~= 6.35`

- **Low-sulfur fuel oil crack:**
  - `C_LU = LU - beta_LU * SC`, with `beta_LU ~= 6.35 to 6.60`

- **Bitumen crack:**
  - `C_BU = BU - beta_BU * SC`, with `beta_BU ~= 6.20 to 6.40`

---

## Practical note

For cross-product comparison, also monitor normalized margins:

- `margin_rate = margin_abs / feedstock_cost`

and compute rolling z-score / percentile rank for signal generation.
