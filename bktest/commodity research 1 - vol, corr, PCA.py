#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import datetime
sys.path.append("C:/dev/pycmqlib/")
sys.path.append("C:/dev/pycmqlib/scripts/")
import ts_tool
import dbaccess as db
import misc
import stats_test
import backtest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api
from statsmodels.graphics.tsaplots import plot_acf

start_date = datetime.date(2016,1,1)
end_date = datetime.date(2020,3,25)

ferrous_products_mkts = ['rb', 'hc', 'i', 'j', 'jm']
ferrous_mixed_mkts = ['ru', 'FG', 'ZC', 'SM', "SF"]
base_metal_mkts = ['cu', 'al', 'zn', 'ni', 'sn', 'pb']
precious_metal_mkts = ['au', 'ag']
ind_metal_mkts = ferrous_products_mkts + ferrous_mixed_mkts + base_metal_mkts  
petro_chem_mkts = ['l', 'pp', 'v', 'TA', 'MA', 'bu'] #, 'sc', 'fu', 'eg']
ind_all_mkts = ind_metal_mkts + petro_chem_mkts
ags_oil_mkts = ['m', 'RM', 'y', 'p', 'OI', 'a', 'c', 'cs'] #, 'b']
ags_soft_mkts = ['CF', 'SR', 'jd']#, 'AP', 'sp']
ags_all_mkts = ags_oil_mkts + ags_soft_mkts
eq_fut_mkts = ['IF', 'IH', 'IC']
bond_fut_mkts = ['T', 'TF']
fin_all_mkts = eq_fut_mkts + bond_fut_mkts
commod_all_mkts = ind_all_mkts + ags_all_mkts # + precious_metal_mkts
all_markets = commod_all_mkts


# In[2]:


need_shift = 2
freq = 'd'
args = {'n': 1, 'roll_rule': '-35b', 'freq': freq, 'need_shift': need_shift}
ferrous_products_args = args
ferro_mixed_mkt_args = args
base_args = {'n': 1, 'roll_rule': '-30b', 'freq': freq, 'need_shift': need_shift}
base2_args = {'n': 1, 'roll_rule': '-40b', 'freq': freq, 'need_shift': need_shift}
eq_args = {'n': 1, 'roll_rule': '-1b', 'freq': freq, 'need_shift': need_shift}
bond_args = {'n': 1, 'roll_rule': '-20b', 'freq': freq, 'need_shift': need_shift}
precious_args = {'n': 1, 'roll_rule': '-25b', 'freq': freq, 'need_shift': need_shift}
df_list = []
for asset in all_markets:
    use_args = args
    if asset in eq_fut_mkts:
        use_args = eq_args
    elif asset in ['cu', 'al', 'zn', 'pb', 'sn']:
        use_args = base_args
    elif asset in ['ni']:
        use_args = base2_args
    elif asset in bond_fut_mkts:
        use_args = bond_args
    elif asset in precious_metal_mkts:
        use_args = precious_args
    use_args['start_date'] = max(backtest.sim_start_dict[asset], start_date)
    use_args['end_date'] = end_date
    print "loading mkt = %s, args = %s" % (asset, use_args)
    df = misc.nearby(asset, **use_args)
    df.rename(columns={'close': asset + '_fut'}, inplace=True)
    df_list.append(df[[asset + '_fut']])

merged = ts_tool.merge_df(df_list)
print merged[-10:]
merged.to_csv("C:\\dev\\data\\commod_data_load.csv")


# In[3]:


#merged = merged.dropna()
xdata = pd.read_csv("C:\\dev\\data\\commod_data_load.csv")
xdata['date'] = xdata['date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
xdata = xdata.set_index('date').dropna()
logret = np.log(xdata).diff().dropna()


# In[18]:


from sklearn.decomposition import PCA
start_d = datetime.date(2019,1,1)
end_d = datetime.date(2020,3,25)
selected_logret = logret[(logret.index >= start_d) & (logret.index<=end_d)]
pca = PCA(n_components=10)
pca.fit(selected_logret.dropna())
explained_ratio = pca.explained_variance_ratio_


# In[19]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(selected_logret.columns))
ax.barh(y_pos, pca.components_[0], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(selected_logret.columns)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC1')
plt.show()

fig, ax = plt.subplots()
y_pos = np.arange(len(selected_logret.columns))
ax.barh(y_pos, pca.components_[1], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(selected_logret.columns)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC2')
plt.show()

fig, ax = plt.subplots()
y_pos = np.arange(len(selected_logret.columns))
ax.barh(y_pos, pca.components_[2], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(selected_logret.columns)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC3')
plt.show()


# In[20]:


col_list = ['rb_fut', 'hc_fut', 'i_fut', 'j_fut', 'jm_fut'] #, 'ZC_fut', 'ru_fut', 'ni_fut', 'cu_fut', 'pp_fut', 'SM_fut']
pca2 = PCA(n_components=5)
pca2.fit(selected_logret[col_list].dropna())
explained_ratio2 = pca2.explained_variance_ratio_
print explained_ratio2
print(pca2.components_)  


# In[21]:


plt.rcdefaults()
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
y_pos = np.arange(len(col_list))
ax = fig.add_subplot(2, 1, 1)
ax.barh(y_pos, pca2.components_[0], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(col_list)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC1')

ax = fig.add_subplot(2, 1, 2)
y_pos = np.arange(len(col_list))
ax.barh(y_pos, pca2.components_[1], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(col_list)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC2')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)
y_pos = np.arange(len(col_list))
ax.barh(y_pos, pca2.components_[2], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(col_list)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC3')

ax = fig.add_subplot(3, 1, 2)
y_pos = np.arange(len(col_list))
ax.barh(y_pos, pca2.components_[3], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(col_list)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC4')

ax = fig.add_subplot(3, 1, 3)
y_pos = np.arange(len(col_list))
ax.barh(y_pos, pca2.components_[4], align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(col_list)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('products')
ax.set_title('PC5')
plt.show()

