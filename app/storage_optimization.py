import pulp
import datetime
import sys
from pycmqlib3.analytics import bsopt
from pycmqlib3.utility import misc

inj_rate = 0.0
wdr_rate = 20000.0

inv_s = 50000
inv_max = 50000
inv_min = 0

fwd = [4080, 3977, 3917, 3862, 3801, 3745, 3713, 3687, 3661, 3632, 3606, 3579, 3552]
#fwd = [4080, 4070, 4060, 4050, 4040, 4030, 4020, 4010, 4000, 4000, 4000, 4000, 4000]
nlen = len(fwd)
idx_list = list(range(nlen))
tenors = [misc.day_shift(datetime.date(2020,12,1), '%dm' % i) for i in range(nlen)]
expiries = [ misc.day_shift(ten, '10b', misc.CHN_Holidays) for ten in tenors]
cov_xy = set([(i, j) for i in idx_list for j in idx_list if i!=j])
Cxy = {}

daily_mov = 10
fade_fact = 0.95
is_call = False
strike = 0.0
ir = 0.0
tday = datetime.date(2020,11,25)

for i, j in cov_xy:
    fwd1 = fwd[i]
    fwd2 = fwd[j]
    front_idx = min(i,j)
    dvol = 10 * 15.5/abs(i-j) * (0.95**front_idx)
    t_exp = (expiries[front_idx] - tday).days/365.25
    Cxy[(i,j)] = bsopt.BSFwdNormal(is_call, fwd1-fwd2, strike, dvol, t_exp, ir)

print([[Cxy[(i,j)] if i!= j else 0 for j in idx_list] for i in idx_list])


prob = pulp.LpProblem("PlantOptimizer", pulp.LpMaximize)

w = pulp.LpVariable.dicts('w', cov_xy, lowBound = 0, upBound = inv_max)
u = pulp.LpVariable.dicts('u', idx_list, lowBound = -inv_max, upBound = inv_max)
prob += sum([w[input] * Cxy[input] for input in cov_xy]) + sum([u[input] * fwd[input] for input in idx_list])
for i in idx_list:
    prob += sum([w[(i, j)] for j in idx_list if j!=i]) - u[i] <= inj_rate
    prob += sum([w[(j, i)] for j in idx_list if j!=i]) + u[i] <= wdr_rate
    prob += inv_s + sum([sum([w[(k, j)] for j in idx_list if j > i]) - u[k] for k in idx_list if k<=i]) <= inv_max
    prob += inv_s + sum([-sum([w[(j, k)] for j in idx_list if j > i]) - u[k] for k in idx_list if k<=i]) >= inv_min
prob.solve()
for v in prob.variables():
    print(v.name, v.varValue)
print(prob.objective.value())
