import datetime
import numpy as np
import misc
from bsopt import *

def LookbackFltStrike(IsCall, S, strike, Vol, Texp, Rd, Rf, mflag = 'd'):
    if IsCall and (strike > S):
        raise ValueError('Lookback call need strike is less than S')
    elif (not IsCall) and (strike < S):
        raise ValueError('Lookback put need strike is less than S')
    if IsCall:
        call_flag = 1.0
    else:
        call_flag = -1.0
    if Texp <= 0:
        return (S - strike) * call_flag
    if mflag in ['d', 'D']:
        discreteAdj = exp(0.5826 * Vol/sqrt(252.0) * call_flag)
    else:
        discreteAdj = 1.0
    strike = strike/discreteAdj
    df = exp(-Rd * Texp)
    dd = exp(-Rf * Texp)
    fd1 = d1(S, strike, Vol, Texp, Rd, Rf)
    fd2 = d2(S, strike, Vol, Texp, Rd, Rf)
    fd3 = fd1 - 2 * (Rd - Rf) * sqrt(Texp)/Vol
    beta = 2 * (Rd - Rf) / (Vol * Vol)
    val = call_flag * (S * dd * cnorm(fd1*call_flag) - strike * df * cnorm(fd2*call_flag))
    if abs(Rd - Rf) >= 1e-4:
        val -= call_flag * S/beta * ( dd * cnorm(-fd1*call_flag) - df * ((strike/S)**beta) * cnorm(-fd3*call_flag))
    else:
        val += S * df * Vol * sqrt(Texp) * (pnorm(-fd1*call_flag) - call_flag * fd1 * cnorm(-fd1*call_flag))
    val = val * discreteAdj - call_flag * (discreteAdj - 1) * S
    return val

def MinOptionOnSpdCall(F1, F2, dv1, dv2, rho, K1, K2, T):
    ' min(max(F1-K1),max(F2-K2)) assuming F1 F2 are spread of two assets'
    v1 = dv1 * numpy.sqrt(T)
    v2 = dv2 * numpy.sqrt(T)
    def int_func1(x):
        return  scipy.stats.norm.cdf(((F1-K1)-(F2-K2) + (v1 * rho - v2) * x)/(v1 * numpy.sqrt(1-rho**2))) \
                        * (v2 * x + F2- K2) * scipy.stats.norm.pdf(x)

    def int_func2(x):
        return  scipy.stats.norm.cdf(((F2-K2)-(F1-K1) + (v2 * rho - v1) * x)/(v2 * numpy.sqrt(1-rho**2))) \
                        * (v1 * x + F1- K1) * scipy.stats.norm.pdf(x)
    res1 = quad(int_func1, (K2-F2)/v2, numpy.inf)
    res2 = quad(int_func2, (K1-F1)/v1, numpy.inf)
    return res1[0] + res2[0]

def OneTouch( IsHigh, IsDelayed, Spot, Strike, Vol, Texp, Rd, Rf ):
    '''Prices a one touch option. IsHigh=True means it knocks up and in; False
    means down and in. IsDelayed=True means it pays at the end; False means it
    pays on hit.'''

    if ( IsHigh and Spot >= Strike ) or ( not IsHigh and Spot <= Strike ):
        if IsDelayed:
            return exp( -Rd * Texp )
        else:
            return 1

    if Vol <= 0 or Texp <= 0: return 0

    Alpha = log( Strike / float( Spot ) )
    Mu    = Rd - Rf - Vol * Vol / 2.

    if IsDelayed:
        if IsHigh:
            Price = exp( -Rd * Texp ) * ( cnorm( ( -Alpha + Mu * Texp ) / Vol / sqrt( Texp ) ) \
                  + exp( 2 * Mu * Alpha / Vol / Vol ) * cnorm( ( -Alpha - Mu * Texp ) / Vol / sqrt( Texp ) ) )
        else:
            Price = exp( -Rd * Texp ) * ( cnorm( (  Alpha - Mu * Texp ) / Vol / sqrt( Texp ) ) \
                  + exp( 2 * Mu * Alpha / Vol / Vol ) * cnorm( (  Alpha + Mu * Texp ) / Vol / sqrt( Texp ) ) )
    else:
        MuHat = sqrt( Mu * Mu + 2 * Rd * Vol * Vol )
        if IsHigh:
            Price = exp( Alpha / Vol / Vol * ( Mu - MuHat ) ) * cnorm( ( -Alpha + MuHat * Texp ) / Vol / sqrt( Texp ) ) \
                  + exp( Alpha / Vol / Vol * ( Mu + MuHat ) ) * cnorm( ( -Alpha - MuHat * Texp ) / Vol / sqrt( Texp ) )
        else:
            Price = exp( Alpha / Vol / Vol * ( Mu + MuHat ) ) * cnorm( (  Alpha + MuHat * Texp ) / Vol / sqrt( Texp ) ) \
                  + exp( Alpha / Vol / Vol * ( Mu - MuHat ) ) * cnorm( (  Alpha - MuHat * Texp ) / Vol / sqrt( Texp ) )

    return Price

def BSKnockout( IsCall, Spot, Strike, KO, IsUp, Vol, Texp, Rd, Rf ):
    '''Knockout option with a continuous barrier: price under constant vol, constant drift BS model.'''

    if ( Spot >= KO and IsUp ) or ( Spot <= KO and not IsUp ): return 0. # knocked

    Mu = Rd - Rf
    SqrtT = sqrt( Texp )

    # as per Haug

    Phi = IsCall and 1 or -1
    Eta = IsUp and -1 or 1

    m  = ( Mu - 0.5 * Vol * Vol ) / Vol / Vol
    Lambda = sqrt( m * m + 2. * Mu / Vol / Vol )
    x1 = log( Spot / Strike ) / Vol / SqrtT + ( 1 + m ) * Vol * SqrtT
    x2 = log( Spot / KO ) / Vol / SqrtT + ( 1 + m ) * Vol * SqrtT
    y1 = log( KO * KO / Spot / Strike ) / Vol / SqrtT + ( 1 + m ) * Vol * SqrtT
    y2 = log( KO / Spot ) / Vol / SqrtT + ( 1 + m ) * Vol * SqrtT

    A = Phi * Spot * exp( -Rf * Texp ) * cnorm( Phi * x1 ) - Phi * Strike * exp( -Rd * Texp ) * cnorm( Phi * x1 - Phi * Vol * SqrtT )
    B = Phi * Spot * exp( -Rf * Texp ) * cnorm( Phi * x2 ) - Phi * Strike * exp( -Rd * Texp ) * cnorm( Phi * x2 - Phi * Vol * SqrtT )
    C = Phi * Spot * exp( -Rf * Texp ) * ( KO / Spot ) ** ( 2 * ( m + 1 ) ) * cnorm( Eta * y1 ) - Phi * Strike * exp( -Rd * Texp ) * ( KO / Spot ) ** ( 2 * m ) * cnorm( Eta * y1 - Eta * Vol * SqrtT )
    D = Phi * Spot * exp( -Rf * Texp ) * ( KO / Spot ) ** ( 2 * ( m + 1 ) ) * cnorm( Eta * y2 ) - Phi * Strike * exp( -Rd * Texp ) * ( KO / Spot ) ** ( 2 * m ) * cnorm( Eta * y2 - Eta * Vol * SqrtT )

    if Strike < KO:
        if IsCall and IsUp:
            return A - B + C - D
        elif IsCall and not IsUp:
            return B - D
        elif not IsCall and IsUp:
            return A - C
        else:
            return 0
    else:
        if IsCall and IsUp:
            return 0
        elif IsCall and not IsUp:
            return A - C
        elif not IsCall and IsUp:
            return B - D
        else:
            return A - B + C - D

def calc_avg(fwd, fixlist, tday):
    fixings = [ fix if (d<=tday) and (fix > 0) else fwd for (d, fix) in fixlist]
    return np.mean(fixings)

def calc_cross_m2(fwd1, fwd2, vol1, vol2, corr, fixlist1, fixlist2, tday, expiry, accr = 'act365'):
    m2 = 0.0
    fwd_prod = fwd1 * fwd2
    vol_prod = vol1 * vol2 *corr
    for (t1, fix1) in fixlist1:
        if (t1 <= tday):
            continue
        for (t2, fix2) in fixlist2:
            if (t2 <= tday):
                continue
            tau = min(min(t1, t2), expiry)
            if tau > tday:
                m2 += fwd_prod * (np.exp(misc.conv_expiry_date(tday + datetime.timedelta(days = 1), tau, accrual = accr) * vol_prod) -1)
    return m2/len(fixlist1)/len(fixlist2)

def CalSpdAsianOption(IsCall, F1, F2, K, vol1, vol2, corr, fixlist1, fixlist2, tday, expiry, rd = 0.0, accr = 'act365', eod_flag = True):
    if not eod_flag:
        tday = tday - datetime.timedelta(days = 1)
    avg = [ calc_avg(fwd, fixes, tday) for (fwd, fixes) in [(F1, fixlist1), (F2, fixlist2)]]
    V1 = calc_cross_m2(F1, F1, vol1, vol1, 1.0, fixlist1, fixlist1, tday, expiry, accr = accr)
    V2 = calc_cross_m2(F2, F2, vol2, vol2, 1.0, fixlist2, fixlist2, tday, expiry, accr=accr)
    V12 = calc_cross_m2(F1, F2, vol1, vol2, corr, fixlist1, fixlist2, tday, expiry, accr=accr)
    var = V1 + V2 - 2 * V12
    avg_fwd = avg[0] - avg[1]
    if var <= 0.0:
        p = max(avg_fwd - K, 0.0)
    else:
        d = (avg_fwd - K) / np.sqrt(var)
        p = (avg_fwd - K) * cnorm(d) + sqrt(var) * pnorm(d)
    if not IsCall:
        p = p - avg_fwd + K
    return p * np.exp( - rd * misc.conv_expiry_date(tday, expiry, accr))

def test_calspdopt():
    fixlist1 = [(datetime.date(2018,12,3), 66.15),
                (datetime.date(2018, 12, 4), 67.60),
                (datetime.date(2018, 12, 5), 67.55),
                (datetime.date(2018, 12, 6), 66.10),
                (datetime.date(2018, 12, 7), 66.95),
                (datetime.date(2018, 12, 10), 66.80),
                (datetime.date(2018, 12, 11), 66.95),
                (datetime.date(2018, 12, 12), 67.15),
                (datetime.date(2018, 12, 13), 68.20),
                (datetime.date(2018, 12, 14), 69.90),
                (datetime.date(2018, 12, 17), 70.55),
                (datetime.date(2018, 12, 18), 69.9),
                (datetime.date(2018, 12, 19), 0.0),
                (datetime.date(2018, 12, 20), 0.0),
                (datetime.date(2018, 12, 21), 0.0),
                (datetime.date(2018, 12, 26), 0.0),
                (datetime.date(2018, 12, 27), 0.0),
                (datetime.date(2018, 12, 28), 0.0),
                ]
    fixlist2 = [(datetime.date(2019, 1, 2), 66.15),
                (datetime.date(2019, 1, 3), 66.15),
                (datetime.date(2019, 1, 4), 67.60),
                (datetime.date(2019, 1, 7), 67.55),
                (datetime.date(2019, 1, 8), 66.10),
                (datetime.date(2019, 1, 9), 66.95),
                (datetime.date(2019, 1, 10), 66.80),
                (datetime.date(2019, 1, 11), 66.95),
                (datetime.date(2019, 1, 14), 67.15),
                (datetime.date(2019, 1, 15), 68.20),
                (datetime.date(2019, 1, 16), 69.90),
                (datetime.date(2019, 1, 17), 70.55),
                (datetime.date(2019, 1, 18), 0.0),
                (datetime.date(2019, 1, 21), 0.0),
                (datetime.date(2019, 1, 22), 0.0),
                (datetime.date(2019, 1, 23), 0.0),
                (datetime.date(2019, 1, 24), 0.0),
                (datetime.date(2019, 1, 25), 0.0),
                (datetime.date(2019, 1, 28), 0.0),
                (datetime.date(2019, 1, 29), 0.0),
                (datetime.date(2019, 1, 30), 0.0),
                (datetime.date(2019, 1, 31), 0.0),
                ]
    fixlist3 = [(datetime.date(2019, 2, 1), 66.15),
                (datetime.date(2019, 2, 4), 66.15),
                (datetime.date(2019, 2, 7), 67.55),
                (datetime.date(2019, 2, 8), 67.55),]
    tday = datetime.date(2018,12,19)
    eod_flag = False
    accr = 'act252'
    fwd1 = 70.8
    fwd2 = 68.6
    fwd3 = 67.55
    vol1 = 0.25
    vol2 = 0.25
    vol3 = 0.25
    corr = 0.99625
    rd = 0.025
    expiry = datetime.date(2019,2,8)
    p12 = CalSpdAsianOption(False, fwd1, fwd2, 0.0, vol1, vol2, corr, fixlist1, fixlist2, tday, expiry, rd = rd, accr = accr, eod_flag=eod_flag)
    p13 = CalSpdAsianOption(False, fwd1, fwd3, 0.0, vol1, vol3, corr, fixlist1, fixlist3, tday, expiry, rd = rd, accr=accr, eod_flag=eod_flag)
    p23 = CalSpdAsianOption(False, fwd2, fwd3, 0.0, vol2, vol3, corr, fixlist2, fixlist3, tday, expiry, rd = rd, accr=accr, eod_flag=eod_flag)
    print(p12, p13, p23)
