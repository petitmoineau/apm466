#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.dates as dates


# In[2]:


CA135087L773 = {'Coupon': 0.25, 'Maturity': datetime.date(2023, 2, 1), 'Last Coupon': datetime.date(2023, 1, 31), 
                'Next Coupon': datetime.date(2023, 2, 1), 
                'Closing Price': [99.84, 99.86, 99.87, 99.90, 99.91, 99.92, 99.93, 99.94, 99.98, 99.99]}
CA135087M359 = {'Coupon': 0.25, 'Maturity': datetime.date(2023, 8, 1), 'Last Coupon': datetime.date(2023, 7, 31), 
                'Next Coupon': datetime.date(2023, 2, 1),
                'Closing Price': [97.80, 97.80, 97.83, 97.84, 97.83, 97.84, 97.87, 97.90, 97.91, 97.90]}
CA135087M920 = {'Coupon': 0.75, 'Maturity': datetime.date(2024, 2, 1), 'Last Coupon': datetime.date(2024, 1, 31), 
                'Next Coupon': datetime.date(2023, 2, 1), 
                'Closing Price': [96.46, 96.44, 96.47, 96.46, 96.38, 96.40, 96.41, 96.47, 96.46, 96.40]}       
CA135087N910 = {'Coupon': 2.75, 'Maturity': datetime.date(2024, 8, 1), 'Last Coupon': datetime.date(2024, 7, 31), 
                'Next Coupon': datetime.date(2023, 2, 1),
                'Closing Price': [98.34, 98.34, 98.44, 98.39, 98.30, 98.26, 98.24, 98.32, 98.31, 98.18]}
CA135087P659 = {'Coupon': 3.75, 'Maturity': datetime.date(2025, 2, 1), 'Last Coupon': datetime.date(2025, 1, 31), 
                'Next Coupon': datetime.date(2023, 2, 1),
                'Closing Price': [100.36, 100.38, 100.53, 100.46, 100.30, 100.24, 100.18, 100.28, 100.28, 100.12]}
CA135087K940 = {'Coupon': 0.5, 'Maturity': datetime.date(2025, 9, 1), 'Last Coupon': datetime.date(2025, 8, 31), 
                'Next Coupon': datetime.date(2023, 3, 1),
                'Closing Price': [92.74, 92.75, 93.03, 93.03, 92.87, 92.82, 92.81, 92.97, 93.00, 92.72]}
CA135087L518 = {'Coupon': 0.25, 'Maturity': datetime.date(2026, 3, 1), 'Last Coupon': datetime.date(2026, 2, 28), 
                'Next Coupon': datetime.date(2023, 3, 1),
                'Closing Price': [91.08, 91.19, 91.43, 91.51, 91.32, 91.23, 91.21, 91.41, 91.45, 91.17]}
CA135087L930 = {'Coupon': 1.0, 'Maturity': datetime.date(2026, 9, 1), 'Last Coupon': datetime.date(2026, 8, 31), 
                'Next Coupon': datetime.date(2023, 3, 1),
                'Closing Price': [92.72, 92.84, 93.24, 93.29, 92.95, 92.90, 92.91, 93.05, 93.04, 92.77]}
CA135087M847 = {'Coupon': 1.25, 'Maturity': datetime.date(2027, 3, 1), 'Last Coupon': datetime.date(2027, 2, 28), 
                'Next Coupon': datetime.date(2022, 3, 1),
                'Closing Price': [93.14, 93.25, 93.73, 93.78, 93.39, 93.33, 93.35, 93.50, 93.46, 93.14]}
CA135087N837 = {'Coupon': 2.75, 'Maturity': datetime.date(2028, 9, 1), 'Last Coupon': datetime.date(2027, 8, 31), 
                'Next Coupon': datetime.date(2023, 3, 1),
                'Closing Price': [99.10, 99.22, 99.71, 99.71, 99.24, 99.14, 99.15, 99.29, 99.24, 98.86]}
CA135087P576 = {'Coupon': 3.5, 'Maturity': datetime.date(2028, 3, 1), 'Last Coupon': datetime.date(2028, 2, 28), 
                'Next Coupon': datetime.date(2023, 3, 1),
                'Closing Price': [102.73, 102.84, 103.40, 103.38, 102.83, 102.73, 102.77, 102.92, 102.82, 102.40]}


# In[45]:


bonds = [CA135087L773, CA135087M359, CA135087M920, CA135087N910,
         CA135087P659, CA135087K940, CA135087L518, CA135087L930,
         CA135087M847, CA135087N837,CA135087P576]


# In[71]:


def diff(d1, d2):
    return (d1-d2).days

def ai(daysPassed, cr):
    return (daysPassed / 365) * cr

def dp(ai, ap):
    return ai + ap

def pv(cr, YTM, currentDate, maturity, nextCouponDate):
    pv = 0
    t_i = diff(currentDate, nextCouponDate) / 365
    periods = round(diff(nextCouponDate, maturity) / 365 * 2)
    for _ in range(0, periods):
        pv += (cr / 2) * np.exp(-YTM * t_i)
        t_i += 0.5
    pv += (100 + cr / 2) * np.exp(-YTM * t_i)
    return pv


# In[72]:


watchDays = {0: datetime.date(2023, 1, 16), 
             1: datetime.date(2023, 1, 17), 
             2: datetime.date(2023, 1, 18), 
             3: datetime.date(2023, 1, 19), 
             4: datetime.date(2023, 1, 20), 
             5: datetime.date(2023, 1, 23), 
             6: datetime.date(2023, 1, 24), 
             7: datetime.date(2023, 1, 25), 
             8: datetime.date(2023, 1, 26), 
             9: datetime.date(2023, 1, 27)}


# In[76]:


def findYTM(YTM, index, bond):
    cr = bond['Coupon']
    maturity = bond['Maturity']
    cp = bond['Closing Price'][index]
    lastCDate = bond['Last Coupon']
    nextCDate = bond['Next Coupon']

    curDate = watchDays[index]
    daysSinceLastCPaym = diff(lastCDate, curDate)
    prV = pv(cr, YTM, watchDays[index], maturity, nextCDate)
    accI = ai(daysSinceLastCPaym, cr)
    dirP = dp(accI, cp)

    return prV - dirP


# In[77]:


def getYTMS(bonds):
    YTMList = np.zeros((10, len(bonds)))
    for day in range(10):
        for bond_idx, bond in enumerate(bonds):
            ytm = optimize.newton(findYTM, bond['Coupon'] / 100, args=(day, bond))
            YTMList[day, bond_idx] = ytm
    return YTMList


# In[78]:


YTMList = getYTMS(bonds)
print(YTMList)


# In[79]:


YTM2324 = YTMList[:, 0:3] 
print(YTM2324.ndim)
DATES2324 = dates.date2num([datetime.date(2023, 2, 1), datetime.date(2023, 8, 1), datetime.date(2024, 2, 1)]) 
print(DATES2324.ndim)

for day in range(10):
    spl = make_interp_spline(DATES2324,YTM2324[day, :], k=2)
    
YTM2425 = ytm_lst[:, 2:5]
DATES2425 = dates.date2num([datetime.date(2024, 3, 1), datetime.date(2024, 6, 1), datetime.date(2025, 3, 1)])

for day in range(10):
    spl = make_interp_spline(DATES2425, YTM2425[day, :], k=2)


# In[81]:


maturityDates = ['2/1/2023', '8/1/2023', '2/1/2024', '8/1/2024', '2/1/2025',
                  '9/1/2025', '3/1/2026', '9/1/2026', '3/1/2027', '9/1/2027', '3/1/2028']
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'turquoise', 'fuchsia', 'violet', 'pink', 'black']
plt.figure(figsize=(15, 8))
for day in range(10):
    plt.plot(maturityDates, ytm_lst[day, :], colors[day])
plt.legend(['Jan 16', 'Jan 17','Jan 18','Jan 19','Jan 20','Jan 23','Jan 24','Jan 25','Jan 26','Jan 27'])
plt.xlabel('Maturity Dates')
plt.ylabel('YTM')
plt.title('YTM Curve')


# In[88]:


def spot(bonds, index):
    spots = []
    curDate = watchDays[index]

    for i, bond in enumerate(bonds):
        cr = bond['Coupon']
        maturity = bond['Maturity']
        cp = bond['Closing Price'][index]
        lastCDate = bond['Last Coupon']

        daysPassedSinceLastC = diff(lastCDate, curDate)
        accI = ai(daysPassedSinceLastC, cr)
        dirP = dp(accI, cp)
        
        ttm = diff(curDate, maturity) / 365
        if i == 0:
            spotRate = -np.log(dirP / (100 + cr / 2)) / ttm
            spots.append(spotRate)
        else:
            pv = 0
            for j in range(i):
                maturity_date_j = bonds[j]['Maturity']
                t_j = diff(curDate, maturity_date_j) / 365 
                pv += (cr / 2) * np.exp(-spots[j] * t_j)
            spotRate = -np.log((dirP - pv) / (100 + cr / 2)) / ttm 
            spots.append(spotRate) 
    return np.array(spots)


# In[92]:


def spots(bonds):  
    spotRates = []
    for day in range(10):
        spotRates.append(spot(bonds, day))
    return np.array(spotRates)

spotRates = spots(bonds)


# In[93]:


colors = ['red', 'orange', 'yellow', 'green', 'blue', 'turquoise', 'fuchsia', 'violet', 'pink', 'black']
time = ['less 0.5 year', '0.5-1 year', '1-1.5 year', '1.5-2 year',
        '2-2.5 year', '2.5-3 year', '3-3.5 year', '3.5-4 year', '4-4.5 year', '4.5-5 year', 'more than 5 year']
print(len(time))
print(len(spotRates))
plt.figure(figsize=(20, 10))
for i in range(len(spotRates)):
    plt.plot(time, spotRates[i], colors[i])
plt.legend(['Jan 16', 'Jan 17', 'Jan 18', 'Jan 19', 'Jan 20', 'Jan 23', 'Jan 24', 'Jan 25', 'Jan 26', 'Jan 27'])
plt.xlabel('Time')
plt.ylabel('Spot Rates')
plt.title('Spot Curve')


# In[22]:


def get_forward_rates_daily(spot_rates, current_date_idx):
    forward_rates = []
    year_1_spot_rate = spot_rates[current_date_idx][2]
    for j in range(4, 11, 2):
        year = j / 2 
        forward_rate = ((spot_rates[current_date_idx][j] * year) - (year_1_spot_rate * 1)) / (year - 1)
        forward_rates.append(forward_rate)
    return forward_rates

def get_all_forward_rates(spot_rates):
    all_forward_rates = []
    for i in range(10):
        daily_forward_rates = get_forward_rates_daily(spot_rates, i)
        all_forward_rates.append(daily_forward_rates)
    return np.array(all_forward_rates)



# In[23]:


all_forward_rates = get_all_forward_rates(all_spot_rates)


# In[24]:


all_forward_rates = get_all_forward_rates(all_spot_rates)
year = ['1y-1y', '1y-2y', '1y-3y', '1y-4y']
plt.figure(figsize=(17, 8))
for day in range(10):
    plt.plot(year, all_forward_rates[day], colours[day])
plt.legend(['Jan 16', 'Jan 17', 'Jan 18', 'Jan 19', 'Jan 20',
            'Jan 23', 'Jan 24', 'Jan 25', 'Jan 26', 'Jan 27'])
plt.xlabel('Year')
plt.ylabel('Forward Rates')
plt.title('Forward Curve')


# In[25]:


def build_yield_X(ytm_lst):
    X = np.zeros((5, 9))
    for i in range(5):
        for j in range(9):
            X[i][j] = np.log(ytm_lst[i][j+1] / ytm_lst[i][j])
    return X

def build_forward_X(forward_rates):
    X = np.zeros((4, 9))
    for i in range(4):
        for j in range(9):
            X[i][j] = np.log(forward_rates[j+1][i] / forward_rates[j][i])
    return X


# In[26]:


ytm_1_to_5 = np.array(
    [ytm_lst[:, 2], ytm_lst[:, 4], ytm_lst[:, 6], ytm_lst[:, 8], ytm_lst[:, 10]])

X_ytm = build_yield_X(ytm_1_to_5)

X_forward = build_forward_X(all_forward_rates)

cov_ytm = np.cov(X_ytm)
cov_forward = np.cov(X_forward)


# In[35]:


eig_ytm = np.linalg.eig(cov_ytm)
eigenvalue_ytm =  eig_ytm[0]
eigenvector_ytm = eig_ytm[1]

eig_forward = np.linalg.eig(cov_forward)
eigenvalue_forward = eig_forward[0]
eigenvector_forward = eig_forward[1]

