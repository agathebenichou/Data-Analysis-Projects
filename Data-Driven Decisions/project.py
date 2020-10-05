#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Prioritizing Hypotheses

# ### Load data

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

# load in data file 
hypotheses = pd.read_csv('/datasets/hypotheses_us.csv', sep=';')

# study general information 
hypotheses.info()

display(hypotheses)

# ### Apply the ICE framework to prioritize hypotheses
# for each row, apply ICE formula = (impact * confidence) / effort
hypotheses['ICE'] = hypotheses.apply(lambda x: (x['Impact'] * x['Confidence']) / x['Effort'], axis=1)

# sort in descending order
ice_results = hypotheses.sort_values(by='ICE', ascending=False)

display(ice_results)

# ### Apply the RICE framework to prioritize hypotheses
# for each row, apply RICE formula = (impact * confidence * reach) / effort
hypotheses['RICE'] = hypotheses.apply(lambda x: (x['Impact'] * x['Confidence'] * x['Reach']) / x['Effort'], axis=1)

# sort in descending order
rice_results = hypotheses.sort_values(by='RICE', ascending=False)

display(rice_results)

# ### Show how the prioritization of hypotheses changes when you use RICE instead of ICE
# plot the difference
ax = hypotheses[['Hypothesis','ICE','RICE']].plot(kind='bar',stacked=False, figsize=(8,6))

for p in ax.patches:
    ax.annotate(str(p.get_height().round()), (p.get_x() * 1.005, p.get_height() * 1.005), rotation=90)
    
plt.title('ICE vs RICE Prioritization')
plt.xlabel('Hypotheses')
plt.ylabel('ICE/RICE Score')


# ## Part 1: Prioritizing Hypotheses
# load in data file 
orders = pd.read_csv('/datasets/orders_us.csv', sep=',')
orders['date'] = pd.to_datetime(orders['date'])

# study general information 
orders.info()
display(orders.head())

# load in data file 
visits = pd.read_csv('/datasets/visits_us.csv', sep=',')
visits['date'] = pd.to_datetime(visits['date'])

# study general information 
visits.info()
display(visits.head())

# ### Data Preprocessing
# drop duplicates
orders = orders.drop_duplicates()
visits = visits.drop_duplicates()

# drop any missing values
orders = orders.dropna()
visits = visits.dropna()


# ### Graph cumulative revenue by group.
# organize data into groups by date
dateGroups = orders[['date','group']].copy()

# drop any duplicates
dateGroups = dateGroups.drop_duplicates()

# DF with unique paired date and group values from orders DF
ordersAgg = dateGroups.apply(
    lambda x: orders[np.logical_and(orders['date'] <= x['date'], orders['group'] == x['group'])]
    .agg({'date' : 'max', 'group' : 'max', 'transactionId' : pd.Series.nunique, 'visitorId' : pd.Series.nunique, 
          'revenue' : 'sum'}), axis=1).sort_values(by=['date','group'])
#display(ordersAgg)

# DF with unique paired date and group values from visits DF
visitsAgg = dateGroups.apply(
    lambda x: visits[np.logical_and(visits['date'] <= x['date'], visits['group'] == x['group'])]
    .agg({'date' : 'max', 'group' : 'max', 'visits' : 'sum'}), axis=1).sort_values(by=['date','group'])
#display(visitsAgg)

#  merge two tables into one 
cumulativeData = ordersAgg.merge(visitsAgg, left_on=['date', 'group'],  right_on=['date','group'])
cumulativeData.columns = ['date','group','orders','buyers','revenue','visitors']
display(cumulativeData)

# cumulative orders and cumulative revenue by day for groups A/B
cumRevenueA = cumulativeData[cumulativeData['group'] == 'A'][['date','revenue','orders']]
cumRevenueB = cumulativeData[cumulativeData['group'] == 'B'][['date','revenue','orders']]

# plot cumulative revenue by group
plt.plot(cumRevenueA['date'], cumRevenueA['revenue'], label='A')
plt.plot(cumRevenueB['date'], cumRevenueB['revenue'], label='B')
plt.legend()
plt.title('Cumulative Revenue by Group over Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Revenue')
plt.xticks(rotation=90)


# ### Graph cumulative average order size by group.
# diving the revenue by the cumulative number of orders
plt.plot(cumRevenueA['date'], cumRevenueA['revenue']/cumRevenueA['orders'], label='A')
plt.plot(cumRevenueB['date'], cumRevenueB['revenue']/cumRevenueB['orders'], label='B')

plt.axhline(y=100, color='black', linestyle='--')
plt.legend()
plt.title('Average Purchase Size by Group over Time')
plt.xlabel('Time')
plt.ylabel('Average Number of Orders')
plt.xticks(rotation=90)

# ### Graph the relative difference in cumulative average order size for group B compared with group A. 
# gather data into one dataframe with group suffixes
mergedCumRevenue = cumRevenueA.merge(cumRevenueB, left_on='date', right_on='date', how='left', suffixes=['A','B'])

plt.plot(mergedCumRevenue['date'], mergedCumRevenue['revenueA'] / mergedCumRevenue['ordersA'] -1, label='A')
plt.plot(mergedCumRevenue['date'], mergedCumRevenue['revenueB'] / mergedCumRevenue['ordersB'] -1, label='B')

plt.axhline(y=100, color='black', linestyle='--')
plt.legend()
plt.title('Relative Difference in Cumulative Average Order Size per Groups')
plt.xlabel('Time')
plt.ylabel('Average Order Size')
plt.xticks(rotation=90)

# ### Calculate each group's conversion rate as the ratio of orders to the number of visits for each day. 
# calculate and store conversion rate
cumulativeData['conversion'] = cumulativeData['orders'] / cumulativeData['visitors']

# select group data
cumDataA = cumulativeData[cumulativeData['group'] == 'A']
cumDataB= cumulativeData[cumulativeData['group'] == 'B']

mergedCumConversion = cumDataA[['date','conversion','visitors']].merge(cumDataB[['date','conversion','visitors']], left_on='date', right_on='date', how='left', suffixes=['A', 'B'])

plt.plot(mergedCumConversion['date'], mergedCumConversion['visitorsA']/mergedCumConversion['conversionA'], label='A')
plt.plot(mergedCumConversion['date'], mergedCumConversion['visitorsB']/mergedCumConversion['conversionB'], label='B')

plt.legend()
plt.title('Conversion Rate')
plt.xlabel('Time')
plt.ylabel('Relativen Gain')
plt.xticks(rotation=90)

# ### Plot a scatter chart of the number of orders per user. 
ordersByUsers = orders.copy()

# drop unneccessary columns and group the orders by users
ordersByUsers= ordersByUsers.drop(['group','revenue','date'], axis=1).groupby('visitorId', as_index=False).agg({'transactionId':pd.Series.nunique})
ordersByUsers.columns = ['userId', 'orders']

# sort data by number of orders in descending orders
ordersByUsers = ordersByUsers.sort_values(by='orders', ascending=False)

# find values for horizontal axis by the number of generated observations 
x_values = pd.Series(range(0, len(ordersByUsers)))

plt.scatter(x_values, ordersByUsers['orders'])
plt.title('Raw Data: Number of Orders per User')
plt.xlabel('Number of Generated Observations')
plt.ylabel('Number of Orders')

# ### Calculate the 95th and 99th percentiles for the number of orders per user. 
print(np.percentile(ordersByUsers['orders'], [95,99]))

# ### Plot a scatter chart of order prices.
ordersByPrices = orders.copy()

# drop unneccessary columns and group the orders by users
ordersByPrices= ordersByPrices.drop(['group','transactionId','date'], axis=1).groupby('visitorId', as_index=False).agg({'revenue':'sum'})
ordersByPrices.columns = ['userId', 'revenue']

# sort data by number of orders in descending orders
ordersByPrices = ordersByPrices.sort_values(by='revenue', ascending=False)

# find values for horizontal axis by the number of generated observations 
x_values = pd.Series(range(0, len(ordersByPrices)))

plt.scatter(x_values, ordersByPrices['revenue'])

plt.title('Raw Data: Revenue per Order')
plt.xlabel('Number of Generated Observations')
plt.ylabel('Revenue per Order')

# ### Calculate the 95th and 99th percentiles of order prices. 
print(np.percentile(ordersByPrices['revenue'], [95,99]))

# ### Find the statistical significance of the difference in conversion between the groups using the raw data. 
# calculate statistical significance of difference in conversion between groups 
ordersByUsersA = orders[orders['group'] == 'A'].groupby('visitorId', as_index=False).agg({'transactionId': pd.Series.nunique})
ordersByUsersA.columns = ['userId', 'orders']

ordersByUsersB = orders[orders['group'] == 'B'].groupby('visitorId', as_index=False).agg({'transactionId': pd.Series.nunique})
ordersByUsersB.columns = ['userId', 'orders']

# delcare vars with users from different groups and the number of users / group
sampleA = pd.concat([ordersByUsersA['orders'], pd.Series(0, index = np.arange(visits[visits['group'] == 'A']['visits'].sum() - len(ordersByUsersA['orders'])), name='orders')], axis=0)

sampleB = pd.concat([ordersByUsersB['orders'], pd.Series(0, index = np.arange(visits[visits['group'] == 'B']['visits'].sum() - len(ordersByUsersA['orders'])), name='orders')], axis=0)

p_value = st.mannwhitneyu(sampleA, sampleB)[1]
print(p_value)

alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')

# ### Find the statistical significance of the difference in average order size between the groups using the raw data. 
p_value = st.mannwhitneyu(orders[orders['group'] == 'A']['revenue'], orders[orders['group'] == 'B']['revenue'])[1]
print("{0:.3f}".format(p_value))
alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')

# ### Find the statistical significance of the difference in conversion between the groups using the filtered data. 

#identify anomalous users with tooo many orders
usersWithManyOrders = pd.concat([ordersByUsersA[ordersByUsersA['orders'] > 2]['userId'], ordersByUsersB[ordersByUsersB['orders'] > 2]['userId']], axis = 0)

# identify anomalous users with expensive orders
usersWithExpensiveOrders = orders[orders['revenue'] > 500]['visitorId']

# join them into abnormal table and remove dupliated
abnormalUsers = pd.concat([usersWithManyOrders, usersWithExpensiveOrders], axis = 0).drop_duplicates().sort_values()

# calculate statistical signifiacne of the difference in conversion between groups using filtered data
sampleAFiltered = pd.concat([ordersByUsersA[np.logical_not(ordersByUsersA['userId'].isin(abnormalUsers))]['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='A']['visits'].sum() - len(ordersByUsersA['orders'])),name='orders')],axis=0)

sampleBFiltered = pd.concat([ordersByUsersB[np.logical_not(ordersByUsersB['userId'].isin(abnormalUsers))]['orders'],pd.Series(0, index=np.arange(visits[visits['group']=='B']['visits'].sum() - len(ordersByUsersB['orders'])),name='orders')],axis=0)

p_value = st.mannwhitneyu(sampleAFiltered, sampleBFiltered)[1]
print("{0:.5f}".format(p_value))

alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')

# ### Find the statistical significance of the difference in average order size between the groups using the filtered data.
p_value = st.mannwhitneyu(
    orders[np.logical_and(
        orders['group']=='A',
        np.logical_not(orders['visitorId'].isin(abnormalUsers)))]['revenue'],
    orders[np.logical_and(
        orders['group']=='B',
        np.logical_not(orders['visitorId'].isin(abnormalUsers)))]['revenue'])[1]
print("{0:.3f}".format(p_value))

alpha = 0.05

if p_value < alpha:
    print('H0 rejected')
else:
    print('Failed to reject H0')