#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st


# Open data files
users = pd.read_csv('/datasets/megaline_users.csv')
plans = pd.read_csv('/datasets/megaline_plans.csv')
calls = pd.read_csv('/datasets/megaline_calls.csv')
messages = pd.read_csv('/datasets/megaline_messages.csv')
internet = pd.read_csv('/datasets/megaline_internet.csv')

# Study general info on data
#print(users.info()) #print(users.head())
#print(calls.info()) #print(calls.head())
#print(messages.info()) #print(messages.head())
#print(internet.info()) #print(internet.head())

# Preprocess tables

# Users table - convert string to datetime object
users['reg_date'] = pd.to_datetime(users['reg_date']).dt.date
users['churn_date'] = pd.to_datetime(users['churn_date']).dt.date

# Rename tariff column to plan for clarity
users = users.rename(columns={'tariff': 'plan'})

# Calls table - convert string to datetime object
calls['call_date'] = pd.to_datetime(calls['call_date']).dt.date

# Add month column for call_date
calls['call_month'] = pd.to_datetime(calls['call_date']).dt.month

# round call duration up to one minute
calls['duration'] = np.ceil(calls['duration'])

# Rename id column to call id for clarity
calls = calls.rename(columns={'id': 'call_id'})

# Messages table - convert string to datetime object
messages['message_date'] = pd.to_datetime(messages['message_date']).dt.date

# Add month column for message_date
messages['message_month'] = pd.to_datetime(messages['message_date']).dt.month

# Rename id column to message id for clarity
messages = messages.rename(columns={'id': 'message_id'})

# Internet table - convert string to datetime object
internet['session_date'] = pd.to_datetime(internet['session_date']).dt.date

# Add month column for session_date
internet['session_month'] = pd.to_datetime(internet['session_date']).dt.month

# round internet session up to one megabyte
internet['mb_used'] = np.ceil(internet['mb_used'])

# Rename id column to internet id for clarity
internet = internet.rename(columns={'id': 'internet_id'})

# Plans table - convert mb_per_month_included to gb_per_month_included for clarity
plans['mb_per_month_included'] = (plans['mb_per_month_included'] / 1024)
plans = plans.rename(columns={'mb_per_month_included': 'gb_per_month_included'})

#print(real_estate['total_images'].unique())
#print(real_estate['total_images'].value_counts())
#print(real_estate['total_images'].isna().sum())

# Additional preprocessing regarding serivce date vs churn data

# get the rows who have a non null churn date
userIdChurn = users[~users['churn_date'].isnull()]
userIdChurn = userIdChurn[['user_id','churn_date']]

# for every non-null churn date row
for index, row in userIdChurn.iterrows():
    user_id = row['user_id']
    churn_date = row['churn_date']
    
    calls.drop(calls.index[(calls['user_id'] == user_id) &(calls['call_date'] > churn_date)], inplace=True)
    messages.drop(messages.index[(messages['user_id'] == user_id) &(messages['message_date'] > churn_date)], inplace=True)
    internet.drop(internet.index[(internet['user_id'] == user_id) &(internet['session_date'] > churn_date)], inplace=True)

# For each user, find the number of calls made and minutes used per month
userCalls = pd.pivot_table(calls, values='duration', index=['user_id', 'call_month'], aggfunc=['count','sum'])
userCalls.columns = ['calls_made','minutes_used_per_month']
userCalls['minutes_used_per_month'] = np.ceil(userCalls['minutes_used_per_month'])
userCalls.reset_index(inplace=True)
display(userCalls)

# For each user, find the number of text messages sent per month
userMessages = pd.pivot_table(messages, values='message_id', index=['user_id', 'message_month'], aggfunc=['count'])
userMessages.columns = ['messages_sent_per_month']
userMessages.reset_index(inplace=True)
display(userMessages)

# For each user, find the volume of data per month
userInternet = pd.pivot_table(internet, values='mb_used', index=['user_id', 'session_month'], aggfunc=['sum'])
userInternet.columns = ['gb_used_per_month']
userInternet['gb_used_per_month'] = np.ceil(userInternet['gb_used_per_month'] / 1024)
userInternet.reset_index(inplace=True)
display(userInternet)

# Find the monthly profit from each user

# Join tables together for clarity
userUsage = pd.merge(userCalls, userMessages, left_on=['user_id','call_month'],right_on=['user_id','message_month'], how='outer')
userUsage = pd.merge(userUsage, userInternet, left_on=['user_id', 'call_month'], right_on=['user_id','session_month'], how='outer')
userPlans = users[['user_id', 'plan']]
userUsage = pd.merge(userUsage, userPlans, on='user_id')

def computeProfit(row):
    
    # extract user usage info
    user_id = row['user_id']
    gb_used = row['gb_used_per_month']
    messages_sent = row['messages_sent_per_month']
    minutes_used = row['minutes_used_per_month']
    
    # extract plan type
    plan = users.loc[users['user_id'] == user_id, 'plan'].iloc[0]
    plan_info = plans.loc[plans['plan_name'] == plan]
    
    # extract plan information
    usd_monthly_pay = plan_info['usd_monthly_pay'].iloc[0]
    minutes_included = plan_info['minutes_included'].iloc[0]
    usd_per_minute = plan_info['usd_per_minute'].iloc[0]
    messages_included = plan_info['messages_included'].iloc[0]
    usd_per_message =plan_info['usd_per_message'].iloc[0]
    gb_per_month_included = plan_info['gb_per_month_included'].iloc[0]
    usd_per_gb = plan_info['usd_per_gb'].iloc[0]
    
    # subtract the free package limit from the total number of calls
    remaining_minutes = minutes_used - minutes_included
    remaining_messages = messages_sent - messages_included
    remaining_data = gb_used - gb_per_month_included
    
    # multiply the result by the calling plan value 
    exceeding_data_cost = 0
    exceeding_message_cost = 0    
    exceeding_minute_cost = 0    
    if remaining_data > 0:
        exceeding_data_cost = remaining_data *  usd_per_gb   
    if remaining_messages > 0:
        exceeding_message_cost = remaining_messages *  usd_per_message
    if remaining_minutes > 0:
        exceeding_minute_cost = remaining_minutes * usd_per_minute  
        
    # add the monthly charge depending on the calling plan
    profit = exceeding_message_cost + exceeding_minute_cost + exceeding_data_cost + usd_monthly_pay

    return profit

userUsage['month_profit'] = userUsage.apply(computeProfit, axis=1)
display(userUsage)

# Find the minutes, texts, and volume of data the users of each plan require per month.

#Calculate the mean, dispersion, and standard deviation. 
surfUsage = userUsage.query('plan == "surf"')
ultimateUsage = userUsage.query('plan == "ultimate"')

surfMinutesMean = surfUsage['minutes_used_per_month'].mean()
surfMinutesVariance = surfUsage['minutes_used_per_month'].var()
surfMinutesStd = surfUsage['minutes_used_per_month'].std()
ultimateMinutesMean = ultimateUsage['minutes_used_per_month'].mean()
ultimateMinutesVariance = ultimateUsage['minutes_used_per_month'].var()
ultimateMinutesStd = ultimateUsage['minutes_used_per_month'].std()
print('Surf Minutes Used: Mean = %d mins, Variance = %d mins, Std = %d'% (surfMinutesMean, surfMinutesVariance, surfMinutesStd))
print('Ultimate Minutes Used: Mean = %d mins, Variance = %d mins, Std = %d'% (ultimateMinutesMean, ultimateMinutesVariance, ultimateMinutesStd))

surfMessagesMean = surfUsage['messages_sent_per_month'].mean()
surfMessagesVariance = surfUsage['messages_sent_per_month'].var()
surfMessagesStd = surfUsage['messages_sent_per_month'].std()
ultimateMessagesMean = ultimateUsage['messages_sent_per_month'].mean()
ultimateMessagesVariance = ultimateUsage['messages_sent_per_month'].var()
ultimateMessagesStd = ultimateUsage['messages_sent_per_month'].std()
print('Surf Messages Used: Mean = %d msgs, Variance = %d msgs, Std = %d' % (surfMessagesMean, surfMessagesVariance, surfMessagesStd))
print('Ultimate Messages Used: Mean = %d msgs, Variance = %d msgs, Std = %d' % (ultimateMessagesMean, ultimateMessagesVariance, ultimateMessagesStd))

surfDataMean = surfUsage['gb_used_per_month'].mean()
surfDataVariance = surfUsage['gb_used_per_month'].var()
surfDataStd = surfUsage['gb_used_per_month'].std()
ultimateDataMean = ultimateUsage['gb_used_per_month'].mean()
ultimateDataVariance = ultimateUsage['gb_used_per_month'].var()
ultimateDataStd = ultimateUsage['gb_used_per_month'].std()
print('Surf Data Used: Mean = %dGB, Variance = %d GB, Std = %d' % (surfDataMean, surfDataVariance, surfDataStd))
print('Ultimate Data Used: Mean = %d GB, Variance = %d GB, Std = %d' % (ultimateDataMean, ultimateDataVariance, ultimateDataStd))

# Plot histograms
kwargs = dict(histtype='stepfilled', alpha=0.5, bins=40)

figMins = plt.figure()
axMins = figMins.add_subplot(1,1,1)
axMins.hist(surfUsage['minutes_used_per_month'], label='surf', **kwargs)
axMins.hist(ultimateUsage['minutes_used_per_month'], label='ultimate', **kwargs)
figMins.suptitle('Surf vs Ultimate: Average Minutes Used')
axMins.set_xlabel("Avg Mins Used")
axMins.set_ylabel("Number of Users")
axMins.legend()

figMes = plt.figure()
axMes = figMes.add_subplot(1,1,1)
axMes.hist(surfUsage['messages_sent_per_month'], label='surf', **kwargs)
axMes.hist(ultimateUsage['messages_sent_per_month'], label='ultimate', **kwargs)
figMes.suptitle('Surf vs Ultimate: Average Messages Sent')
axMes.set_xlabel("Avg Mins Used")
axMes.set_ylabel("Number of Users")
axMes.legend()

figInt = plt.figure()
axInt = figInt.add_subplot(1,1,1)
axInt.hist(surfUsage['gb_used_per_month'], label='surf', **kwargs)
axInt.hist(ultimateUsage['gb_used_per_month'], label='ultimate', **kwargs)
figInt.suptitle('Surf vs Ultimate: Average GB Used')
axInt.set_xlabel("Avg Mins Used")
axInt.set_ylabel("Number of Users")
axInt.legend()

# not part of any steps

surfUsage = userUsage.query('plan == "surf"')
surfUsage = pd.pivot_table(surfUsage, values=['minutes_used_per_month','messages_sent_per_month','gb_used_per_month'], index=['call_month'], aggfunc=['mean'])
surfUsage.reset_index(inplace=True)
surfUsage.columns = ['month','surf_gb_used_per_month', 'surf_messages_sent_per_month','surf_minutes_used_per_month']
#display(surfUsage)

ultimateUsage = userUsage.query('plan == "ultimate"')
ultimateUsage = pd.pivot_table(ultimateUsage, values=['minutes_used_per_month','messages_sent_per_month','gb_used_per_month'], index=['call_month'], aggfunc=['mean'])
ultimateUsage.reset_index(inplace=True)
ultimateUsage.columns = ['month','ultimate_gb_used_per_month', 'ultimate_messages_sent_per_month','ultimate_minutes_used_per_month']
#display(ultimateUsage)

minutesData = pd.DataFrame({
    "surf":surfUsage['surf_minutes_used_per_month'],
    "ultimate":ultimateUsage['ultimate_minutes_used_per_month'],
    }, 
    index=[0,1,2,3,4,5,6,7,8,9,10,11]
)
minutesData.plot(kind="bar", title='Surf vs Ultimate: Minutes Used Per Month')
plt.xlabel("Month")
plt.ylabel("Minutes Used")

messagesData = pd.DataFrame({
    "surf":surfUsage['surf_messages_sent_per_month'],
    "ultimate":ultimateUsage['ultimate_messages_sent_per_month'],
    }, 
    index=[0,1,2,3,4,5,6,7,8,9,10,11]
)
messagesData.plot(kind="bar", title='Surf vs Ultimate: Messages Sent Per Month')
plt.xlabel("Month")
plt.ylabel("Messages Sent")

internetData = pd.DataFrame({
    "surf":surfUsage['surf_gb_used_per_month'],
    "ultimate":ultimateUsage['ultimate_gb_used_per_month'],
    }, 
    index=[0,1,2,3,4,5,6,7,8,9,10,11]
)
internetData.plot(kind="bar", title='Surf vs Ultimate: Gb Used Per Month')
plt.xlabel("Month")
plt.ylabel("Gb Used")

# Test the hypothesis: The average profit from users of Ultimate and Surf calling plans differs.

# filter outliers from monthly profit using zscore
z_scores = st.zscore(userUsage['month_profit'])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
userUsage = userUsage[filtered_entries]

# query surf users
surfUsers = userUsage.query('plan == "surf"')
surfAverageProfit = surfUsers['month_profit'].mean().round(decimals=2)
print('The average monthly profit made from surf users is: $' + str(surfAverageProfit))

# query ultimate users
ultimateUsers = userUsage.query('plan == "ultimate"')
ultimateAverageProfit = ultimateUsers['month_profit'].mean().round(decimals=2)
print('The average monthly profit made from ultimate users is: $' + str(ultimateAverageProfit))

# perform a t-test
results = st.ttest_ind(surfUsers['month_profit'], ultimateUsers['month_profit'], equal_var=False)
p_value = results.pvalue
alpha = 0.5

if p_value < alpha:
    print('Reject H0')
else:
    print('Cannot reject H0')

# Test the hypothesis: The average profit from users in NY-NJ area is different from that of the users from other regions.

# get the subset of users in NY-NJ area
NYNJ = users[users['city'].str.contains('NY-NJ')]
users_nynj = NYNJ['user_id']
userUsageNYNJ = userUsage.query('user_id in @users_nynj')
avgProfitNYNJ = userUsageNYNJ['month_profit'].mean().round(decimals=2)
print('The average monthly profit made users in NY-NJ area is: $' + str(avgProfitNYNJ))

# get the subset of users not in NY-NJ area
notNYNJ= users[~users['city'].str.contains('NY-NJ')]
users_notnynj = notNYNJ['user_id']
userUsageNotNYNJ = userUsage.query('user_id in @users_notnynj')
avgProfitNotNYNJ = userUsageNotNYNJ['month_profit'].mean().round(decimals=2)
print('The average monthly profit made users NOT in NY-NJ area is: $' + str(avgProfitNotNYNJ))

results = st.ttest_ind(userUsageNYNJ['month_profit'], userUsageNotNYNJ['month_profit'], equal_var=False)
p_value = results.pvalue
alpha = 0.5

if p_value < alpha:
    print('Reject H0')
else:
    print('Cannot reject H0')
