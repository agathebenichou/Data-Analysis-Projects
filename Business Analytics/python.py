#!/usr/bin/env python
# coding: utf-8

# The task at hand is to optimize marketing expenses for the Yandex.Afisha product.

# Open data files, study tables and optimize where possible
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Import visit logs data, converting category type and datetime types
visit_logs = pd.read_csv('/datasets/visits_log_us.csv', dtype={'Device':'category'},parse_dates=['Start Ts','End Ts'])
print(visit_logs.info())
print(visit_logs.describe())

# Import order logs data, converting datetime types
order_logs = pd.read_csv('/datasets/orders_log_us.csv',parse_dates=['Buy Ts'])
print(order_logs.info())
print(order_logs.describe())

# Import costs data, converting datetime types
cost = pd.read_csv('/datasets/costs_us.csv',parse_dates=['dt'])
print(cost.info())
print(cost.describe())

# Data Preprocessing

# Modify visit_logs column names
visit_logs = visit_logs.rename(columns={"Device": "device_type", "End Ts":"session_end", "Start Ts":"session_start", "Source Id":"source_id","Uid":"user_id"})

# Drop duplicates
visit_logs = visit_logs.drop_duplicates().reset_index(drop=True)

display(visit_logs.head())

# Modify order_logs column names
order_logs = order_logs.rename(columns={"Buy Ts":"order_datetime", "Revenue":"revenue","Uid":"user_id"})

# Drop duplicates
order_logs = order_logs.drop_duplicates().reset_index(drop=True)

display(order_logs.head())

# Modify cost column names
cost = cost.rename(columns={"dt":"ad_datetime","costs":"ad_expenses"})

# Drop duplicates
cost = cost.drop_duplicates().reset_index(drop=True)

display(cost.head())

# Modify and preprocess all tables by modifying column anmes to be more intuitive and drop any duplicates.
# Remove outliers

#  Drop rows where session_start = session_end 
outlierIndex = visit_logs[(visit_logs['session_start'] == visit_logs['session_end'])].index
visit_logs.drop(outlierIndex, inplace=True)
display(visit_logs)

# Drop outliers in order_logs DF where revenue is -2sigma <= and >= 2sigma 
ad_expenses_mean, ad_expenses_std = order_logs['revenue'].mean(), order_logs['revenue'].std()
cut_off = ad_expenses_std #*2
outlierIndex = order_logs[(order_logs['revenue'] <= ad_expenses_mean + cut_off) &  (ad_expenses_mean - cut_off <= order_logs['revenue'])].index
order_logs.drop(outlierIndex, inplace=True)
display(order_logs)

# Drop outliers in cost DF where the ad_expenses is -2sigma <= and >= 2sigma 
ad_expenses_mean, ad_expenses_std = cost['ad_expenses'].mean(), cost['ad_expenses'].std()
cut_off = ad_expenses_std #*2
outlierIndex = cost[(cost['ad_expenses'] <= ad_expenses_mean + cut_off) &  (ad_expenses_mean - cut_off <= cost['ad_expenses'])].index
cost.drop(outlierIndex, inplace=True)
display(cost)

# Remove any and all outliers where: session start datetime is the same as session end datetime, the revenue is out of the +/- 2 sigma range and the ad costs is out of the +/- 2 sigma range. 

# Product evaluation of Yandex.Afisha

# How many people use it every day, week and month?

# How many people use it every day, week and month?

# Create product usage from visit logs table
product_usage = visit_logs.copy()

# Separate columsn for day, week, month values
product_usage['session_date'] = product_usage['session_start'].dt.date
product_usage['session_week'] = product_usage['session_start'].dt.week
product_usage['session_month'] = product_usage['session_start'].dt.month
product_usage['session_year'] = product_usage['session_start'].dt.year

# Calculate DAU: number of daily active unique users
dau_total = product_usage.groupby('session_date').agg({"user_id":"nunique"}).reset_index()
dau_total.columns = ['session_date', 'n_users']
dau_avg = dau_total['n_users'].mean()
print("Daily Total Usage: " + str((dau_avg.round(decimals=2))) + " users")

# Calculate WAU: number of weekly active unique users
wau_total = product_usage.groupby('session_week').agg({"user_id":"nunique"}).reset_index()
wau_total.columns = ['session_week', 'n_users']
wau_avg = wau_total['n_users'].mean()
print("Weekly Total Usage: " + str((wau_avg.round(decimals=2))) + " users")

# Calculate DAU: number of monthly active unique users
mau_total = product_usage.groupby('session_month').agg({"user_id":"nunique"}).reset_index()
mau_total.columns = ['session_month', 'n_users']
mau_avg = mau_total['n_users'].mean()
print("Monthly Total Usage: " + str((mau_avg.round(decimals=2))) + " users")

# plot daily active users
plt.figure(figsize=(8,6))
plt.plot(dau_total['session_date'],dau_total['n_users'])
plt.title('Daily Number of Active Users')
plt.xlabel('Dates')
plt.ylabel('# of Active Users')

# plot weekly active users
plt.figure(figsize=(8,6))
plt.plot(wau_total['session_week'],wau_total['n_users'])
plt.title('Weekly Number of Active Users')
plt.xlabel('Dates')
plt.ylabel('# of Active Users')

# plot monthly active users
plt.figure(figsize=(8,6))
plt.plot(mau_total['session_month'],mau_total['n_users'])
plt.title('Monthly Number of Active Users')
plt.xlabel('Dates')
plt.ylabel('# of Active Users')

# How many sessions are there per day? 

# How many sessions are there per day? (One user might have more than one session.)

# group the visit data by session date and then count the # of unqiue users and # of unique sessions
sessions_per_day = product_usage.groupby('session_date').agg({'user_id':['count','nunique']}).reset_index()
sessions_per_day.columns = ['session_date','n_sessions','n_users']
sessions_per_day_amount = sessions_per_day['n_sessions'].mean()
print('There are ' + str(sessions_per_day_amount.round(decimals=2)) + " sessions per day")

#  Plot sessions per day
plt.figure(figsize=(8,6))
plt.plot(pd.to_datetime(sessions_per_day['session_date']),sessions_per_day['n_sessions'])
plt.title('Sessions per Day')
plt.xlabel('Dates')
plt.ylabel('# of Sessions')

# What is the length of each session?

# calculate the duration of a session(session_end - session_start)
product_usage['session_duration_sec'] = (product_usage['session_end'] - product_usage['session_start']).dt.seconds

# calculate the average duration
average_length_per_session = product_usage['session_duration_sec'].median()
print(" The average length per session is " + str(average_length_per_session.round(decimals=2)) + " seconds")

plt.hist(product_usage['session_duration_sec'],range=[0,2000],bins=50)
plt.title('Session Length')
plt.xlabel('Length (secs)')
plt.ylabel('# of Sessions')

#  How often do users come back?
# How often do users come back?

# calculate sticky factor metrics
sticky_weekly_avg = (dau_avg / wau_avg) * 100
print(str(sticky_weekly_avg.round(decimals=2)) + "% of users on average come back weekly")

sticky_monthly_avg = (dau_avg / mau_avg) * 100
print(str(sticky_monthly_avg.round(decimals=2)) + "% of users on average come back monthly")

# graph sticky factor metrics 
wau_total['sticky_weekly'] = (dau_avg / wau_total['n_users']) * 100
plt.figure(figsize=(8,6))
plt.plot(wau_total['session_week'],wau_total['sticky_weekly'])
plt.title('% of users that come back weekly')
plt.xlabel('Weeks')
plt.ylabel('% of returning users')

mau_total['sticky_monthly'] = (dau_avg / mau_total['n_users']) * 100
plt.figure(figsize=(8,6))
plt.plot(mau_total['session_month'],mau_total['sticky_monthly'])
plt.title('% of users that come back monthly')
plt.xlabel('Months')
plt.ylabel('% of returning users')

# Sales Evaluation for Yandex.Afisha

# When do people start buying?

# for each user, find date of first order
first_order_dates = order_logs.groupby('user_id').agg({'order_datetime':'min'}).reset_index()
first_order_dates.columns= ['user_id','first_order_date']
first_order_dates['first_order_month'] = first_order_dates['first_order_date'].dt.month

# for each user, find date of first session
first_session_dates = visit_logs.groupby('user_id').agg({'session_start':'min'}).reset_index()
first_session_dates.columns= ['user_id','first_session_date']
first_session_dates['first_session_month'] = first_session_dates['first_session_date'].dt.month

# merge tables on user_id
turnover = pd.merge(first_session_dates, first_order_dates, on='user_id')

# calculate time between first session and first order
turnover['turnover_time_days'] = (turnover['first_order_date'] - turnover['first_session_date']).dt.days
display(turnover)

avg_turnover_time = turnover['turnover_time_days'].mean()
print('People start buying ' + str(avg_turnover_time.round(decimals=2)) +" days after first registering")

# categorize cohorts by turnover time 
bins = [-2, 5, 10, 20, 30, 90, 180, 400]
labels = ['0-5d','5-10d','10-20d','20-30d','30-90d','90-180d','>180d']
turnover['turnover_cohort'] = pd.cut(x=turnover['turnover_time_days'], bins=bins, labels=labels)

# how many users in each cohort
users_per_cohort = turnover.groupby('turnover_cohort')['user_id'].nunique().reset_index()
users_per_cohort.columns = ['turnover_cohort','n_users']

# calculate % of users in each cohort
total_users = users_per_cohort['n_users'].sum()
users_per_cohort['users %'] = (users_per_cohort['n_users'] / total_users) * 100

display(users_per_cohort)

# How many orders do they make during a given period of time?

# How many orders do they make during a given period of time? 

# extract order month for each order
order_logs['order_month'] = order_logs['order_datetime'].astype('datetime64[M]').dt.month

# group by order month and find # of unique users placing orders each month and find # of orders made each month
unique_orders = order_logs.groupby('order_month').agg({'user_id':'nunique','order_datetime':'count'}).reset_index()
unique_orders.columns = ['order_month','n_unique_users','n_orders']

# create col with average monthly orders per month
unique_orders['average_monthly_orders'] = unique_orders['n_orders'] / unique_orders['n_unique_users']

# calculate average monthly purchases
avg_monthly_purchases = unique_orders['average_monthly_orders'].mean().round(decimals=2)

print('The average number of orders made in any given month is ' + str(avg_monthly_purchases))

# What is the average purchase size?

# What is the average purchase size?

avg_purchase_size = order_logs['revenue'].mean().round(decimals=2)
print("The average purchase size is " + str(avg_purchase_size) + " dollars per purchase")

# How much money do they bring? (LTV)
# LTV (lifetime value) is the total amount of money a customer brings to the company on average by making purchases

# # copy dataframes
orders = order_logs.copy()
visit = visit_logs.copy()
visit = visit[['session_start','user_id']]

# extract month and date for each datetime row 
orders['order_month'] = orders['order_datetime'].astype('datetime64[M]')
orders['order_date'] = orders['order_datetime'].dt.date

# Calculate when first order for each customer have happened

visits_and_orders = orders.join(visit
                      .sort_values(by='session_start')
                      .groupby(['user_id'])
                      .agg({'session_start': 'min'}),
                      on='user_id', how='inner')

visits_and_orders['first_session_month'] = visits_and_orders['session_start'].astype('datetime64[M]')

# Create cohorts based on first purchase date and revenue

cohort_sizes = visits_and_orders.groupby('first_session_month').agg({'user_id': 'nunique'}).reset_index()
cohort_sizes.rename(columns={'user_id': 'n_buyers'}, inplace=True)

cohorts = visits_and_orders.groupby(['first_session_month', 'order_month']).agg({'revenue': ['sum', 'count']}).reset_index()

# Calculate cohort age

cohorts['age_month'] = (cohorts['order_month'] - cohorts['first_session_month']) / np.timedelta64(1, 'M')
cohorts['age_month'] = cohorts['age_month'].round().astype('int')
cohorts.columns = ['first_session_month', 'order_month', 'revenue', 'n_orders', 'age_month']

# Merge our cohort tables to the final cohort report
report = pd.merge(cohort_sizes, cohorts, on='first_session_month')
report['ltv'] = cohort_report['revenue'] / cohort_report['n_buyers']

# Create LTV table
ltv_cohort = report.pivot_table(
    index = 'first_session_month',
    columns = 'age_month',
    values = 'ltv',
    aggfunc = 'sum').cumsum(axis=1)

plt.figure(figsize=(15, 5))
sns.heatmap(ltv_cohort, annot=True, fmt='.2f');

#  How much money was spent? Overall/per source/over time

# Overall
cost_overall = cost['ad_expenses'].sum()
print('The overall amount spent on marketing was $' + str(cost_overall))

# per source
cost_per_source = cost.groupby('source_id').agg({'ad_expenses':'sum'}).reset_index()
cost_per_source.columns = ['source_id','total_ad_expenses']

plt.figure(figsize=(8,6))
plt.bar(cost_per_source['source_id'],cost_per_source['total_ad_expenses'])
plt.title('Ad Costs Per Source')
plt.xlabel('Source')
plt.ylabel('Amount Spent ($)')

# Over time
cost_over_time = cost.copy()
cost_over_time['ad_month'] = cost['ad_datetime'].astype('datetime64[M]').dt.month
cost_over_time = cost_over_time.groupby('ad_month').agg({'ad_expenses':'sum'}).reset_index()
cost_over_time.columns = ['ad_month','ad_expenses']

plt.figure(figsize=(8,6))
plt.plot(cost_over_time['ad_month'],cost_over_time['ad_expenses'])
plt.title('Ad Costs Over Time')
plt.xlabel('Months')
plt.ylabel('Amount Spent ($)')

# How much did customer acquisition from each of the sources cost?

# CAC
# calculate ad expenses per month
monthly_ad_cost = cost.copy()
monthly_ad_cost['ad_month'] = monthly_ad_cost['ad_datetime'].dt.month
monthly_ad_cost = monthly_ad_cost[['ad_month','ad_expenses']]
monthly_ad_cost = monthly_ad_cost.groupby('ad_month')['ad_expenses'].sum().reset_index()

# incorpordate data on costs
report['order_month'] = report['order_month'].dt.month

report = pd.merge(report, monthly_ad_cost, left_on='order_month', right_on='ad_month')

# calculate cac
report['cac'] = report['ad_expenses'] / report['n_buyers']

# starting at first order month and continuing through order months,what is cac per cohort
result = report.pivot_table(index='order_month',columns='ad_month',values='cac',aggfunc='mean').round()
result.fillna('')

plt.figure(figsize=(13,9))
plt.title('CAC per Cohort')
sns.heatmap(result, annot=True, fmt='.1f', linewidths=1, linecolor='black')

# How worthwhile where the investments? (ROI)

# calculate return on marketing investment 
report['romi'] = report['ltv'] / report['cac']

output = report.pivot_table(index='order_month', columns='ad_month',values='romi',aggfunc='mean')
output = output.cumsum(axis=1).round(2)

plt.figure(figsize=(13,9))
plt.title('ROI per Cohort')
sns.heatmap(output, annot=True, fmt='.1f', linewidths=1, linecolor='black')
