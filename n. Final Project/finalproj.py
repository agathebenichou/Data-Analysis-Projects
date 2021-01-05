#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import datetime 
import matplotlib.dates
from scipy import stats as st

# Open data files
data = pd.read_csv('/datasets/telecom_dataset_us.csv')

# Study general information
data.info()
display(data.head())

# Open data files
clients = pd.read_csv('/datasets/telecom_clients_us.csv')

# Study general information
clients.info()
display(clients.head())

# Drop any duplicates in the database
clients.drop_duplicates(inplace=True)

# convert dates to datetime objects, only date should remain
clients['date_start'] = pd.to_datetime(clients['date_start'], format="%Y-%m-%d").dt.date

# drop null values for internal and operator_id
data.dropna(subset=['internal', 'operator_id'], inplace=True)

# Drop any duplicates in the database
data.drop_duplicates(inplace=True)

# convert dates to datetime objects, only date should remain
data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d").dt.date

# convert operator_id from float to int
data['operator_id'] = data['operator_id'].astype(int)

# rename columns to include units
data = data.rename(columns={'call_duration': 'call_duration_sec', 'total_call_duration': 'total_call_duration_sec'})

# plot pie chart with proportions
plt.figure(figsize=(10, 5))

internal_external = data.groupby('internal')['operator_id'].count().reset_index()
internal_external.columns = ['internal', 'count']

labels = ['False', 'True']
plt.pie(internal_external['count'], labels=labels, autopct='%0.f%%', shadow=True, startangle=145)
plt.title('Shares of Internal')
plt.show()

# group rows into call duration categories
def determineCallDurationCategory(row):
    call_duration = row['call_duration_sec']
    
    # 0 to 1 min
    if call_duration <= 60:
        return 'less than 1 min'
    
    # 1 to 15 min
    if 60 < call_duration <= 900:
        return 'between 1 to 15 mins'
    
    # 15 to 30 min
    if 900 < call_duration <= 1800:
        return 'between 15 to 30 mins'
    
    # 30 to 60 min
    if 1800 < call_duration <= 3600:
        return 'between 30 to 60 mins'
    
    # 60+ mins
    if call_duration > 3600:
        return 'more than 1 hour'

# apply result from method to new category
data['call_length_category'] = data.apply(determineCallDurationCategory, axis=1)

# group each category
call_category = data.groupby('call_length_category')['operator_id'].count().reset_index()
call_category.columns = ['call_length_category', 'count']

# plot pie chart with proportions
plt.figure(figsize=(10, 5))

labels = ['less than 1 min', 'between 1 to 15 mins', 'between 15 to 30 mins', 'between 30 to 60 mins', 'more than 1 hour']
plt.pie(call_category['count'], labels=labels,autopct='%0.f%%', shadow=True, startangle=145)
plt.title('Distribution of Call Length')
plt.show()

# plot it
fig = plt.figure(figsize=(10,7))

plt.hist(data['date'], bins=50, color='lightblue')
plt.title('Calls Per Day')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.show()

# plot pie chart with proportions
plt.figure(figsize=(10, 5))

tariff = clients.groupby('tariff_plan')['user_id'].count().reset_index()
tariff.columns = ['tariff_plan', 'count']

labels = ['A', 'B', 'C']
plt.pie(tariff['count'], labels=labels, autopct='%0.f%%', shadow=True, startangle=145)
plt.title('Tariff Plans')
plt.show()

# plot it
fig = plt.figure(figsize=(10,7))

plt.hist(clients['date_start'], bins=50, color='lightblue')

plt.title('User Registrations Per Day')
plt.xlabel('Date')
plt.ylabel('Number of Registrations')
plt.show()

# merge clients dataframe with data dataframe
data = pd.merge(data, clients, on='user_id')

incoming_data = data[data['direction'] == 'in']

# calculate waiting time and add it as a collumn
incoming_data['in_waiting_time_sec'] = incoming_data['total_call_duration_sec'] - incoming_data['call_duration_sec']
avg_waiting_time_incoming_calls = incoming_data['in_waiting_time_sec'].mean().round()
print('The total average waiting time incoming calls: ' + str(avg_waiting_time_incoming_calls) + " seconds")

# calculate 1 sigma to use as a measure
wait_mean = incoming_data['in_waiting_time_sec'].mean()
wait_std = incoming_data['in_waiting_time_sec'].std()
sigma = wait_mean + (1 * wait_std)

# group operators by their average wait time
op_waiting_time = incoming_data.groupby('operator_id')['in_waiting_time_sec'].mean().round().reset_index()

# group rows into wait time categories
def determineWaitTime(row):
    wait_time = row['in_waiting_time_sec']
    
    # if avg operator wait time is less than total avg wait time, not long waiting op
    if wait_time < sigma:
        return 'False'
    else:
        return 'True'
    
# determine if operators have a long wait time or not
op_waiting_time['long_waiting_time'] = op_waiting_time.apply(determineWaitTime, axis=1)
display(op_waiting_time[op_waiting_time['long_waiting_time'] == 'True'])

# plot pie chart with proportions
plt.figure(figsize=(10, 5))

missed_call_ratio = incoming_data.groupby('is_missed_call')['operator_id'].count().reset_index()
missed_call_ratio.columns = ['is_missed_call', 'count']

labels = ['False', 'True']
plt.pie(missed_call_ratio['count'], labels=labels,autopct='%0.f%%', shadow=True, startangle=145)
plt.title('Shares of Missed Calls')
plt.show()

missed_calls = incoming_data['is_missed_call'].value_counts().reset_index()
missed_calls.columns = ['is_missed_call', 'count']

avg_missed_incoming_calls = missed_calls['count'][1] / missed_calls['count'][0]
print('The total average number of missed incoming calls: ' + str(avg_missed_incoming_calls))

# get data where missed calls is true
op_missed_calls = incoming_data[incoming_data['is_missed_call'] == True]
op_missed_calls = op_missed_calls.groupby('operator_id')['is_missed_call'].count().reset_index()
op_missed_calls.columns = ['operator_id', 'missed_call_count']

avg_missed_incoming_calls_per_op = op_missed_calls.mean()['missed_call_count'].round()

# calculate 1 sigma to use as a measure
wait_mean = op_missed_calls['missed_call_count'].mean()
wait_std = op_missed_calls['missed_call_count'].std()
sigma = wait_mean + (1 * wait_std)

# group rows into missed callcategories
def determineMissedCalls(row):
    missed_call = row['missed_call_count']
    
    # if  num of missed calls per op is less than avg
    if missed_call < sigma:
        return 'False'
    else:
        return 'True'
    
op_missed_calls['many_missed_calls'] = op_missed_calls.apply(determineMissedCalls, axis=1)
display(op_missed_calls[op_missed_calls['many_missed_calls'] == 'True'])

# only outgoing direction
outgoing_data = data[data['direction'] == 'out']

avg_outgoing_per_op = outgoing_data['calls_count'].mean().round()
print('The average number of outgoing calls per operator is: ' + str(avg_outgoing_per_op))

# calculate number of outgoing calls per operators
outgoing_per_op = outgoing_data.groupby('operator_id')['calls_count'].count().reset_index()
outgoing_per_op.columns = ['operator_id', 'total_calls_count']

# operators who are supposed to make outgoing calls are those with more than one outgoing call
outgoing_per_op = outgoing_per_op[outgoing_per_op['total_calls_count'] > 1]

# calculate 1 sigma to use as a measure
wait_mean = outgoing_per_op['total_calls_count'].mean()
wait_std = outgoing_per_op['total_calls_count'].std()
sigma = wait_mean + (1 * wait_std)

# group rows into missed callcategories
def determineOutgoingCalls(row):
    outgoing_call = row['total_calls_count']
    
    # if op num of outgoing calls is greater than avg
    if outgoing_call > wait_mean:
        return 'False'
    else:
        return 'True'
    
outgoing_per_op['few_outgoing_calls'] = outgoing_per_op.apply(determineOutgoingCalls, axis=1)
display(outgoing_per_op[outgoing_per_op['few_outgoing_calls'] == 'True'])

# combine dataframes
total_in_operators = pd.merge(op_missed_calls, op_waiting_time, on='operator_id')
total_in_operators = pd.merge(total_in_operators, outgoing_per_op, on='operator_id')

# identify ineffective ops for incoming calls
ineffective_incoming_ops = total_in_operators[(total_in_operators['many_missed_calls'] == 'True') & (total_in_operators['long_waiting_time'] == 'True')]
display(ineffective_incoming_ops)

# identify ineffective ops for outgoing calls
ineffective_outgoing_ops = total_in_operators[total_in_operators['few_outgoing_calls'] == 'True']
display(ineffective_outgoing_ops)

# concatenate results together
result = pd.concat([ineffective_incoming_ops, ineffective_outgoing_ops], ignore_index=True, sort=False)
display(result)


# get operator id of ineffective operators
ineffective_op_id = result['operator_id']
ineffective_op_id.columns = ['operator_id']

# merge with bigger data set
data_ineffective = pd.merge(data, ineffective_op_id, on='operator_id')
total_ineffective_ops = data_ineffective['operator_id'].nunique()

# organize ineffective operators by tariff plan
tariff = data_ineffective.groupby('tariff_plan')['operator_id'].nunique().reset_index()
tariff.columns = ['tariff_plan', 'ineffective_operators']

# organize all operators by tariff plan
tariff_all = data.groupby('tariff_plan')['operator_id'].nunique().reset_index()
tariff_all.columns = ['tariff_plan', 'total_operators']
tariff['total_operators'] = tariff_all['total_operators']

# add the share of ineffective ops / total ops per tariff plan
tariff['op_share (%)'] = (tariff['ineffective_operators'] / tariff['total_operators'] * 100).round()
display(tariff)

# Test hypothesis: Average call duration for ineffective operators and effective operators are the same.

# get operator id of ineffective operators
ineffective_op_id = result['operator_id'].to_numpy()

# get data relating to ineffective operators
ineffective_op_data = data[data['operator_id'].isin(ineffective_op_id)]

# calculate average call duration for ineffective ops
ineffective_call_duration_average = ineffective_op_data['call_duration_sec'].mean().round(decimals=2)

# find operator id of best operators 
effective_ops = total_in_operators[(total_in_operators['many_missed_calls'] == 'False') & (total_in_operators['long_waiting_time'] == 'False')]

# get operator id in best operators
effective_op_id = effective_ops['operator_id'].to_numpy()

# get data relating to best operators
effective_op_data = data[data['operator_id'].isin(effective_op_id)]

# calculate average call duration for ineffective ops
effective_call_duration_average = effective_op_data['call_duration_sec'].mean().round(decimals=2)

# perform a t-test
results = st.ttest_ind(effective_op_data['call_duration_sec'], ineffective_op_data['call_duration_sec'], equal_var=False)
p_value = results.pvalue
alpha = 0.05

if p_value < alpha:
    print('Reject H0')
else:
    print('Cannot reject H0')
