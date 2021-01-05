#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install plotly --upgrade')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import matplotlib.dates as mdates
import datetime 
import math
import plotly.express as px

# Open data files
data = pd.read_csv('/datasets/logs_exp_us.csv', sep='\t')

# Study general information
data.info()
data.head()

#Rename the columns
data.columns = ['event_name', 'user_id', 'event_datetime', 'exp_id']

# Drop any duplicates in the database
data.drop_duplicates(inplace=True)

# convert datetime from unix
data['event_datetime']=pd.to_datetime(data['event_datetime'], unit='s')

# Add a separate date and time column
data['event_date'] = data['event_datetime'].dt.date
data['event_time'] = data['event_datetime'].dt.time
display(data)

# ### How many events are in the logs?

# transform value_counts for event name to dataframe
tmp = data['event_name'].value_counts().rename_axis('event_name').reset_index(name='count')
print('There are 5 events in the logs: ' + str(tmp['event_name'].to_numpy()))

# ### How many users are in the logs?
print('There are ' + str(len(data['user_id'].unique())) + ' unique users in the logs.')

# ### What's the average number of events per user?
# calculate the average number of events per user
avgEvents = data.groupby('user_id')['event_name'].count().reset_index()
avgEvents.columns = ['user_id', 'num_of_events']
print('The average number of events per user is ' + str(avgEvents['num_of_events'].mean().round()))

# for each user, how many actions per event did they complete and then group it by event to see distribution by action
uniqueEvents = data.groupby('user_id')['event_name'].nunique().reset_index().groupby('event_name')['user_id'].nunique().reset_index()
uniqueEvents.columns = ['event_number', 'unique_users']
uniqueEvents = uniqueEvents.sort_values(by='unique_users', ascending=False)
uniqueEvents['event_name'] = data['event_name']
display(uniqueEvents)

# ### What period of time does the data cover? Is there equally complete data for the entire period? 
# determine min and max dates recorded
minDate = data['event_date'].min()
maxDate = data['event_date'].max()
print('The period of time that the data covers is from: ' + str(minDate) +  ' to ' + str(maxDate))

# count the unique dates and their frequencies
timePeriod = data['event_date'].value_counts().rename_axis('event_date').reset_index(name='count').sort_values(by='event_date')

# plot bar graph
plt.figure(figsize=(15, 9))
ax = sns.barplot(data = timePeriod, x='event_date', y='count')

# Add titles and captions
plt.title('Date Frequencies')
plt.xlabel('Dates')
plt.ylabel('Number of Entries')

# label bars with data
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
    
# rotate x axis labels
for item in ax.get_xticklabels():
    item.set_rotation(45)

plt.show()

# ### Find the moment at which the data starts to be complete and ignore the earlier section.
# remove older date entries
new_data = data[data['event_date'] >= pd.to_datetime('2019-08-01')]

oldNumOfUsers = len(data['user_id'].unique())
oldNumOfEvents = len(data['event_datetime'].unique())

newNumOfUsers = len(new_data['user_id'].unique())
newNumOfEvents = len(new_data['event_datetime'].unique())

print('We have lost ', (oldNumOfUsers - newNumOfUsers), ' users and ', (oldNumOfEvents - newNumOfEvents), ' events by removing any data entires from the month of July.')

expUsers = new_data.groupby('exp_id')['user_id'].nunique().reset_index()
expUsers.columns = ['exp_id', 'unique_users']
display(expUsers)

# ## Study the event funnel
# use value_counts() to find frequency of events
eventsFreq = new_data['event_name'].value_counts().rename_axis('event_name').reset_index(name='count')
display(eventsFreq)

# ### Find the number of users who performed each of these actions. Calculate the proportion of users who performed the action at least once.
# find the number of users who performed each action
eventUsers = new_data.groupby('event_name')['user_id'].nunique().sort_values(ascending=False) #/ new_data.user_id.nunique()
eventUsers = eventUsers.reset_index()
eventUsers.columns = ['event_name', 'user_unique']
eventUsers['user_percentage'] = (eventUsers['user_unique'] / new_data.user_id.nunique()) * 100

# calculate the ratio of how many users performaed the action at least once
# count the number of event timestamps for each user for each event
at_least_once = new_data.groupby(['user_id', 'event_name'])['event_datetime'].count().reset_index()

# extract only the users who had more than one timestamp per event
at_least_once = at_least_once[at_least_once['event_datetime'] > 1]

# count how many unique users had more than one action per event compared to all unique users
at_least_once = at_least_once.groupby('event_name')['user_id'].nunique() / new_data.groupby('event_name')['user_id'].nunique()
at_least_once = at_least_once.reset_index()
at_least_once.columns = ['event_name', 'at_least_once_percentage']
at_least_once['at_least_once_percentage'] = at_least_once['at_least_once_percentage'] * 100

# merge table together
eventUsers = eventUsers.merge(at_least_once, on='event_name')
display(eventUsers)

# plot bar graph
plt.figure(figsize=(11, 7))
ax = sns.barplot(data = eventUsers, x='event_name', y='user_unique')

# Add titles and captions
plt.title('Users Performing Each Action')
plt.xlabel('Event Name')
plt.ylabel('Number of Uses')

# label bars with data
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
    
# rotate x axis labels
for item in ax.get_xticklabels():
    item.set_rotation(45)

plt.show()

# ### Use the event funnel to find the share of users that proceed from each stage to the next. At what stage do you lose the most users?  What share of users make the entire journey from their first event to payment?
# create funnel of unique users for each event
funnel = new_data.groupby('event_name')['user_id'].nunique().reset_index().sort_values(by='user_id', ascending=False)

# create percentage change from one funnel to another 
funnel['percentage_change'] = funnel['user_id'].pct_change() * 100

# merge table together
eventUsers = eventUsers.merge(funnel, on='event_name')
eventUsers =eventUsers.fillna(0)
eventUsers = eventUsers[['event_name', 'user_unique', 'user_percentage', 'percentage_change']]
display(eventUsers)

# ### Plot the funnel
# plot funnel visualization
funnel_by_groups = []

# for every experiment group 
for i in new_data.exp_id.unique():
    group = new_data[new_data.exp_id == i].groupby(['event_name', 'exp_id'])['user_id'].nunique().reset_index().sort_values(by='user_id', ascending=False)
    
    funnel_by_groups.append(group)

# concatenate the data to have one big dataframe
funnel_by_groups = pd.concat(funnel_by_groups)

# display funnel
fig = px.funnel(funnel_by_groups, x='user_id', y='event_name', color='exp_id')
fig.show()

# ## Study the results of the experiment
# ### How many users are there in each group?
usersExp = new_data.groupby('exp_id')['user_id'].nunique().reset_index()
usersExp.columns = ['exp_id', 'unique_users']
display(usersExp)

# ### In each of the control groups, find the number of users who performed each event.
# create a pivot table with the number of unique users in each control gorup that goes through each action
expGroups = new_data.pivot_table(index='event_name', values='user_id', columns='exp_id', aggfunc=lambda x: x.nunique()).reset_index()
expGroups.columns = ['event_name', '246', '247', '248']
expGroups = expGroups.sort_values(by='246', ascending=False)
display(expGroups)

# ### In each of the control groups, find the number of users who performed the most popular event and find their share.
# The most popular event is the main screen appearing
mainGroups = expGroups[expGroups['event_name'] == 'MainScreenAppear']
mainGroups.columns = ['event_name', '246_performed', '247_performed', '248_performed']

# calculate share 
mainGroups['246_share'] = mainGroups['246_performed'] / usersExp.loc[0,'unique_users'] * 100
mainGroups['247_share'] = mainGroups['247_performed'] / usersExp.loc[1,'unique_users'] * 100
mainGroups['248_share'] = mainGroups['248_performed'] / usersExp.loc[2,'unique_users'] * 100

display(mainGroups)

# ### Check if there is a statistically significant difference between all of the control groups.
expGroups = new_data.pivot_table(index='event_name', values='user_id', columns='exp_id', aggfunc=lambda x: x.nunique()).reset_index()

# find statistical significance for each group for each event
def check_hypothesis(group1, group2, alpha):

    # for every event
    for event in expGroups.event_name.unique():

        # define successes 
        successes1 = expGroups[expGroups.event_name == event][group1].iloc[0]
        successes2 = expGroups[expGroups.event_name == event][group2].iloc[0]

        # define trials
        trials1 = new_data[new_data.exp_id == group1]['user_id'].nunique()
        trials2 = new_data[new_data.exp_id == group2]['user_id'].nunique()

        # proportion for success in group 1
        p1 = successes1 / trials1

        # proportion for success in group 2
        p2 = successes2 / trials2

        # proportion in a combined dataset
        p_combined = (successes1 + successes2) / (trials1 + trials2)

        # define difference and z value
        difference = p1 - p2
        z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1/trials1 + 1/trials2))

        # calculate distribution
        distr = stats.norm(0,1)

        # calculate p_value
        p_value = (1 - distr.cdf(abs(z_value))) * 2
        print('p_value: ', p_value)
        if (p_value < alpha):
            print("Reject H0 for",event, 'and groups ',group1,' and ', group2, '\n')
        else:
            print("Fail to Reject H0 for", event,'and groups ',group1,' and ', group2, '\n')

# ### Calculate statistical difference between control groups 246 and 247.
check_hypothesis(246, 247, 0.05)

# ### Calculate statistical difference between control groups 246 and 248.
check_hypothesis(246, 248, 0.05)

# ### Calculate statistical difference between control groups 247 and 248.
check_hypothesis(247, 248, 0.05)

# ### Calculate how many statistical hypothesis tests you carried out and run it through the Bonferroni correction. What should the significance level be? 
# calculate corrected bonferroni correction

# family wise error rate = 1 - (1 - alpha for individual test) ^ number of tests
alpha = 0.05 

# 3 control groups, 5 events per cotrol group
num_of_tests = 15
fwer = 1 - (1 - alpha)** (num_of_tests)
print('The alpha level with Bonferroni correction is: ',fwer)

check_hypothesis(246, 247, 0.5)
check_hypothesis(246, 248, 0.5)
check_hypothesis(247, 248, 0.5)
