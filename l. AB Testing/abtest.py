#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates
import seaborn as sns
from scipy import stats
import math
import matplotlib.dates as mdates
import datetime 
import math
import plotly.express as px


# ## Download the data
# read in the data and load into DataFrame
data = pd.read_csv('/datasets/ab_project_marketing_events_us.csv')
display(data)

marketing = data[data['name'] == 'Christmas&New Year Promo']
display(marketing)

# read in the data and load into DataFrame
new_users = pd.read_csv('/datasets/final_ab_new_users_us.csv')

display(new_users)
new_users.info()

# read in the data and load into DataFrame
events = pd.read_csv('/datasets/final_ab_events_us.csv')
display(events)
events.info()

# read in the data and load into DataFrame
test_participants = pd.read_csv('/datasets/final_ab_participants_us.csv')
display(test_participants)
test_participants.info()

# marketing dataframe - convert dates to datetime objects
marketing['start_dt'] = pd.to_datetime(marketing['start_dt'], format="%Y-%m-%d")
marketing['finish_dt'] = pd.to_datetime(marketing['finish_dt'], format="%Y-%m-%d")

# new_users dataframe - convert dates to datetime objects
new_users['first_date'] = pd.to_datetime(new_users['first_date'], format="%Y-%m-%d")
new_users = new_users.rename(columns={'first_date': 'join_date'})

# events dataframe - convert dates to datetime objects
events['event_dt'] = pd.to_datetime(events['event_dt'], format="%Y-%m-%d %H:%M:%S")
events = events.rename(columns={'details': 'purchase_amount'})

# ### Are there any missing or duplicate values
# drop any duplicate rows
new_users.drop_duplicates(inplace=True)
events.drop_duplicates(inplace=True)
test_participants.drop_duplicates(inplace=True)

# missing values
events['purchase_amount'] = events['purchase_amount'].fillna(0.00)
display(events)


# ## Carry out exploratory data analysis
# ### Find the number of users who performed each of these actions. 
events_count = events.groupby('event_name').agg({'user_id':'nunique'}).reset_index()

# plot bar graph
plt.figure(figsize=(8, 7))
ax = sns.barplot(data = events_count, x='event_name', y='user_id')

# Add titles and captions
plt.title('Users Performing Each Event')
plt.xlabel('Event Name')
plt.ylabel('Number of Users')

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

# ### Study conversion at different stages of the funnel.
# ### Find the number of users who performed each of these actions.

# query only ab_test for interface_eu_test
test_participants_interface = test_participants[test_participants['ab_test'] == 'interface_eu_test']

# merge test_participants_interface and events into one big table
participants_events = pd.merge(test_participants_interface, events, on='user_id')

# find the number of users who performed each event
eventUsers = participants_events.groupby('event_name')['user_id'].nunique().sort_values(ascending=False) #/ new_data.user_id.nunique()
eventUsers = eventUsers.reset_index()
eventUsers.columns = ['event_name', 'user_unique']
eventUsers['user_percentage'] = (eventUsers['user_unique'] / participants_events.user_id.nunique()) * 100

# calculate the ratio of how many users performaed the event at least once
# count the number of event timestamps for each user for each event
at_least_once = participants_events.groupby(['user_id', 'event_name'])['event_dt'].count().reset_index()

# extract only the users who had more than one timestamp per event
at_least_once = at_least_once[at_least_once['event_dt'] > 1]

# count how many unique users had more than one action per event compared to all unique users
at_least_once = at_least_once.groupby('event_name')['user_id'].nunique() / participants_events.groupby('event_name')['user_id'].nunique()
at_least_once = at_least_once.reset_index()
at_least_once.columns = ['event_name', 'at_least_once_percentage']
at_least_once['at_least_once_percentage'] = at_least_once['at_least_once_percentage'] * 100

# merge table together
eventUsers = eventUsers.merge(at_least_once, on='event_name')
display(eventUsers)

# ### Use the event funnel to find the share of users that proceed from each stage to the next.
#create funnel of unique users for each event
funnel = participants_events.groupby('event_name')['user_id'].nunique().reset_index().sort_values(by='user_id', ascending=False)

# create percentage change from one funnel to another 
funnel['percentage_change'] = funnel['user_id'].pct_change() * 100

# merge table together
eventUsers = eventUsers.merge(funnel, on='event_name')
eventUsers =eventUsers.fillna(0)
eventUsers = eventUsers[['event_name', 'percentage_change']]
display(eventUsers)

# ### Is the number of events per user distributed equally among the samples?
# query only ab_test for interface_eu_test
test_participants_interface = test_participants[test_participants['ab_test'] == 'interface_eu_test']

# merge test_participants_interface and events into one big table
participants_events = pd.merge(test_participants_interface, events, on='user_id')
#display(participants_events)

# count number of events per event per group
events_per_user = participants_events.groupby(['group','event_name']).count()
events_per_user = events_per_user.reset_index()
events_per_user = events_per_user[['group', 'event_name', 'user_id']]
events_per_user.columns = ['group', 'event_name', 'event_count']
display(events_per_user)

# line graph - extract names
event_names = np.array(events_per_user['event_name'][0:4])
group_a_event_count = np.array(events_per_user['event_count'][0:4])
group_b_event_count = np.array(events_per_user['event_count'][4:8])

# plot graph
fig, ax = plt.subplots()
ax.plot(event_names, group_a_event_count, label="Group A")
ax.plot(event_names, group_b_event_count, label="Group B")
ax.set_title('Number of Events Per Sample')
ax.set_xlabel('Event Names')
ax.set_ylabel('Number of Events')
ax.legend()
plt.show()

# ### Are there users who are present in both samples?
group_a = participants_events[participants_events['group'] == 'A']
group_b = participants_events[participants_events['group'] == 'B']

doubles = pd.merge(group_a, group_b, on='user_id')
display(doubles)

# ### How is the number of events distributed among days?
# add date only column
events['event_date'] = events['event_dt'].dt.date

# group by event name and date, aggregate count of event per date
event_dates = events.groupby(['event_date']).count()
event_dates = event_dates.reset_index()
event_dates = event_dates[['event_date', 'user_id']]
event_dates.columns = ['event_date', 'event_count']

# line graph
plt.figure(figsize=(10,7))
dates = matplotlib.dates.date2num(event_dates['event_date'])
plt.plot_date(dates, event_dates['event_count'])
plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator())
plt.plot(dates, event_dates['event_count'])
plt.gcf().autofmt_xdate()
plt.title('Number of Events Per Date')
plt.xlabel('Dates')
plt.ylabel('Number of Events')

# ## Evaluate the A/B test results
# ### Test whether the recommendation system yields better conversion.
# calculate how manytusers in each group
usersExp = participants_events.groupby('group')['user_id'].nunique().reset_index()
usersExp.columns = ['exp_id', 'unique_users']
display(usersExp)

# create a pivot table with the number of unique users in each gorup that goes through each action
expGroups = participants_events.pivot_table(index='event_name', values='user_id', columns='group', aggfunc=lambda x: x.nunique()).reset_index()
expGroups.columns = ['event_name', 'group_a', 'group_b']
expGroups = expGroups.sort_values(by='group_a', ascending=False)
display(expGroups)

# The most important event is purchase
purchase = expGroups[expGroups['event_name'] == 'purchase']
purchase.columns = ['event_name', 'group_a_performed', 'group_b_performed']

# calculate share 
purchase['group_a_share'] = purchase['group_a_performed'] / usersExp.loc[0,'unique_users'] * 100
purchase['group_b_share'] = purchase['group_b_performed'] / usersExp.loc[1,'unique_users'] * 100

display(purchase)

# ### Use a z-test to check the statistical difference between the proportions.

expGroups = participants_events.pivot_table(index='event_name', values='user_id', columns='group', aggfunc=lambda x: x.nunique()).reset_index()
group1 = 'A'
group2 = 'B'
alpha = 0.05

# find statistical significance for each group for each event
for event in expGroups.event_name.unique():

    # define successes 
    successes1 = expGroups[expGroups.event_name == event][group1].iloc[0]
    successes2 = expGroups[expGroups.event_name == event][group2].iloc[0]

    # define trials
    trials1 = participants_events[participants_events.group == group1]['user_id'].nunique()
    trials2 = participants_events[participants_events.group == group2]['user_id'].nunique()

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
