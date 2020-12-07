#!/usr/bin/env python
# coding: utf-8

# # Integrated Project 2
# 
# I work at a startup that sells food products and I need to investigate user behavior for the company's app. First, study the sales funnel to find out how users reach the purchase stage. Then, look at the results of an A/A/B test: The designers would like to change the fonts for the entire app, but the managers are afraid the users might find the new design intimidating. They decide to make a decision based on the results of an A/A/B test.

# In[1]:


get_ipython().system('pip install plotly --upgrade')


# ## Open data and study information

# In[2]:


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


# After a brief observation of the raw data, there are 4 columns where EventName is the name of the event, DeviceIDHash is the user ID, EventTimestamp is the Unix timestamp of when the user did the event and ExpId is which experiment group the user has been split to. 

# ## Data Preprocessing

# In[3]:


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


# To clean up and preprocess the data, I renamed the columns to more intuitive names, I dropped any duplicates found within the data and I converted the timestamp column from Unix time to year month day hour minute second format. Additionally, I created two new columns: event_date to store just the date and event_time to store just the time. 

# ## Study the data

# ### How many events are in the logs?

# In[4]:


# transform value_counts for event name to dataframe
tmp = data['event_name'].value_counts().rename_axis('event_name').reset_index(name='count')
print('There are 5 events in the logs: ' + str(tmp['event_name'].to_numpy()))


# ### How many users are in the logs?

# In[5]:


print('There are ' + str(len(data['user_id'].unique())) + ' unique users in the logs.')


# ### What's the average number of events per user?

# In[6]:


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


# The 7551 unique users in the logs are distributed by the event number as seen in the print out above. There were 2707 users who only completed event 1 (main screen appears), 1021 users who completed events 1 and 2 (offer screen appears), 317 users who completed events 1, 2, and 3 (cart screen appears) and 3035 users who went full circle by completing all events, including the payment screen successful. We expect every user to complete at least the 4 events and only 3035 / 7551 users (less than half) actually complete all of those events. 

# ### What period of time does the data cover? Is there equally complete data for the entire period? 

# In[7]:


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


# The period of time that the data covers is from: 2019-07-25 to 2019-08-07. Once the frequency of the recorded dates are plotted, one can see that the majority of the date timestamps are from the first week of the month of August as there are over 30,000 entries for each day during that week. Although the period of time covers July 25th to August 8th of 2019, the moment where the data starts to be complete is on August 1st.

# ### Find the moment at which the data starts to be complete and ignore the earlier section.

# In[8]:


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


# In order to skew the overall picture, the data used from now on will only be data after August 1st - therefore removing any data entries from July. After excluding older data, the data lost 2,610 events and 17 users. Additionally, we still have users presented from all 3 groups with 2484 users from group 246, 2513 users from group 247 and 2537 users from group 248.

# ## Study the event funnel

# ### See what events are in the logs and their frequency of occurrence.

# In[9]:


# use value_counts() to find frequency of events
eventsFreq = new_data['event_name'].value_counts().rename_axis('event_name').reset_index(name='count')
display(eventsFreq)


# The result from value_counts() shows the frequency of each event in the logs: the most frequent event is the appearance of the main screen (117,328 events) which shrinks in number by more than half by the time the appearance of the offer sceen occurs (46,333 events). After the appearance of the offer sceen occurs, most of these events are brought to the appearance of the cart screen (42,303 events) and a large majority of these events are brought to payment screen successful (33,918 events). A substantially low number of events are the tutoral (1005 events).   

# ### Find the number of users who performed each of these actions. Calculate the proportion of users who performed the action at least once.

# In[10]:


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


# The number of users who performed each of these event actions can be seen in the table above: the user_unique column shows how many unique users performed each of the actions (any number of times) and the user_proportion column shows the percentage of users who performed each of these actions.
# 
# The proportion of users who performed the action at least once can be seen in the at_least_once_percentage column. As shown: out of the 7419 unique users that saw the main screen appear (98% of users), about 96% of them saw the main screen appear at least once. This column can be seen as a retention rate and tells us that a high percentage of users come back to the event.

# In[11]:


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


# The journey of users with each event name can be seen in the bar graph above where the most interacted with event is the appearance of the main screen with over 7000 users, which continues dropping as the sequence of events goes on.

# ### In what order do you think the actions took place? Are all of them part of a single sequence?

# I believe that the order in which the actions took place is: 
# 
# Main Screen -> Offer Screen -> Cart Screen -> Payment Successful Screen
# 
# They are not all part of a single sequence: it's possible to make a purchase without viewing the cart or make a purchase without seeing an offer pag. I'm not sure what the tutorial event is but I don't believe that it falls in a sequence.

# ### Use the event funnel to find the share of users that proceed from each stage to the next. At what stage do you lose the most users?  What share of users make the entire journey from their first event to payment?

# In[12]:


# create funnel of unique users for each event
funnel = new_data.groupby('event_name')['user_id'].nunique().reset_index().sort_values(by='user_id', ascending=False)

# create percentage change from one funnel to another 
funnel['percentage_change'] = funnel['user_id'].pct_change() * 100

# merge table together
eventUsers = eventUsers.merge(funnel, on='event_name')
eventUsers =eventUsers.fillna(0)
eventUsers = eventUsers[['event_name', 'user_unique', 'user_percentage', 'percentage_change']]
display(eventUsers)


# In the table above within the percentage_change column, we can see each event name, the number of unique users who had an action with th event and the percentage change from one event to antoher. All the users started out on the main screen but there was a 38% decrease in users who processed to the offer screen, etc. This is stage that lost the most users, with exception to the tutorial event which is not part of the logical sequence. 
# 
# The share of users that make the entire journey from the first main screen appear event to the payment screen successful event can be seen in the user_percentage column in the payment screen successful row: about 47% of users complete all of the events.

# ### Plot the funnel

# In[13]:


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

# In[14]:


usersExp = new_data.groupby('exp_id')['user_id'].nunique().reset_index()
usersExp.columns = ['exp_id', 'unique_users']
display(usersExp)


# The table above shows how many users there are in each group: there are about the same amount of users in each group, with group 246 being slightly smaller than the other groups. 

# ### In each of the control groups, find the number of users who performed each event.

# In[15]:


# create a pivot table with the number of unique users in each control gorup that goes through each action
expGroups = new_data.pivot_table(index='event_name', values='user_id', columns='exp_id', aggfunc=lambda x: x.nunique()).reset_index()
expGroups.columns = ['event_name', '246', '247', '248']
expGroups = expGroups.sort_values(by='246', ascending=False)
display(expGroups)


# We can split up the user actions with event names by the different experiment groups, as shown by the table above. This shows the funnel according to experiment groups.

# ### In each of the control groups, find the number of users who performed the most popular event and find their share.

# In[16]:


# The most popular event is the main screen appearing
mainGroups = expGroups[expGroups['event_name'] == 'MainScreenAppear']
mainGroups.columns = ['event_name', '246_performed', '247_performed', '248_performed']

# calculate share 
mainGroups['246_share'] = mainGroups['246_performed'] / usersExp.loc[0,'unique_users'] * 100
mainGroups['247_share'] = mainGroups['247_performed'] / usersExp.loc[1,'unique_users'] * 100
mainGroups['248_share'] = mainGroups['248_performed'] / usersExp.loc[2,'unique_users'] * 100

display(mainGroups)


# The most popular event is the main screen appearing: the number of users who performed this event is shown under the _ performed columns and their share compared to the total can be seen in the _ share columns. For each group, over 98% of the users performed this main screen appear action.

# ### Check if there is a statistically significant difference between all of the control groups.

# Having 3 different experiment groups, it is important to ensure that that the results from these groups are based on fair numbers. In order to do so, we want to check if there is a statistically significant difference between all of the control groups. If we find that there is a significant difference, then the control groups have not be split up equally and any results we deduct will not accurately represent the population. 

# In[17]:


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

# In[18]:


check_hypothesis(246, 247, 0.05)


# We want to test the statistical significance of the difference in conversion between control groups 246 and 247. This can be done using the CDF function which returns the expected probability for observing a value (number of unique users per event in group 246) less than or equal to a given value (number of unique users per event in group 247).
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between control groups 246 and 247. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups 246 and 247.
# 
# For each event, the p_value is greater than the defined alpha level of 0.05 which means that we cannot reject the null hypothesis and we determine that there is a statistically significant difference between the two control groups for each event.

# ### Calculate statistical difference between control groups 246 and 248.

# In[19]:


check_hypothesis(246, 248, 0.05)


# We want to test the statistical significance of the difference in conversion between control groups 246 and 248. This can be done using the CDF function which returns the expected probability for observing a value (number of unique users per event in group 246) less than or equal to a given value (number of unique users per event in group 248).
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between control groups 246 and 248. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups 246 and 248.
# 
# For each event, the p_value is greater than the defined alpha level of 0.05 which means that we cannot reject the null hypothesis and we determine that there is a statistically significant difference between the two control groups for each event.

# ### Calculate statistical difference between control groups 247 and 248.

# In[20]:


check_hypothesis(247, 248, 0.05)


# We want to test the statistical significance of the difference in conversion between control groups 247 and 248. This can be done using the CDF function which returns the expected probability for observing a value (number of unique users per event in group 247) less than or equal to a given value (number of unique users per event in group 248).
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between control groups 247 and 248. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups 247 and 248.
# 
# For each event, the p_value is greater than the defined alpha level of 0.05 which means that we cannot reject the null hypothesis and we determine that there is a statistically significant difference between the two control groups for each event.

# ###  Can you confirm that the groups were split properly?

# After calculating the statistical differences between all control groups (246 and 247, 246 and 248, 247 and 247) with a CDF function, I found that the p_value was always greaters than the defined alpha level of 0.05 which means that there was a statistically significant difference between all the control groups for each event. Thus, it can be confirmed that the groups were not split properly. 

# ### Do the same thing for the group with altered fonts. Compare the results with those of each of the control groups for each event in isolation. Compare the results with the combined results for the control groups. What conclusions can you draw from the experiment?

# Do I need to combined grouped 246 and 247 by taking the average and compare it to 248 with the same hypothesis test i did above? i'm not sure what i'm supposed to do in this question

# ### Calculate how many statistical hypothesis tests you carried out and run it through the Bonferroni correction. What should the significance level be? 

# In[21]:


# adjust the alpha value and run the tests again

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


# Previously, I set the alpha significance level to 0.05. Using the Bonferroni correction, we can use the family wise error rate formula to calculate a new alpha level which is dependent on the number of tests being performed. This error rate indicates the probability of making one or more false discoveries when performing multiple hypothesis tests. This error rate for our calculations comes out to be 0.5 which means that one in every five results could be false. Given this, we should change the significance level to be set to 0.5 to avoid an error rate of 20%. 
# 
# When I ran the statistical significance tests again with an alpha level of 0.5: 8 out of the 15 tests were rejected which means that there is no statistical significance between more of the control groups than previously (where 0 out of the 15 tests were rejected.)
