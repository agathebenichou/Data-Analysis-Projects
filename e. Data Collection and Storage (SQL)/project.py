#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install squarify')
import squarify
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

company_name_trips_amount = pd.read_csv('/datasets/project_sql_result_01.csv')

dropoff_location_avg_trips = pd.read_csv('/datasets/project_sql_result_04.csv')

 ### Study data

#display(company_name_trips_amount)
company_name_trips_amount.info()

#display(dropoff_location_avg_trips)
dropoff_location_avg_trips.info()

# ### Pre-process data

# company_name_trips_amount table

# Change company_name to string
company_name_trips_amount['company_name'] = company_name_trips_amount['company_name'].astype(str)

# Drop duplicates
company_name_trips_amount.drop_duplicates(inplace=True)

# Drop rows with null
company_name_trips_amount.dropna(inplace=True)

# Remove unneccessary numbers and - from the name
company_name_trips_amount['company_name'] = company_name_trips_amount['company_name'].str.replace('\d+ - (\d+)* -*', '')
company_name_trips_amount['company_name'] = company_name_trips_amount['company_name'].str.replace('\d+ - ', '')

# Change dropoff_location_name to string
dropoff_location_avg_trips['dropoff_location_name'] = dropoff_location_avg_trips['dropoff_location_name'].astype(str)

# Drop duplicates
dropoff_location_avg_trips.drop_duplicates(inplace=True)

# Drop rows with null
dropoff_location_avg_trips.dropna(inplace=True)

# ### Identify the top 10 neighborhoods by number of dropoffs (Nov 2017)

top10dropoffs = dropoff_location_avg_trips.sort_values(by='average_trips',ascending=False).round(decimals=2).head(10)

fig, ax = plt.subplots(figsize=(17,10))
ax.vlines(x=top10dropoffs.dropoff_location_name, ymin=0, ymax=top10dropoffs.average_trips,color='purple',alpha=0.7,linewidth=1)
ax.scatter(x=top10dropoffs.dropoff_location_name,y=top10dropoffs.average_trips, s=75, color='black',alpha=0.7)

ax.set_title("Top 10 Dropoff Neighborhoods", fontdict={'size':15})
ax.set_ylabel('Avg Number of Dropoffs (Nov 2017)')
ax.set_xlabel('Dropoff Neighborhood')
ax.set_xticks(top10dropoffs.dropoff_location_name)
ax.set_xticklabels(top10dropoffs.dropoff_location_name, rotation=90, fontdict={'horizontalalignment':'right','size':12})
for row in top10dropoffs.itertuples():
    ax.text(row.dropoff_location_name, row.average_trips+30,s=round(row.average_trips,2))

# ### Taxi Companies and number of rides (Nov 15-16, 2017)

# Plot all taxi companies

plotData = company_name_trips_amount.sort_values(by='trips_amount')
ax = plotData.plot(kind='bar', x='company_name', y='trips_amount',figsize=(15,10))

ax.set_title("All Taxi Companies and # of rides on Nov 15-15, 2017", fontsize=18)
ax.set_ylabel("Number of Rides", fontsize=18);
ax.set_xlabel("Taxi Companies", fontsize=18);

# Plot only the top taxi companies
plotData = company_name_trips_amount.sort_values(by='trips_amount')
plotData = plotData[plotData['trips_amount'] > 1000] # Only taxi companies with > 1000 rides in a 1 day period
sizes = plotData.trips_amount.values.tolist()
labels = plotData.apply(lambda x: str(x[0]) + "\n" + str(round(x[1])),axis=1)
plt.figure(figsize=(15,9))
squarify.plot(sizes=sizes,label=labels,alpha=0.5)

plt.title('Distribution of Platform Market', fontsize=22)

# ### Test whether the average duration of rides from the Loop to O'Hare International Airport changes on rainy Saturdays

# import query data
loop_airport_rides = pd.read_csv('/datasets/project_sql_result_07.csv')

# drop Nans
loop_airport_rides = loop_airport_rides.dropna()

# rainy Saturday data (180 rows)
rainy_rides = loop_airport_rides[loop_airport_rides['weather_conditions'] == 'Bad']

# calculate average duration of rainy Saturday rides
avgRainyRideDuration = rainy_rides['duration_seconds'].mean()
avgRainyRideDuration = (avgRainyRideDuration/60).round(decimals=2)
print('The average duration of rainy Saturday rides is: ' + str(avgRainyRideDuration) + " minutes")

# non-rainy Saturday data (888 rows)
nonrainy_rides = loop_airport_rides[loop_airport_rides['weather_conditions'] == 'Good']

# calculate average duration of non rainy Saturday rides
avgNonRainyRideDuration = nonrainy_rides['duration_seconds'].mean()
avgNonRainyRideDuration = (avgNonRainyRideDuration/60).round(decimals=2)
print('The average duration of non-rainy Saturday rides is: ' + str(avgNonRainyRideDuration) + " minutes")

# perform a t-test
results = st.ttest_ind(rainy_rides['duration_seconds'], nonrainy_rides['duration_seconds'], equal_var=False)
p_value = results.pvalue
alpha = 0.05

if p_value < alpha:
    print('Reject H0')
else:
    print('Cannot reject H0')
