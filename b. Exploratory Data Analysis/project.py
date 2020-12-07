#!/usr/bin/env python
# coding: utf-8

#  Step 1. Open the data file and study the general information. 

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('/datasets/real_estate_data_us.csv', sep='\t')
real_estate = pd.DataFrame(data=data)
print(real_estate.info())

#  Step 2. Data preprocessing

# airport_dist column - filled in NA values with the mean, convert to float data type
real_estate['airport_dist'].fillna(real_estate['airport_dist'].mean(), inplace=True)
real_estate['airport_dist'] = pd.to_numeric(real_estate['airport_dist']).astype(float)

# balconies column - replace NaN values with 0, converted to int data type
real_estate['balconies'].fillna(0, inplace=True)
real_estate['balconies'] = pd.to_numeric(real_estate['balconies']).astype(int)

# ceiling_height column - replace NaN values with the mean, round values to 2 decimal points
real_estate['ceiling_height'].fillna(real_estate['ceiling_height'].mean(), inplace=True)
real_estate['ceiling_height'] = pd.to_numeric(real_estate['ceiling_height']).round(decimals=2)

# city_center_dist column - replace NaN values with mean, round values to 2 decimal points
real_estate['city_center_dist'].fillna(real_estate['city_center_dist'].mean(), inplace=True)
real_estate['city_center_dist'] = pd.to_numeric(real_estate['city_center_dist']).round(decimals=2)

# days_listed column - replace NaN values with 0, convert to int data type
real_estate['days_listed'].fillna(0, inplace=True)
real_estate['days_listed'] = pd.to_numeric(real_estate['days_listed']).astype(int)

# date_posted column - convert datetime values to just the date (time not needed)
real_estate['date_posted'] = pd.to_datetime(real_estate['date_posted']).dt.date

# floor column - nothing needed

# floors_total column - replace NaN values with 0, converted to int data type
real_estate['floors_total'].fillna(0, inplace=True)
real_estate['floors_total'] = pd.to_numeric(real_estate['floors_total']).astype(int)

# bike_parking column - replace NaN values with False
real_estate['bike_parking'].fillna(False, inplace=True)

# kitchen_area column - replace NaN values with mean, round values to 2 decimal points
real_estate['kitchen_area'].fillna(real_estate['kitchen_area'].mean(), inplace=True)
real_estate['kitchen_area'] = pd.to_numeric(real_estate['kitchen_area']).round(decimals=2)

# last_price column - convert to int data type
real_estate['last_price'] = pd.to_numeric(real_estate['last_price']).astype(int)

# living_area column - replace NaN values with mean, round values to 2 decimal points 
real_estate['living_area'].fillna(real_estate['living_area'].mean(), inplace=True)
real_estate['living_area'] = pd.to_numeric(real_estate['living_area']).round(decimals=2)

# locality_name column - lowercase the strings, remove any 'village', drop NaN vaalues
real_estate['locality_name'] = real_estate['locality_name'].str.lower()
real_estate['locality_name'] = real_estate['locality_name'].str.replace(' village','')
real_estate['locality_name'] = real_estate['locality_name'].str.replace('village ','')
real_estate['locality_name'].dropna(inplace=True)

# is_open_plan column - nothing to fix

# parks_within_3000 column - convert to int data type, replace NaN values with 0
real_estate['parks_within_3000'].fillna(0, inplace=True)
real_estate['parks_within_3000'] = pd.to_numeric(real_estate['parks_within_3000']).astype(int)

# park_dist column - if park_dist is null but it has parks within 3000m, fill in mean
# if park_dist is null and it has no parks within 3000m, fill in 0
filterParks = real_estate['park_dist'].isna() & real_estate['parks_within_3000'] >= 1
real_estate['park_dist'].where(filterParks).fillna(real_estate['park_dist'].mean(),inplace=True)
filterParks = real_estate['park_dist'].isna() & real_estate['parks_within_3000'] == 0
real_estate['park_dist'].where(filterParks).fillna(0,inplace=True)

# ponds_within_3000 column - convert to int data type, replace NaN values with 0
real_estate['ponds_within_3000'].fillna(0, inplace=True)
real_estate['ponds_within_3000'] = pd.to_numeric(real_estate['ponds_within_3000']).astype(int)

# pond_dist column - if pond_dist is null but it has ponds within 3000m, fill in mean
# if pond_dist is null and it has no ponds within 3000m, fill in 0
filterParks = real_estate['pond_dist'].isna() & real_estate['ponds_within_3000'] >= 1
real_estate['pond_dist'].where(filterParks).fillna(real_estate['pond_dist'].mean(),inplace=True)
filterParks = real_estate['pond_dist'].isna() & real_estate['ponds_within_3000'] == 0
real_estate['pond_dist'].where(filterParks).fillna(0,inplace=True)

# bedrooms column - nothing to fix
# is_studio column - nothing to fix
# total_area column - nothing to fix
# total_images column - nothing to fix

#print(real_estate['total_images'].unique())
#print(real_estate['total_images'].value_counts())
#print(real_estate['total_images'].isna().sum())
#print(real_estate[real_estate['city_center_dist'].isnull()].count())
#print(real_estate.duplicated().sum())

#  Step 3. Make calculations and add them to the table

# the price per square meter = last_price column / total_area column
real_estate['price_per_sqm'] = (real_estate['last_price'] / real_estate['total_area']).round(decimals=2)

# the day of the week, month, and year that the ad was published
real_estate['weekday_posted'] = pd.to_datetime(real_estate['date_posted']).dt.day
real_estate['month_posted'] = pd.to_datetime(real_estate['date_posted']).dt.month
real_estate['year_posted'] = pd.to_datetime(real_estate['date_posted']).dt.year

# which floor the apartment is on (first, last, or other) - 
def determineFloor(row):
    floor = row['floor']
    totalFloors = row['floors_total']
    if totalFloors == 0 or floor == 1:
        return 'first'
    if (floor/totalFloors) <  1:
        return 'other'
    if (floor/totalFloors) == 1:
        return 'last'
real_estate['floor_category'] = real_estate.apply(determineFloor, axis=1)
 
# the ratio between the living space and the total area
real_estate['living_ratio'] = real_estate['living_area'] / real_estate['total_area']
real_estate['living_ratio'] = pd.to_numeric(real_estate['living_ratio']).round(decimals=2)

# the ratio between the kitchen  space and the total area
real_estate['kitchen_ratio'] = real_estate['kitchen_area'] / real_estate['total_area']
real_estate['kitchen_ratio'] = pd.to_numeric(real_estate['kitchen_ratio']).round(decimals=2)

#  Step 4. Conduct exploratory data analysis and follow the instructions below:

# for square area, price, number of rooms, and ceiling height, plot a histogram for each
real_estate.hist('last_price', range=[0,1000000], bins=100)
real_estate.hist('ceiling_height', range=[2,4.5], bins=30)
real_estate.hist('total_area', range=[0,350], bins=100)
real_estate.hist('bedrooms', range=[0,10], bins=10)

# Examine the time it's taken to sell the apartment and plot a histogram. 
# Calculate the mean and median and explain the average time it usually takes to complete a sale. 
# When can a sale be considered to have happened rather quickly or taken an extra long time?
print("Mean Sale Time: " + str(real_estate['days_listed'].mean().round(decimals=2)) + " days")  # 157 days is average time it takes
print("Median Sale Time: " + str(real_estate['days_listed'].median()) + " days") # 74 days is middle value
countBeforeDrop = real_estate['days_listed'].count()
#print(real_estate['days_listed'].quantile(0.75))

#Remove rare and outlying values and describe the patterns you've discovered.
# Was originally dropped Q3 but that dropped 24% of data so altered the drop value a bit
indexNames = real_estate[real_estate['days_listed'] > 430 ].index 
real_estate.drop(indexNames, inplace=True)
countAfterDrop = real_estate['days_listed'].count()
percentDropped = (((countBeforeDrop - countAfterDrop) / countBeforeDrop) * 100).astype(int)
print(str(percentDropped) + "% of data was dropped")
real_estate.boxplot('days_listed')

'''
The mean (average) of the time it takes to sell an listing (days_listed column) is 181 days and the 
median (most frequenctly occuring value) of the time it takes to sell a listening is 95 days. 
A sale can be considered to have happened rather quickly when it is less than or equal to the median, 
74 days. A sale can be considered to have taken extra long time when it is greater than or equal to the Q3, 199 days. 
To remove rare and outlying values, I identified the indices of the values who are above a certain threshold 
(which I identified as 430) and dropped those rows from the dataset - thereby eliminating outliers. By dropping 
values at this threshold, I removed 9% of the data. 
''' 

# Which factors have had the biggest influence on an apartment’s price? 
# Examine whether the value depends on the total square area, # of rooms, floor type, or proximity to city center. 

print("Significant Correlation between last_price and total_area is: " + str(real_estate['last_price'].corr(real_estate['total_area']).round(decimals=2)))
print("Weak Correlation between last_price and bedrooms is: " + str(real_estate['last_price'].corr(real_estate['bedrooms']).round(decimals=2)))
print("Negative Correlation between last_price and city_center_dist is: " + str(real_estate['last_price'].corr(real_estate['city_center_dist']).round(decimals=2)))
print("No significant Correlation between last_price and floor is: " + str(real_estate['last_price'].corr(real_estate['floor']).round(decimals=2)))

real_estate.plot(kind='scatter', x='last_price', y='total_area', xlim=[0,1500000])
real_estate.plot(kind='scatter', x='last_price', y='bedrooms', xlim=[0,1500000])
real_estate.plot(kind='scatter', x='last_price', y='city_center_dist', xlim=[0,1500000])
real_estate.plot(kind='scatter', x='last_price', y='floor', xlim=[0,1500000])

# Also check whether the publication date has any effect on the price: specifically, day of the week, month, and year. 
print("No significant Correlation between last_price and weekday_posted is: " + str(real_estate['last_price'].corr(real_estate['weekday_posted']).round(decimals=2)))
print("No significant Correlation between last_price and month_posted is: " + str(real_estate['last_price'].corr(real_estate['month_posted']).round(decimals=2)))
print("No significant Correlation between last_price and year_posted is: " + str(real_estate['last_price'].corr(real_estate['year_posted']).round(decimals=2)))

real_estate.plot(kind='scatter', x='last_price', y='weekday_posted', xlim=[0,1500000])
real_estate.plot(kind='scatter', x='last_price', y='month_posted', xlim=[0,1500000])
real_estate.plot(kind='scatter', x='last_price', y='year_posted', xlim=[0,1500000])

#print(real_estate['total_images'].unique())
#print(real_estate['total_images'].value_counts())
#print(real_estate['total_images'].isna().sum())
#print(real_estate[real_estate['city_center_dist'].isnull()].count())
#print(real_estate.duplicated().sum())

# Select the 10 localities with the largest # of ads
topLocalities = real_estate.groupby('locality_name').count().sort_values(by='date_posted',ascending=False).head(10)
topLocalities.reset_index(inplace=True)
topLocalities = topLocalities['locality_name'].tolist()
print('The top localities with the largest # of ads are: ' + str(topLocalities))

# calculate the average price/square meter in these localities.
topLocalityData = real_estate[real_estate['locality_name'].isin(topLocalities)].reset_index()
topLocalityData['avg_price_per_sqm'] =  (topLocalityData['last_price'] / topLocalityData['total_area']).round(decimals=2)

# Determine which ones have the highest and lowest housing prices. 
priceRanges = topLocalityData.groupby('locality_name')['avg_price_per_sqm'].mean().sort_values(ascending=False)
print(priceRanges)

'''
To select the top 10 frequently occurring localities in the data, I created a subset by grouping the data by the
locality name, counted the number of rows each locality has, sorted it in descending order and returned only the 
top 10. I reset the indices so that the subset will be indexed by row number and not locality name and I converted 
the DataFrame subset to a list. 
To calculate the average price per square meter within these localities, I queried the original DataFrame using the 
list of top 10  localities so that only data relating to listings within these top 10 localities are added and then I 
divided the last_price column by the total_area column. 
To calculate which of the top 10 localities have the highest and lowest housing prices, I grouped them by locality 
and calculated the mean over the avg_price_per_sqm column for each grouped locality. As a result, Saint Petersburg has 
the highest housing prices and Vyborg has the lowest housing prices, out of the top 10 localities. 
'''

# Each apartment has info  about the distance to the city center. Select apartments in Saint Petersburg (‘locality_name’). 
saintPetersburg = real_estate.query('locality_name == "saint petersburg"').reset_index()

# Your task is to pinpoint which area is considered to be in the city center. 
# In order to do that, create a column with the distance to the city center in km and round to the nearest whole number. 
saintPetersburg['city_center_dist_km'] = (saintPetersburg['city_center_dist'] / 1000).round()

# Next, calculate the average price for each kilometer
saintPetersburgPrices = saintPetersburg.groupby('city_center_dist_km')['price_per_sqm'].mean().round(decimals=2).reset_index()
#print(saintPetersburgPrices)

# Plot a graph to display how prices are affected by the distance to the city center
saintPetersburgPrices.plot.line(x='city_center_dist_km', y='price_per_sqm')

# Find a place on the graph where it shifts significantly. That's the city center border.
print('The city center border is at 2km')

# Select all the apts in city center
cityCenter = saintPetersburg.query('city_center_dist_km <= 2.0').reset_index()

# Examine correlations between the parameters: total area, price, number of rooms, ceiling height. 
cityCenterSub = cityCenter[['total_area', 'bedrooms', 'last_price','ceiling_height']]
print(cityCenterSub.corr())
print('There is a significant correlation between total area and bedroom, as well as total area and last price. ')

# Identify the factors that affect an apt’s price: number of rooms, floor, distance to the city center, ad publication date.
print("No significant correlation between last_price and bedrooms is: " + str(cityCenter['last_price'].corr(cityCenter['bedrooms']).round(decimals=2)))
print("No significant correlation between last_price and floor is: " + str(cityCenter['last_price'].corr(cityCenter['floor']).round(decimals=2)))
print("No significant correlation between last_price and city_center_dist_km is: " + str(cityCenter['last_price'].corr(cityCenter['city_center_dist_km']).round(decimals=2)))
#print("No significant correlation between last_price and date_posted is: " + str(cityCenter['last_price'].corr(cityCenter['date_posted']).round(decimals=2)))

cityCenter.plot(kind='scatter', x='bedrooms', y='last_price', ylim=[0,1500000])
cityCenter.plot(kind='scatter', x='floor', y='last_price', ylim=[0,1500000])
cityCenter.plot(kind='scatter', x='city_center_dist_km', y='last_price', ylim=[0,1500000])
#cityCenter.plot(kind='scatter', x='last_price', y='date_posted')