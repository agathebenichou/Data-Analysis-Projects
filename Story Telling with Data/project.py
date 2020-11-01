#!/usr/bin/env python
# coding: utf-8

# # Telling A Story Using Data
# Task: You’ve decided to open a small robot-run cafe in Los Angeles. The project is promising but expensive, so you and your partners decide to try to attract investors. They’re interested in the current market conditions—will you be able to maintain your success when the novelty of robot waiters wears off? Your partners have asked you to prepare some market research using open-source data on restaurants in LA.

# ### Load Data
get_ipython().system(' pip install -q usaddress')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st
import usaddress
import re


# load in data file 
data = pd.read_csv('/datasets/rest_data_us.csv', sep=',')

# study general information 
data.info()

display(data.head())

# ### Pre-process the data
# drop duplicates and NAN values
data = data.drop_duplicates()

# rename column names
data.columns = ['id', 'name', 'address', 'chain', 'type', 'total_seats']

# chain column - drop rows with NA values
data['chain'].dropna(inplace=True)

# name column - replace NaN and standardize names 

# drop name colums with empty strings
data.replace("", float("NaN"), inplace=True)
data.dropna(subset = ["name"], inplace=True)

# method to clean names
def cleanNames(name):
    name = name

    # if # symbol in name, split it to remove numbers
    if '#' in name:
        name = name.split('#')[0].strip()
    
    # if - symbol in name, split it to remove numbers
    if '-' in name:
        tmp = name.split('-')

        if tmp[1].isdecimal():
            name = tmp[0].strip()
    
    return name
    
# clean names to remove branch numbers
data['name'] = data.name.apply(cleanNames)

# method to clean addresses
def cleanAddress(address):
    address = address

    # hardcode singular cases
    if address.startswith('OLVERA'):
        address = 'OLVERA,Los Angeles,USA'
    elif address.startswith('1033 1/2 LOS ANGELES ST'):
        address = '1033 1/2 LOS ANGELES ST,Los Angeles, USA'
        
    # standard cases
    else:
        raw = usaddress.parse(address)
        addressDict = {}
        for i in raw:
            addressDict.update({i[1]:i[0]})
        
        if 'StreetNamePostType' in addressDict:
            address = addressDict['AddressNumber'] + " " + str(addressDict['StreetName']) +                 " " + str(addressDict['StreetNamePostType'])+str(',Los Angeles,USA')
        else:
            address = addressDict['AddressNumber'] + " " + str(addressDict['StreetName']) +                 " "+str(',Los Angeles,USA')
    
    return address
    
# clean names to remove branch numbers
data['address'] = data.address.apply(cleanAddress)


# ### Investigate the proportions of the various types of establishments. 
# transform value_counts for type of establishment to dataframe
typeData = data['type'].value_counts().rename_axis('type').reset_index(name='count')

# plot pie chart with proportions
plt.figure(figsize=(10, 5))
plt.pie(typeData['count'], labels=typeData['type'],autopct='%0.f%%', shadow=True, startangle=145)
plt.title('Types of Establishments')
plt.show()


# ### Investigate the proportions of chain and nonchain establishments.
# transform value_counts for chain establishment to dataframe
chainData = data['chain'].value_counts().rename_axis('chain').reset_index(name='count')

# plot pie chart with proportions
plt.pie(chainData['count'], labels=chainData['chain'],autopct='%0.f%%', shadow=True, startangle=145)
plt.title('Chain vs Non-Chain Establishments')


# ### Which type of establishment is typically a chain?
# pull out data from only chains
typeChainData = data[data['chain'] == True]

# count the number of types of establishments
typeChainData = pd.pivot_table(typeChainData,index=['chain','type'], values=['id'], aggfunc='count').reset_index()
typeChainData.columns = ['chain', 'type', 'count']
typeChainData = typeChainData.sort_values(by='count', ascending=False)

# extract total number of establishments
totalData = data['type'].value_counts().rename_axis('type').reset_index(name='total')

# combine data frames
typeChainData = pd.merge(typeChainData, totalData, how='inner', on='type')
typeChainData['ratio'] = typeChainData['count'] / typeChainData['total'] * 100

# plot bar graph
plt.figure(figsize=(10, 7))

ax = sns.barplot(data = typeChainData.sort_values('type', ascending=False), 
                 x='type', 
                 y='ratio')

# Add titles and captions
plt.xlabel('Type of Establishment')
plt.ylabel('Ratio of Chains')
plt.title('Ratio of Chains per Establishment')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')


plt.show()


# ### What characterizes chains: many establishments with a small number of seats or a few establishments with a lot of seats?
# extract data from only chains, sort by count
chainData = data[data['chain'] == True]
chainData = pd.pivot_table(chainData, index=['total_seats'], values=['id'], aggfunc=['count']).reset_index()
chainData.columns = ['total_seats','count']
chainData = chainData.sort_values(by='count', ascending=False)

# extract data from only nonchains, sort by count
nonChainData = data[data['chain'] == False]
nonChainData = pd.pivot_table(nonChainData, index=['total_seats'], values=['id'], aggfunc=['count']).reset_index()
nonChainData.columns = ['total_seats','count']
nonChainData = nonChainData.sort_values(by='count', ascending=False)

# plot scatter plot
plt.scatter(chainData['total_seats'], chainData['count'], color='r', label='chains')
plt.scatter(nonChainData['total_seats'], nonChainData['count'], color='g', label='non-chains')
plt.xlabel('Total Seats per Establishment')
plt.ylabel('Number of Establishments')
plt.legend()
plt.title('Correlation between # of seats and # of establishments')


# ### Determine the average number of seats for each type of establishment. On average, which type of establishment has the greatest number of seats? 
# bar graph labels and values
labels = ['Cafe', 'Restaurant', 'Fast Food', 'Pizza', 'Bar', 'Bakery']    
values = []

# for each establishment, extract data, calculate avg and append to list
for establishment in range(len(labels)):
    estType = labels[establishment]
    currData = data[data['type'] == estType]
    avgSeats = currData['total_seats'].mean()
    values.append(avgSeats)

# create df from labels and values
df = pd.DataFrame({"Establishment":labels, "Seats":values})

plt.figure(figsize=(10, 7))

# Plot barplot
ax = sns.barplot(data = df.sort_values('Seats', ascending=False), 
                 x='Establishment', 
                 y='Seats')

# Add titles and captions
plt.xlabel('Type of Establishment')
plt.ylabel('Average Number of Seats')
plt.title('Average Number of Seats per Establishment')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')


plt.show()


# ### Put the data on street names from the address column in a separate column.
# method to extract only street name
def streetNames(street):
    street = street.split(',')[0].split(' ')[-2:]
    street = " ".join(street)
    if street == '103 ':
        street = '103RD ST'
    return street
    
# extract only street names
data['street'] = data.address.apply(streetNames)
display(data)


# ### Plot a graph of the top ten streets by number of establishments.
# sort street data by number of occurrences
streetData = pd.pivot_table(data, index=['street'], values=['id'], aggfunc=['count']).reset_index()
streetData.columns = ['street','count']
topStreetData = streetData.sort_values(by='count', ascending=False).head(10)

# Plot barplot
plt.figure(figsize=(10, 7))

ax = sns.barplot(data = topStreetData.sort_values('count', ascending=False), 
                 x='street', 
                 y='count')

# Add titles and captions
plt.xlabel('Most Popular Streets')
plt.ylabel('Number of Establishments')
plt.title('Number of Establishments per Most Popular Streets')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')

plt.xticks(rotation=90)
plt.show()


# ### Find the number of streets that only have one restaurant.
# extract streets with a single restaurant
singleStreetData = streetData[streetData['count'] == 1]
numStreets = len(singleStreetData)
print('There are ' + str(numStreets) + " with only one restaurant on them.")


# ### For streets with a lot of establishments, look at the distribution of the number of seats. What trends can you see?
# calculate avg number of restaurants per street, extract only those greater than avg
avgNumRestaurants = streetData['count'].mean().round()
avgStreetData = streetData[streetData['count'] > avgNumRestaurants]

# plot distribution plot
ax = sns.distplot(avgStreetData['count'])
ax.set_title('Distribution of the Number of Seats')
ax.set_xlabel('Number of Seats')
ax.set_ylabel('Density of Streets')
