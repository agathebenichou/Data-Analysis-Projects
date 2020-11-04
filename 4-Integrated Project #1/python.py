#!/usr/bin/env python
# coding: utf-8

#Open data file and study general information

get_ipython().system('pip install squarify')
import squarify
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

# Open data files
data = pd.read_csv('/datasets/games.csv')

# Study general information
data.info()

# Data Preprocessing

# Make the column names lowercase
data.columns = data.columns.str.lower()

#genre column: convert to string type, Drop rows with 'nan' values
data['genre'] = data['genre'].astype(str)
data.drop(data[data['genre'] == 'nan'].index,inplace=True)
            
# dropped rows with no sales
data.drop(data[(data['na_sales'] == 0) & (data['eu_sales'] == 0) & (data['jp_sales'] == 0) & (data['other_sales'] == 0)].index,inplace=True)

# Drop any duplicates in the database
data.drop_duplicates(inplace=True)

# year_of_release column: Replace rows 'nan' values in the year_of_release column based on if the game already exists 
data['year_of_release'] = data['year_of_release'].fillna(value=0)
missingYearRows = data[data['year_of_release'] == 0.0]
missingYearRows['index'] =  missingYearRows.index.tolist()
for index,row in missingYearRows.iterrows():
    name = row['name']
    year = 0
    originalIndex = row['index']
    matchingNameRows = data.query('name == @name')
    for index1, row1 in matchingNameRows.iterrows():
        if row1['name'] is not 0.0:
            year = row1['year_of_release']
            data.loc[originalIndex,'year_of_release'] = year
            break

# Drop rows whose year_of_release wasn't filled from table
data.drop(data[data['year_of_release'] == 0.0].index, inplace=True)
data.reset_index()

# 'critic_score' column: replace relevant nan with average of critic scores of the same games
data['critic_score'] = data.groupby('name')['critic_score'].transform(lambda grp: grp.fillna(np.mean(grp)))

# 'user_score' column: fill 'tbd' values with nan, replace relevant nan with average of user scores of the same games
data['user_score'] = data['user_score'].replace('tbd',None)
data['user_score'] = data['user_score'].astype(float)
data['user_score'] = data.groupby('name')['user_score'].transform(lambda grp: grp.fillna(np.mean(grp)))

# 'rating' column: fill null values with most frequenctly occuring rating in its genre
data['rating'] = data.groupby('genre')['rating'].transform(lambda grp: grp.fillna(grp.mode().iloc[0]))

# Calculate the sum of sales in all regions for each game and put these values in a separate column
data['total_sales'] = data[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)
display(data)

# Look at how many games were released in different years. Is the data for every period significant?

yearGames = data[['year_of_release','name']].groupby('year_of_release').count().sort_values(by='year_of_release').reset_index()
yearGames.columns = ['year_of_release','total_games']
yearGames = yearGames[yearGames['year_of_release'] != 0]
#display(yearGames)

fig, ax = plt.subplots(figsize=(17,10))
ax.vlines(x=yearGames.year_of_release, ymin=0, ymax=yearGames.total_games,color='purple',alpha=0.7,linewidth=1)
ax.scatter(x=yearGames.year_of_release,y=yearGames.total_games, s=75, color='black',alpha=0.7)

ax.set_title("Released Games per Year", fontdict={'size':15})
ax.set_ylabel('Number of Released Games')
ax.set_xticks(yearGames.year_of_release)
ax.set_xticklabels(yearGames.year_of_release, rotation=90, fontdict={'horizontalalignment':'right','size':12})
for row in yearGames.itertuples():
    ax.text(row.year_of_release, row.total_games+30,s=round(row.total_games,2))


# Look at how sales varied from platform to platform. 
salesPlatform = data[['platform','total_sales']].groupby('platform').sum().sort_values(by='total_sales').reset_index()
salesPlatform['sales_zscore'] = (salesPlatform['total_sales'] - salesPlatform['total_sales'].mean())/ salesPlatform['total_sales'].std()
salesPlatform['color'] = ['red' if x<0 else 'green' for x in salesPlatform['sales_zscore']]

plt.figure(figsize=(14,10))
plt.hlines(y=salesPlatform.platform, xmin=0, xmax=salesPlatform.sales_zscore, colors=salesPlatform.color, alpha=0.4, linewidth=10)

# Choose the platforms with the greatest total sales and build a distribution based on data for each year. 
greatestTotalSalesPlatforms = salesPlatform.sort_values(by='total_sales', ascending=False).head(6)['platform'].tolist()
greatestTotalSales = data.query('platform in @greatestTotalSalesPlatforms')[['platform','year_of_release','total_sales']]
greatestTotalSales = pd.pivot_table(greatestTotalSales, values=['total_sales'],index=['platform','year_of_release'],aggfunc='sum').reset_index()
greatestTotalSales = greatestTotalSales[greatestTotalSales['year_of_release']!=0]

plt.figure(figsize=(16,10), dpi= 80)
colors = ['red','orange','yellow','green','blue','purple']
i=0
for platform in greatestTotalSalesPlatforms:
    platData = greatestTotalSales.loc[greatestTotalSales['platform'] == platform, 'year_of_release']
    sns.kdeplot(platData, shade=False , color=colors[i], label=platform, alpha=0.7)
    i += 1

plt.title('Distribution of Total Sales of each platform for each year', fontsize=22)
plt.legend()
plt.show()

# Identify platforms that used to be popular but now have 0 sales

# Find platforms that used to be popular but now have zero sales. 
lowestTotalSalesPlatforms = salesPlatform[(salesPlatform['sales_zscore'] < 0) & (salesPlatform['total_sales'] > 15.00)]['platform'].tolist()
lowestTotalSales = data.query('platform in @lowestTotalSalesPlatforms')[['platform','year_of_release','total_sales']]
lowestTotalSales = pd.pivot_table(lowestTotalSales, values=['total_sales'],index=['platform','year_of_release'],aggfunc='sum').reset_index()
lowestTotalSales = lowestTotalSales[(lowestTotalSales['year_of_release']!=0) & (lowestTotalSales['year_of_release'] < 2000)]

plt.figure(figsize=(14,8), dpi= 80)
colors = ['red','orange','yellow','green','blue','purple','pink','brown','black','magenta','gray','deeppink']
i=0
for platform in lowestTotalSales['platform'].unique():
    platData = lowestTotalSales.loc[lowestTotalSales['platform'] == platform, 'year_of_release']
    sns.kdeplot(platData, shade=False , color=colors[i], label=platform, alpha=0.7)
    i += 1

plt.title('Distribution of platforms that used to be popular', fontsize=22)
plt.legend()
plt.show()

# How long does it take for new platforms to appear and old ones to fade?

# How long does it generally take for new platforms to appear and old ones to fade?
fadeTimePerPlatform = (lowestTotalSales.groupby('platform')['year_of_release'].max() - lowestTotalSales.groupby('platform')['year_of_release'].min()).reset_index()
averageFadeTime = fadeTimePerPlatform['year_of_release'].mean().round(decimals=1)
print(averageFadeTime)

# Determine what period to take data from.
# Work only with the data that you've decided is relevant. 
data = data[data['year_of_release'] >= 2005]

# Determine which platforms are leading in sales.

# Which platforms are leading in sales? 
salesData = data[['platform','total_sales']].groupby('platform').sum().sort_values(by='total_sales').reset_index()
salesData = salesData[salesData['total_sales'] != 0]
sizes = salesData.total_sales.values.tolist()
labels = salesData.apply(lambda x: str(x[0]) + "\n" + "$" + str(round(x[1])),axis=1)
plt.figure(figsize=(15,9))
squarify.plot(sizes=sizes,label=labels,alpha=0.5)

plt.title('Distribution of Platform Market', fontsize=22)

# Determine which platforms are growing and which are shrinking.

# Which ones are growing or shrinking? Select several potentially profitable platforms.
growData = pd.pivot_table(data, index='year_of_release', columns='platform',values='total_sales',aggfunc='sum',fill_value=0)
dynamics = growData - growData.shift(+1)
#display(dynamics)
plt.figure(figsize=(13,9))
sns.heatmap(dynamics.T,cmap='RdBu_r')


# Build a box plot for the global sales of all games, broken down by platform. 
# Are the differences in sales significant? What about average sales on various platforms? 

globalSales = data.groupby(['platform','year_of_release'])['total_sales'].sum().reset_index()
ordered = globalSales.groupby(['platform'])['total_sales'].sum().sort_values().reset_index()['platform']
plt.figure(figsize=(13,10))
sns.boxplot(x='platform',y='total_sales',data=globalSales,order=ordered)

# Examine the correlation between user / critic reviews and sales.

# Take a look at how user and professional reviews affect sales for one popular platform (you choose). 
x360Data = data[data['platform'] == 'X360']

# Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.
x360DataCritic = x360Data.groupby(['critic_score'])['total_sales'].sum().reset_index()
axCritic = x360DataCritic.plot.scatter(x='critic_score', y='total_sales', ylim=(-1,100),figsize=(7,5))
axCritic.set_title('Critic Score effect on Total Sales (X360)')
criticCorrelation = data['critic_score'].corr(data['total_sales'])
print('Correlation between critic score and total sales for X360 is: ' + str(criticCorrelation.round(decimals=2)))

x360DataUser = x360Data.groupby(['user_score'])['total_sales'].sum().reset_index()
axUser = x360DataUser.plot.scatter(x='user_score', y='total_sales',figsize=(7,5))
axUser.set_title('User Score effect on Total Sales (X360)')
userCorrelation = data['user_score'].corr(data['total_sales'])
print('Correlation between user score and total sales for X360 is: ' + str(userCorrelation.round(decimals=2)))

# Compare the sales of the same games on other platforms.
# Compare the sales of the same games on other platforms.
x360Games = x360Data['name'].unique().tolist()

sameGamesOtherPlatforms = data.query('platform != "X360" and name in @x360Games')
sameGamesOtherPlatforms = sameGamesOtherPlatforms.query('platform in ("PS2","PS3","PS4","Wii")')

sameGamesOtherPlatformsCritic = sameGamesOtherPlatforms.groupby(['platform','critic_score'])['total_sales'].sum().reset_index()
fig, axCritic = plt.subplots(figsize=(10,8))
platforms = ["PS2","PS3","PS4","Wii"]
colors = ['r','y','g','b']
i =0
for p in platforms:
    d = sameGamesOtherPlatformsCritic[sameGamesOtherPlatformsCritic['platform'] == p]
    axCritic.scatter(d['critic_score'],d['total_sales'],c=colors[i],label=p)
    i+=1
axCritic.legend(loc='upper left')
axCritic.set_title("Critic Score effect on Total Sales (PS2,PS3,PS4,Wii)")
axCritic.set_xlabel('Critic Score')
axCritic.set_ylabel('Total Sales')
criticCorrelation = sameGamesOtherPlatformsCritic['critic_score'].corr(sameGamesOtherPlatformsCritic['total_sales'])
print('Average correlation between critic score and total sales for (PS2,PS3,PS4,Wii) is: ' + str(criticCorrelation.round(decimals=2)))

sameGamesOtherPlatformsUser = sameGamesOtherPlatforms.groupby(['platform','user_score'])['total_sales'].sum().reset_index()
fig, axUser = plt.subplots(figsize=(10,8))
platforms = ["PS2","PS3","PS4","Wii"]
colors = ['r','y','g','b']
i =0
for p in platforms:
    d = sameGamesOtherPlatformsUser[sameGamesOtherPlatformsUser['platform'] == p]
    axUser.scatter(d['user_score'],d['total_sales'],c=colors[i],label=p)
    i+=1
axUser.legend(loc='upper left')
axCritic.set_title("User Score effect on Total Sales (PS2,PS3,PS4,Wii)")
axUser.set_xlabel('User Score')
axUser.set_ylabel('Total Sales')
criticCorrelation = sameGamesOtherPlatformsUser['user_score'].corr(sameGamesOtherPlatformsUser['total_sales'])
print('Average correlation between user score and total sales for (PS2,PS3,PS4,Wii) is: ' + str(criticCorrelation.round(decimals=2)))

# Examine the general distribution of games by genre. 

# Take a look at the general distribution of games by genre. 
# What can we say about the most profitable genres? 

genreData = data[['genre','total_sales']].groupby(['genre']).sum().sort_values(by='total_sales',ascending=False).reset_index()
fig, ax = plt.subplots(figsize=(17,10))
ax.vlines(x=genreData.genre, ymin=0, ymax=genreData.total_sales,color='purple',alpha=0.7,linewidth=1)
ax.scatter(x=genreData.genre,y=genreData.total_sales, s=75, color='black',alpha=0.7)

ax.set_title("Total Sales per Genre", fontdict={'size':15})
ax.set_ylabel('Total Sales')
ax.set_xlabel('Genre Type')
ax.set_xticks(genreData.genre)
ax.set_xticklabels(genreData.genre, rotation=90, fontdict={'horizontalalignment':'right','size':12})
for row in genreData.itertuples():
    ax.text(row.genre, row.total_sales+30,s=round(row.total_sales,2))

#Examine genres with high and low sales on 3 platforms.
# What can you generalize about genres with high and low sales?
profitablePlatforms = ['X360','PS3','Wii']
genrePlatform = data.query('platform in @profitablePlatforms')
genrePlatform = genrePlatform[['platform','genre','total_sales']].groupby(['platform','genre']).sum().sort_values(by='genre').reset_index()
genreList = genrePlatform['genre'].unique().tolist()

x360 = []
ps3 = []
wii = []
for genre in genreList:
    x360.append(genrePlatform[(genrePlatform['platform'] == "X360") & (genrePlatform['genre'] == genre)]['total_sales'].values[0].round(decimals=1))
    ps3.append(genrePlatform[(genrePlatform['platform'] == "PS3") & (genrePlatform['genre'] == genre)]['total_sales'].values[0].round(decimals=1))
    wii.append(genrePlatform[(genrePlatform['platform'] == "Wii") & (genrePlatform['genre'] == genre)]['total_sales'].values[0].round(decimals=1))
display(x360)

plotdata = pd.DataFrame({
    "X360":x360,
    "PS3":ps3,
    "Wii":wii
    }, 
    index=genreList
)
plotdata.plot(kind="bar")
plt.title("Total Sales Per Genre for (X360,PS3,Wii)")
plt.xlabel("Genre")
plt.ylabel("Total Sales")

#  Examine the top 5 platforms per region (EU, NA, JP)
#For each region (NA, EU, JP), determine:

naData = data[['platform','genre','rating','na_sales']]
jpData = data[['platform','genre','rating','jp_sales']]
euData =  data[['platform','genre','rating','eu_sales']]

#The top five platforms. Describe variations in their market shares from region to region
naPlatforms = naData.groupby('platform').sum().sort_values(by='na_sales',ascending=False)
jpPlatforms = jpData.groupby('platform').sum().sort_values(by='jp_sales',ascending=False)
euPlatforms = euData.groupby('platform').sum().sort_values(by='eu_sales',ascending=False)
totalPlatforms = np.unique(np.array(naPlatforms.reset_index().head(5)['platform'].tolist()+jpPlatforms.reset_index().head(5)['platform'].tolist()+euPlatforms.reset_index().head(5)['platform'].tolist()))

regionPlatforms = pd.merge(naPlatforms, jpPlatforms, how='inner', on='platform')
regionPlatforms = pd.merge(regionPlatforms, euPlatforms, how='inner', on='platform')
regionPlatforms = regionPlatforms.query('platform in @totalPlatforms')
regionPlatforms.plot(kind='bar',stacked=True, figsize=(8,6))

regionPlatforms.plot(kind='bar',stacked=False, figsize=(8,6))

# Examine the top 5 genres per region (EU, NA, JP)
# For each region (NA, EU, JP), determine: The top five genres. Explain the difference.
naGenres = naData.groupby('genre').sum().sort_values(by='na_sales',ascending=False)
jpGenres = jpData.groupby('genre').sum().sort_values(by='jp_sales',ascending=False)
euGenres = euData.groupby('genre').sum().sort_values(by='eu_sales',ascending=False)
totalGenres = np.unique(np.array(naGenres.reset_index().head(5)['genre'].tolist()+jpGenres.reset_index().head(5)['genre'].tolist()+euGenres.reset_index().head(5)['genre'].tolist()))

regionGenres = pd.merge(naGenres, jpGenres, how='inner', on='genre')
regionGenres = pd.merge(regionGenres, euGenres, how='inner', on='genre')
regionGenres = regionGenres.query('genre in @totalGenres')
regionGenres.plot(kind='bar',stacked=True, figsize=(8,6))

#  Examine whether ratings affect sales in each region.

# For each region (NA, EU, JP), Do ESRB ratings affect sales in individual regions?
naRatings = naData.groupby('rating').sum().sort_values(by='na_sales',ascending=False)
jpRatings = jpData.groupby('rating').sum().sort_values(by='jp_sales',ascending=False)
euRatings = euData.groupby('rating').sum().sort_values(by='eu_sales',ascending=False)
totalRatings = np.unique(np.array(naRatings.reset_index().head(5)['rating'].tolist()+jpRatings.reset_index().head(5)['rating'].tolist()+euRatings.reset_index().head(5)['rating'].tolist()))

regionRatings = pd.merge(naRatings, jpRatings, how='inner', on='rating')
regionRatings = pd.merge(regionRatings, euRatings, how='inner', on='rating')
regionRatings = regionRatings.query('rating in @totalRatings')
regionRatings.plot(kind='bar',stacked=True, figsize=(8,6))

# Test hypothesis: Average user ratings of the Xbox One and PC platforms are the same.

# Query Xbox One user ratings
xboxOneData = data[data['platform'] == 'XOne']

# drop Nans
xboxOneData = xboxOneData[xboxOneData['user_score'].notna()]

# calculate average
xboxOneAverageUserRating = xboxOneData['user_score'].mean().round(decimals=2)
print('The average user rating of XBox One is: ' + str(xboxOneAverageUserRating) + "/10")

# query PC user rating
pcData = data[data['platform'] == 'PC']

# drop Nans
pcData = pcData[pcData['user_score'].notna()]

# calculate average
pcAverageUserRating = pcData['user_score'].mean().round(decimals=2)
print('The average user rating of PC is: ' + str(pcAverageUserRating) + "/10")

# perform a t-test
results = st.ttest_ind(xboxOneData['user_score'], pcData['user_score'], equal_var=False)
p_value = results.pvalue
alpha = 0.05

if p_value < alpha:
    print('Reject H0')
else:
    print('Cannot reject H0')

# Test hypothesis: Average user ratings for the Action and Sports genres are different.

# Query action genre user ratings
actionData = data[data['genre'] == 'Action']

# drop Nans
actionData = actionData[actionData['user_score'].notna()]

# calculate average
actionAverageUserRating = actionData['user_score'].mean().round(decimals=2)
print('The average user rating of Action genre is: ' + str(actionAverageUserRating) + "/10")

# query sport genre user rating
sportData = data[data['genre'] == 'Sports']

# drop Nans
sportData = sportData[sportData['user_score'].notna()]

# calculate average
sportAverageUserRating = sportData['user_score'].mean().round(decimals=2)
print('The average user rating of Sports genre is: ' + str(sportAverageUserRating) + "/10")

# perform a t-test
results = st.ttest_ind(actionData['user_score'], sportData['user_score'], equal_var=False)
p_value = results.pvalue
alpha = 0.05

if p_value < alpha:
    print('Reject H0')
else:
    print('Cannot reject H0')
