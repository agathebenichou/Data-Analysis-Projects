#!/usr/bin/env python
# coding: utf-8

pip install seaborn --upgrade


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

# ## Download the data
# read in the data and load into DataFrame
data = pd.read_csv('/datasets/gym_churn_us.csv')

# rename columns to be more legible 
data.columns = ['gender', 'near_location', 'employer_partner', 'friend_promo', 'phone_exists', 'total_contract_period_months', 
               'group_sessions', 'age', 'other_services_total_dollars', 'contract_remaining_months', 'lifetime_months',
               'avg_visits_per_week_total', 'avg_visits_per_week_last_month', 'churn']

display(data)
print(data.info())

# ### Does it contain any missing features? Study the mean values and standard deviation 
# Find and deal with missing values
display(data.describe())

# ### Look at the mean feature values in two groups: for those who left and for those who stayed.
churn_data = data.groupby('churn').mean()
display(churn_data)

# ### Plot bar histograms and feature distributions for those who left and those who stayed.

# define churned and not churned dataframes
user_churned = data[data['churn'] == 1]
user_not_churned = data[data['churn'] == 0]

# method to label bar graphs
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# columns with values of 0 or 1 = BAR HISTOGRAM
binary_cols = ['gender','near_location', 'employer_partner', 'friend_promo', 'phone_exists', 'group_sessions']

for col in binary_cols:
    
    # create subplit
    fig = plt.figure(figsize=(5,5))
    ax = fig.subplots()
    
    # define x axis labels based on feature
    if col == 'gender':
        x = ['Male', 'Female']
        labels = ['Male', 'Female']
        
    if col == 'near_location' or col == 'employer_partner' or col == 'friend_promo' or col == 'phone_exists' or col == 'group_sessions':
        x = ['Yes', 'No']
        labels = ['Yes', 'No']
        
    # create array of churned data
    y_churned = [user_churned[user_churned[col] == 1][col].count(),
           user_churned[user_churned[col] == 0][col].count()]

    # create array of unchurned data
    y_not_churned = [user_not_churned[user_not_churned[col] == 1][col].count(),
           user_not_churned[user_not_churned[col] == 0][col].count()]
    
    # the label locations
    x = np.arange(len(labels)) 
    
    # the width of the bars
    width = 0.35 

    # plot bar graphs
    rects1 = ax.bar(x - width/2, y_churned, width, color = 'red', label='Churn')
    rects2 = ax.bar(x + width/2, y_not_churned, width, color='blue',label='No Churn')

    # edit labels and formalities
    ax.set_ylabel('Count')
    ax.set_title(col)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

    # add number
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()

# columns with numerical values = FEATURE DISTRIBUTIONS
cols = ['total_contract_period_months', 'age', 'other_services_total_dollars', 'contract_remaining_months', 
        'lifetime_months', 'avg_visits_per_week_total', 'avg_visits_per_week_last_month']

fig = make_subplots(rows=7, cols=1, subplot_titles=('Age',
                                                    'Total Contract Period (Months)',
                                                    'Contract Remaining (Months)',
                                                    'Lifetime (Months)', 
                                                    'Other Services Total (Dollars)',
                                                    'Average Visits Per Week (Total)',
                                                    'Average Visits Per Week (Last Month)'
                                                   ))
# age
fig.add_trace(go.Histogram(
    x = user_churned['age'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=True),row=1, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['age'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='blue',
    showlegend=True),row=1, col=1)

# total_contract_period_months
fig.add_trace(go.Bar(
    x=[1,6,12],
    y=[user_churned[user_churned['total_contract_period_months'] == 1]['total_contract_period_months'].count(),
       user_churned[user_churned['total_contract_period_months'] == 6]['total_contract_period_months'].count(),
       user_churned[user_churned['total_contract_period_months'] == 12]['total_contract_period_months'].count()],
    name='Churn',
    marker_color='red', showlegend=False), row=2, col=1)
fig.add_trace(go.Bar(
    x=[1,6,12],
    y=[user_not_churned[user_not_churned['total_contract_period_months'] == 1]['total_contract_period_months'].count(),
       user_not_churned[user_not_churned['total_contract_period_months'] == 6]['total_contract_period_months'].count(),
       user_not_churned[user_not_churned['total_contract_period_months'] == 12]['total_contract_period_months'].count()],
    name='No churn',
    marker_color='blue', showlegend=False), row=2, col=1)

# contract_remaining_months
fig.add_trace(go.Bar(
    x=[1,2,3,4,5,6,7,8,9,10,11,12],
    y=[user_churned[user_churned['contract_remaining_months'] == 1]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 2]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 3]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 4]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 5]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 6]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 7]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 8]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 9]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 10]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 11]['contract_remaining_months'].count(),
       user_churned[user_churned['contract_remaining_months'] == 12]['contract_remaining_months'].count()],
    name='Churn',
    marker_color='red', showlegend=False), row=3, col=1)
fig.add_trace(go.Bar(
    x=[1,2,3,4,5,6,7,8,9,10,11,12],
    y=[user_not_churned[user_not_churned['contract_remaining_months'] == 1]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 2]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 3]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 4]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 5]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 6]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 7]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 8]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 9]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 10]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 11]['contract_remaining_months'].count(),
       user_not_churned[user_not_churned['contract_remaining_months'] == 12]['contract_remaining_months'].count()],
    name='No churn',
    marker_color='blue', showlegend=False), row=3, col=1)

# lifetime_months
fig.add_trace(go.Histogram(
    x = user_churned['lifetime_months'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=4, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['lifetime_months'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='blue',
    showlegend=False),row=4, col=1)

# other_services_total_dollars
fig.add_trace(go.Histogram(
    x = user_churned['other_services_total_dollars'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=5, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['other_services_total_dollars'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='blue',
    showlegend=False),row=5, col=1)

# avg_visits_per_week_total
fig.add_trace(go.Histogram(
    x = user_churned['avg_visits_per_week_total'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=6, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['avg_visits_per_week_total'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='blue',
    showlegend=False),row=6, col=1)

# avg_visits_per_week_last_month
fig.add_trace(go.Histogram(
    x = user_churned['avg_visits_per_week_last_month'],
    name = 'Churn',
    histnorm = 'density',
    marker_color='red',
    showlegend=False),row=7, col=1)
fig.add_trace(go.Histogram(
    x = user_not_churned['avg_visits_per_week_last_month'],
    name= 'No churn',
    histnorm = 'density',
    marker_color='blue',
    showlegend=False),row=7, col=1)

# plot all of them
fig.update_layout(height=1700, width=1000,
              title_text="Destribution of features")
fig.show()


# ![omg1.PNG](attachment:omg1.PNG)
# ![omg2.PNG](attachment:omg2.PNG)
# ![omg3.PNG](attachment:omg3.PNG)
# ![omg4.PNG](attachment:omg4.PNG)
# ![omg5.PNG](attachment:omg5.PNG)
# ![omg6.PNG](attachment:omg6.PNG)
# ![omg7.PNG](attachment:omg7.PNG)

# ### Build a correlation matrix

cm = data.corr()
plt.figure(figsize=(15, 9))
sns.heatmap(cm, annot=True, square=True)
plt.show()

# ### Build a binary classification model for customers where the target feature is the user's leaving next month.

# features (X matrix)
X = data.drop(['churn'], axis=1)

# target variable (y)
y = data['churn']

# divide data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)

# Create scaler object and apply it to train set
scaler = StandardScaler()

# Train scaler and transform the matric for train set
X_train_st = scaler.fit_transform(X_train)

# apply standardization of feature matric for test set
X_test_st = scaler.transform(X_test)

# define the models to compare
models = [LogisticRegression(random_state=0), RandomForestClassifier(random_state=0)]

# function that predicts model by taking data as input and outputting metrics
def make_prediction(model, X_train, y_train, X_test, y_test):
    model = model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Model: ', model)
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred)))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred)))
    print('\n')

# output metric for both models
for i in models:
    make_prediction(i, X_train, y_train, X_test, y_test)

# ## Create user clusters

# ### Identify object (user) clusters

# Standardize the data
sc = StandardScaler()
x_sc = sc.fit_transform(X)

# Build a matrix of distances based on the standardized feature matrix
linked = linkage(x_sc, method='ward')

# Plot a dendrogram
plt.figure(figsize=(15,10))
dendrogram(linked, orientation='top')
plt.show()

# Train the clustering model with the K-means algorithm and predict customer clusters. 
km = KMeans(n_clusters = 5, random_state=0)
labels = km.fit_predict(x_sc)

# calculate silhouette score
sil_score = silhouette_score(x_sc, labels)

# Look at the mean feature values for clusters
data['cluster'] = labels

cluster_data = data.groupby('cluster').mean()
display(cluster_data)

# ### Plo distributions of features for the clusters.

# columns for overlap
numerical_col = ['age','other_services_total_dollars','lifetime_months','avg_visits_per_week_total','avg_visits_per_week_last_month']

# create subplots
fig = make_subplots(rows=3, cols=2,subplot_titles=numerical_col)

r = 1
c = 1
idx = 1
legend = True

# for every plot
for i in numerical_col:
    # add cluster data
    fig.add_trace(go.Histogram(x=data.query('cluster == 0')[i], name='cluster0', legendgroup='cluster0',
                               marker = {'color':'Red'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 1')[i],name='cluster1', legendgroup='cluster1',
                               marker = {'color':'Orange'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 2')[i],name='cluster2', legendgroup='cluster2',
                               marker = {'color':'Yellow'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 3')[i],name='cluster3', legendgroup='cluster3',
                               marker = {'color':'Green'},showlegend=legend),row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 4')[i],name='cluster4', legendgroup='cluster4',
                               marker = {'color':'Blue'},showlegend=legend),row=r, col=c)
    # rotate to next row col
    legend = False
    r = (math.floor(idx/2) + 1)
    c = (idx%2 + 1)
    idx = idx+1
    
fig.update_layout(barmode='overlay', height=1000)
fig.update_traces(opacity=0.65)
fig.show()

# columns for group
group_columns = ['gender','near_location','employer_partner','friend_promo','phone_exists','total_contract_period_months',
             'group_sessions','contract_remaining_months']

# create subplots
fig = make_subplots(rows=4, cols=2,subplot_titles=group_columns)

idx = 0
r = (math.floor(idx/2) + 1)
c = (idx%2 + 1)
legend = True

# for every plot
for i in group_columns:
    # add cluster data
    fig.add_trace(go.Histogram(x=data.query('cluster == 0')[i],name='cluster0', legendgroup='cluster0',
                               marker = {'color':'Red'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 1')[i],name='cluster1', legendgroup='cluster1',
                               marker = {'color':'Orange'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 2')[i], name='cluster2', legendgroup='cluster2',
                               marker = {'color':'Yellow'},showlegend=legend), row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 3')[i],name='cluster3', legendgroup='cluster3',
                               marker = {'color':'Green'},showlegend=legend),row=r, col=c)
    fig.add_trace(go.Histogram(x=data.query('cluster == 4')[i],name='cluster4', legendgroup='cluster4',
                               marker = {'color':'Blue'},showlegend=legend),row=r, col=c)
    # rotate to next row col
    idx = idx+1
    r = (math.floor(idx/2) + 1)
    c = (idx%2 + 1)
    legend = False
    
fig.update_xaxes(type="category", row=3, col=2)
fig.update_layout(barmode='group', height=1200)
fig.show()

# ![img1.PNG](attachment:img1.PNG)
# ![img2.PNG](attachment:img2.PNG)
# ![img3.PNG](attachment:img3.PNG)
# ![img4.PNG](attachment:img4.PNG)
# ![img5.PNG](attachment:img5.PNG)
# ![img6.PNG](attachment:img6.PNG)
# ![img7.PNG](attachment:img7.PNG)

# ### Calculate the churn rate for each cluster
# create pivot table where it is separated by cluster and calculated based on churn
churn_pivot = data.pivot_table(index='cluster', values='churn', 
                               aggfunc=['count', 'sum', lambda x: abs(round(((x == 0).sum() / x.count()-1)*100,2))]).reset_index()
churn_pivot.columns = ['cluster', 'total # of entries', 'churn sum', 'churn rate (%)']
display(churn_pivot)
