#!/usr/bin/env python
# coding: utf-8

import pandas as pd
data = pd.read_csv('/datasets/credit_scoring_eng.csv')
creditScore = pd.DataFrame(data=data)

# Processing missing values

# handle negative or missing children
creditScore['children'] = creditScore['children'].replace(-1,0)
creditScore['children'] = creditScore['children'].replace(20,0)
creditScore['children'].fillna(0,inplace=True)

# handle missing days employee
creditScore['days_employed'].fillna(0,inplace=True)

# handle missing total income
creditScore['total_income'].fillna(0,inplace=True)
print(creditScore.duplicated().sum())

# change float type to int for days_employed
creditScore['days_employed'] = creditScore['days_employed'].astype(int)

for days in creditScore['days_employed']:
    if days < 0:
        creditScore['days_employed'] = creditScore['days_employed'].replace(days, 0)
        
creditScore['years_employed'] = (creditScore['days_employed']/365).astype(int).round()

# change float type to int for total_income
creditScore['total_income'] = creditScore['total_income'].astype(int)

#handle duplicates in education
creditScore['education'] = creditScore['education'].str.lower()

#handle duplicates in purpose
import nltk
from nltk.stem import WordNetLemmatizer

wordnet_lemmma = WordNetLemmatizer()

for purpose in creditScore['purpose']:
    words = nltk.word_tokenize(purpose)

    if 'education' in words or 'university' in words or 'educated' in words:
        creditScore['purpose'].replace(purpose, 'education',inplace=True)
    
    if 'car' in words or 'cars' in words:
        creditScore['purpose'].replace(purpose,'car',inplace=True)
        
    if 'house' in words or 'housing' in words or 'estate' in words or 'property' in words:
        creditScore['purpose'].replace(purpose,'real estate',inplace=True)

    if 'wedding' in words:
        creditScore['purpose'].replace(purpose,'wedding',inplace=True)

#print(creditScore['purpose'].unique())

# ### Processing duplicates

#print(creditScore.duplicated().sum())
creditScore = creditScore.drop_duplicates().reset_index(drop=True)


#Step 3. Answer these questions

# - Is there a relation between having kids and repaying a loan on time?

kidsData = creditScore.groupby('children')
print(kidsData['debt'].describe())
print(kidsData['debt'].count()/kidsData['debt'].sum())
creditScore.pivot_table(index = 'children', values = 'debt', aggfunc = ['count', 'mean'])

# - Is there a relation between marital status and repaying a loan on time?

maritalData = creditScore.groupby('family_status')
print(maritalData['debt'].describe())
print(maritalData['debt'].count()/maritalData['debt'].sum())
creditScore.pivot_table(index = 'family_status', values = 'debt', aggfunc = ['count', 'mean'])


# - Is there a relation between income level and repaying a loan on time?

incomeData = creditScore.groupby('income_type')
print(incomeData['debt'].describe())
print(incomeData['debt'].count()/incomeData['debt'].sum())
creditScore.pivot_table(index = 'income_type', values = 'debt', aggfunc = ['count', 'mean'])

# - How do different loan purposes affect on-time repayment of the loan?

purposeData = creditScore.groupby('purpose')
print(purposeData['debt'].describe())
print(purposeData['debt'].count() / purposeData['debt'].sum())
