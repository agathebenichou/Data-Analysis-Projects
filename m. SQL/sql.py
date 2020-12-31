#!/usr/bin/env python
# coding: utf-8

# # SQL Project

# The coronavirus took the entire world by surprise, changing everyone's daily routine. City dwellers no longer spent their free time outside, going to cafes and malls; more people were home, reading books. That attracted the attention of startups that rushed to develop new apps for book lovers.
# 
# You've been given a database of one of the services competing in this market. It contains data on books, publishers, authors, and customer ratings and reviews of books. This information will be used to generate a value proposition for a new product.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates
import seaborn as sns
import math
import matplotlib.dates as mdates
import datetime 
import math
from sqlalchemy import create_engine


# ## Download the data

# ### Connect to the database

# In[2]:


db_config = {'user': 'practicum_student',         # username
             'pwd': 's65BlTKV3faNIGhmvJVzOqhs', # password
             'host': 'rc1b-wcoijxj3yxfsf3fs.mdb.yandexcloud.net',
             'port': 6432,              # connection port
             'db': 'data-analyst-final-project-db'}          # the name of the database

connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
                                                                     db_config['pwd'],
                                                                       db_config['host'],
                                                                       db_config['port'],
                                                                       db_config['db'])

engine = create_engine(connection_string, connect_args={'sslmode':'require'})


# ### Query engine for data

# In[3]:


books = pd.io.sql.read_sql('''SELECT * FROM books''', con = engine)

# books table - convert to datetime
books['publication_date'] = pd.to_datetime(books['publication_date'], format="%Y-%m-%d")

display(books)


# The books table contains information in books such as book id, author id, book title, number of pages in a book, publication date and publisher id. 

# In[4]:


authors = pd.io.sql.read_sql('''SELECT * FROM authors''', con = engine)
display(authors)


# The authors table contains information on authors such as author id and author name

# In[5]:


publishers = pd.io.sql.read_sql('''SELECT * FROM publishers''', con = engine)
display(publishers)


# The publishers table contains data on publishers such as publisher id and publisher name.

# In[6]:


ratings = pd.io.sql.read_sql('''SELECT * FROM ratings''', con = engine)
display(ratings)


# The ratings table contains information on book ratings such as the rating id, the associated book id, the username of the person who posted the rating and the rating itself.

# In[7]:


reviews = pd.io.sql.read_sql('''SELECT * FROM reviews''', con = engine)
display(reviews)


# The reviews table contains information on book reviews such as review id, the associated book id, the username of the person who posted the review and the review itself.

# ## Find the number of books released after January 1, 2000.

# In[8]:


query = ''' SELECT COUNT(*) FROM books WHERE publication_date > '2000-01-01' '''

result = pd.io.sql.read_sql(query, con = engine)
print('The number of books released after January 1st, 2000 is: ' + str(result['count'][0]))


# ## Find the number of user reviews and the average rating for each book.

# In[9]:


query = ''' SELECT 
                books.title AS title,
                AVG(ratings.rating) as avg_rating,
                COUNT( DISTINCT reviews.review_id) as num_reviews
            FROM books
            INNER JOIN ratings on ratings.book_id = books.book_id
            INNER JOIN reviews on reviews.book_id = books.book_id
            GROUP BY books.book_id
            ORDER BY num_reviews DESC'''

result = pd.io.sql.read_sql(query, con = engine)
display(result.head(10))


# The table above displays the query results for the number of user reviews and the average ratings for the top 10 books. 

# ## Identify the publisher that has released the greatest number of books with more than 50 pages.

# In[10]:


# by reviewer 20:32 31.12.2020
query = ''' SELECT publishers.publisher as publisher, COUNT( DISTINCT books.book_id) as num_books
            FROM publishers
            INNER JOIN books on books.publisher_id = publishers.publisher_id
            WHERE books.num_pages > 50
            GROUP BY publisher
            ORDER BY num_books DESC'''

result = pd.io.sql.read_sql(query, con = engine)
display(result.head(10))


# The table above displays the top 10 publishers who released the greatest number of books with more than 50 pages.

# ## Identify the author with the highest average book rating: only books with at least 50 ratings.

# In[12]:


query = ''' SELECT 
                authors.author as author_name,
                AVG(ratings.rating) as avg_rating
            FROM authors
            INNER JOIN books on books.author_id = authors.author_id
            INNER JOIN ratings on ratings.book_id = books.book_id
            GROUP BY author_name
            HAVING COUNT(ratings.rating) > 50
            ORDER BY avg_rating DESC
                '''

result = pd.io.sql.read_sql(query, con = engine)
display(result.head(10))


# The table above displays the authors with the highest average book rating for books only with greater than 50 pages. 

# ## Find the average number of text reviews among users who rated more than 50 books.

# In[13]:


query = ''' SELECT reviews.username, count(reviews.review_id) as avg_num_reviews
            FROM reviews
            WHERE reviews.username in (SELECT ratings.username AS user            
                    FROM ratings
                    GROUP BY ratings.username
                    HAVING COUNT(ratings.rating_id) > 50) 
            GROUP BY reviews.username
            '''

result = pd.io.sql.read_sql(query, con = engine)
display(result.head(10))


# The table above shows the average number of text reviews among users who rated more than 50 books. There were only 6 users who rated more than 50 books and using these users, I calculated the average number of text reviews for them. 
