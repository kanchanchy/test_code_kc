#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import connectorx as cx
import ast
import evadb


# In[2]:


connection = evadb.connect()
cursor = connection.cursor()


# In[3]:


sql_connect_db = """
CREATE DATABASE IF NOT EXISTS postgres_data WITH ENGINE = 'postgres', PARAMETERS = {
     "user": "postgres",
     "password": "postgres",
     "host": "localhost",
     "port": "5432",
     "database": "postgres"
};
"""

cursor.query(sql_connect_db).df()


# In[4]:


# deregister function
cursor.query("DROP FUNCTION IF EXISTS IS_POPULAR_STORE_EVADB;").df()
# register function
cursor.query(
    """
    CREATE FUNCTION
    IF NOT EXISTS IS_POPULAR_STORE_EVADB
    IMPL './function_trip_type_evadb.py';
    """
).df()


# In[5]:


# deregister function
cursor.query("DROP FUNCTION IF EXISTS DNN_TRIP_TYPE_EVADB;").df()
# register function
cursor.query(
    """
    CREATE FUNCTION
    IF NOT EXISTS DNN_TRIP_TYPE_EVADB
    IMPL './function_trip_type_evadb.py';
    """
).df()


# In[6]:


tStart = time.time()
table_name1 = "postgres_data.order_data"
table_name2 = "postgres_data.store_data"
join_query = f"SELECT o_order_id, DNN_TRIP_TYPE_EVADB(o_customer_sk, o_date, weekday, store_feature) FROM {table_name1} JOIN {table_name2} ON o_store = s_store WHERE weekday != 'Sunday' AND IS_POPULAR_STORE_EVADB(store_feature).label = 1"
joinDf = cursor.query(join_query).df()
print(joinDf.head())
print(joinDf.count())


# In[7]:


tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")

