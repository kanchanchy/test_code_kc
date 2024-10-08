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
cursor.query("DROP FUNCTION IF EXISTS IS_WORKING_DAY_EVADB;").df()
# register function
cursor.query(
    """
    CREATE FUNCTION
    IF NOT EXISTS IS_WORKING_DAY_EVADB
    IMPL './function_fraud_detect_evadb.py';
    """
).df()


# In[5]:


# deregister function
cursor.query("DROP FUNCTION IF EXISTS IS_ADULT_DURING_TRANSACTION_EVADB;").df()
# register function
cursor.query(
    """
    CREATE FUNCTION
    IF NOT EXISTS IS_ADULT_DURING_TRANSACTION_EVADB
    IMPL './function_fraud_detect_evadb.py';
    """
).df()


# In[6]:


# deregister function
cursor.query("DROP FUNCTION IF EXISTS IS_FRAUD_XGB_TRANS_EVADB;").df()
# register function
cursor.query(
    """
    CREATE FUNCTION
    IF NOT EXISTS IS_FRAUD_XGB_TRANS_EVADB
    IMPL './function_fraud_detect_evadb.py';
    """
).df()


# In[7]:


# deregister function
cursor.query("DROP FUNCTION IF EXISTS DNN_FRAUD_DETECT_EVADB;").df()
# register function
cursor.query(
    """
    CREATE FUNCTION
    IF NOT EXISTS DNN_FRAUD_DETECT_EVADB
    IMPL './function_fraud_detect_evadb.py';
    """
).df()


# In[8]:


tStart = time.time()
connector_name = "postgres_data"
table_name_account = "account_data"
table_name_customer = "customer_data"
table_name_transaction = "transaction_data"


# In[9]:


cursor.query(
    """
    USE postgres_data {
    CREATE OR REPLACE VIEW customer_full_data AS
    SELECT c_customer_sk, c_current_addr_sk, c_preferred_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, transaction_limit FROM account_data INNER JOIN customer_data ON fa_customer_sk = c_customer_sk
    };
"""
).df()


# In[ ]:


join_query = f"SELECT transaction_id, DNN_FRAUD_DETECT_EVADB(c_current_addr_sk, c_preferred_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, transaction_limit, t_time, amount) FROM postgres_data.customer_full_data JOIN postgres_data.transaction_data ON c_customer_sk = sender_id WHERE IS_WORKING_DAY_EVADB(t_time).label = 1 AND IS_ADULT_DURING_TRANSACTION_EVADB(t_time, c_birth_year).label = 1 AND IS_FRAUD_XGB_TRANS_EVADB(t_time, amount).label = 1"
joinDf = cursor.query(join_query).df()
print(joinDf.head())
print(joinDf.count())


# In[ ]:


tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")


# In[ ]:




