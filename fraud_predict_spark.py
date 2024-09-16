#!/usr/bin/env python
# coding: utf-8

# # This Example shows the Prediction of Bike Flow in the NYC City using the deep learning model ST-ResNet.

# Find the details of the ST-ResNet model in the <a href="https://dl.acm.org/doi/10.5555/3298239.3298479">corresponding paper</a>

# Details of the dataset can be found <a href="https://github.com/FIBLAB/DeepSTN">here</a>.

# ### Import Modules and Define Parameters

# In[1]:


import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, when, udf, max, count, trim
from pyspark.sql.types import IntegerType, ArrayType, FloatType, StringType, LongType
import datetime
import time
import xgboost as xgb


# In[2]:


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=0)


# In[3]:


# Load DNN model
def load_dnn_model(num_feature, model_path):
    model = SimpleNN(num_feature)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model


# In[4]:


# Load XGBoost model
def load_xgboost_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model


# In[5]:


spark = SparkSession.builder.appName("FraudDetectionSpark").config("spark.driver.memory", "16g").config("spark.dynamicAllocation.enabled", "true").getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "209715200")


# In[6]:


countryDict = {}
with open('resources/model/country_mapping.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(",")
        countryDict[key] = int(value)
#print(countryDict)


# In[7]:


def mapCountry(countrStr):
    return countryDict.get(countrStr, -1)  # Returns original value if not found

# Register the UDF (specify the return type)
map_country_udf = udf(mapCountry, IntegerType())


# In[8]:


def remove_quotes(numberStr):
    if len(numberStr) >= 2 and numberStr[0] == '"' and numberStr[-1] == '"':
        return numberStr[1:-1]
    return numberStr

remove_quotes_udf = udf(remove_quotes, StringType())


# In[9]:


dfOrder = spark.read.csv("resources/data/500_mb/order.csv", header=True, inferSchema=True)
dfOrder = dfOrder.drop('store')

str_cols = [field.name for field in dfOrder.schema.fields if isinstance(field.dataType, StringType)]
for c in str_cols:
    dfOrder = dfOrder.withColumn(c, remove_quotes_udf(col(c)))

dfOrder = dfOrder.withColumn("o_order_id", col("o_order_id").cast(IntegerType())).withColumn("o_customer_sk", col("o_customer_sk").cast(IntegerType())).withColumn("o_date", col("date").cast(StringType()))
dfOrder = dfOrder.drop('date')
dfOrder = dfOrder.repartition("o_customer_sk")
dfOrder.collect()
print("Order table size: ", dfOrder.count())
dfOrder.cache()
#dfOrder.show()


# In[10]:


dfTransaction = spark.read.csv("resources/data/500_mb/financial_transactions.csv", header=True, inferSchema=True)
dfTransaction = dfTransaction.drop('receiverID')
#dfTransaction = dfTransaction.dropna()

str_cols = [field.name for field in dfTransaction.schema.fields if isinstance(field.dataType, StringType)]
for c in str_cols:
    dfTransaction = dfTransaction.withColumn(c, remove_quotes_udf(col(c)))
    
dfTransaction = dfTransaction.withColumn("amount", col("amount").cast(FloatType())).withColumn("senderID", col("senderID").cast(IntegerType())).withColumn("transactionID", col("transactionID").cast(IntegerType())).withColumn("t_time", col("time").cast(StringType()))
dfTransaction = dfTransaction.drop('time')
dfTransaction = dfTransaction.repartition("senderID")
dfTransaction.collect()
print("Transaction table size: ", dfTransaction.count())
dfTransaction.cache()
#dfTransaction.show()


# In[11]:


dfCustomer = spark.read.csv("resources/data/500_mb/customer.csv", header=True, inferSchema=True)
dfCustomer = dfCustomer.drop('c_customer_id').drop('c_first_name').drop('c_last_name').drop('c_birth_day').drop('c_birth_month').drop('c_login').drop('c_email_address')
#dfCustomer = dfCustomer.dropna()

'''str_cols = [field.name for field in dfCustomer.schema.fields if isinstance(field.dataType, StringType)]
for c in str_cols:
    dfCustomer = dfCustomer.withColumn(c, remove_quotes_udf(col(c)))'''

column_names = dfCustomer.columns
trimmed_columns = [trim(col(c)).alias(c) for c in column_names]
dfCustomer = dfCustomer.select(*trimmed_columns)

dfCustomer = dfCustomer.withColumn("c_customer_sk", col("c_customer_sk").cast(IntegerType())).withColumn("c_current_addr_sk", col("c_current_addr_sk").cast(IntegerType())).withColumn("c_birth_year", col("c_birth_year").cast(IntegerType())) 
dfCustomer = dfCustomer.withColumn("c_preferred_cust_flag", when(col("c_preferred_cust_flag") == 'N', 0).otherwise(1))
dfCustomer = dfCustomer.withColumn("c_birth_country", map_country_udf(col("c_birth_country")))
dfCustomer = dfCustomer.fillna(0)
dfCustomer = dfCustomer.repartition("c_customer_sk")
dfCustomer.collect()
print("Customer table size: ", dfCustomer.count())
dfCustomer.cache()
#dfCustomer.show()


# In[55]:


def date_to_timestamp(date_str):
    # Parse the date string to a datetime object
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    # Return the timestamp in milliseconds
    return int(dt.timestamp())

# Register the UDF (specifying the return type as LongType)
#date_to_timestamp_udf = udf(date_to_timestamp, LongType())
spark.udf.register("date_to_timestamp_udf", date_to_timestamp, LongType())


# In[56]:


def datetime_to_timestamp(date_str):
    # Parse the date string to a datetime object
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    # Return the timestamp in milliseconds
    return int(dt.timestamp()) - 25200

# Register the UDF (specifying the return type as LongType)
#date_to_timestamp_udf = udf(date_to_timestamp, LongType())
spark.udf.register("datetime_to_timestamp_udf", datetime_to_timestamp, LongType())


# In[14]:


def is_weekday(timestamp):
    # Convert timestamp to datetime object
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    weekday = (dt.weekday() + 1) % 7  # Monday is 1 and Sunday is 0
    if weekday == 0 or weekday == 6:
        return 0
    else:
        return 1

# Register the UDF with Spark
#weekday_udf = udf(get_weekday, IntegerType())
spark.udf.register("is_weekday_udf", is_weekday, IntegerType())


# In[15]:


def time_diff_in_days(timestamp1, timestamp2):
    secondsInADay = 86400
    return abs(timestamp1 - timestamp2)//secondsInADay

# Register the UDF (specifying the return type as LongType)
#time_diff_in_days_udf = udf(time_diff_in_days, LongType())
spark.udf.register("time_diff_in_days_udf", time_diff_in_days, LongType())


# In[16]:


def calculate_age(birth_year):
    '''if birth_year is None:
        birth_year = 0'''
    current_year = datetime.datetime.now().year
    return current_year - birth_year

# Register the UDF (specify return type as IntegerType)
#calculate_age_udf = udf(calculate_age, IntegerType())
spark.udf.register("calculate_age_udf", calculate_age, IntegerType())


# In[17]:


def get_weekday(timestamp):
    # Convert timestamp to datetime object
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    return (dt.weekday() + 1) % 7  # Monday is 0 and Sunday is 6

# Register the UDF with Spark
#weekday_udf = udf(get_weekday, IntegerType())
spark.udf.register("get_weekday_udf", get_weekday, IntegerType())


# In[18]:


def get_customer_features(address_num, cust_flag, birth_country, age):
    '''if address_num is None:
        address_num = 0
    if cust_flag is None:
        cust_flag = 0
    if birth_country is None:
        birth_country = 0
    if age is None:
        age = 0'''
        
    fAddressNum = float(address_num) / 35352.0
    fCustFlag = float(cust_flag)
    fBirthCountry = float(birth_country) / 211.0
    fAge = float(age) / 94.0
    return [fAddressNum, fCustFlag, fBirthCountry, fAge]

# Register the UDF with Spark
#get_customer_features_udf = udf(get_customer_features, ArrayType(FloatType()))
spark.udf.register("get_customer_features_udf", get_customer_features, ArrayType(FloatType()))


# In[19]:


def get_transaction_features(total_order, amount, time_diff, timestamp):
    '''if total_order is None:
        total_order = 0
    if amount is None:
        amount = 0.0
    if time_diff is None:
        time_diff = 0
    if timestamp is None:
        timestamp = 0'''
        
    fTotalOrder = float(total_order) / 79.0
    fAmount = amount / 16048.0
    fTimeDiff = float(time_diff) / 729.0
    
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    fDayOfWeek = float((dt.weekday() + 1) % 7) / 6.0
    
    fDaysSinceEpoch = float(timestamp // 86400) / 15338.0
    return [fTotalOrder, fAmount, fTimeDiff, fDayOfWeek, fDaysSinceEpoch]

# Register the UDF with Spark
#get_transaction_features_udf = udf(get_transaction_features, ArrayType(FloatType()))
spark.udf.register("get_transaction_features_udf", get_transaction_features, ArrayType(FloatType()))


# In[20]:


def concatenate_features(array1, array2):
    return array1 + array2

# Register the UDF
#concatenate_features_udf = udf(concatenate_features, ArrayType(FloatType()))
spark.udf.register("concatenate_features_udf", concatenate_features, ArrayType(FloatType()))


# In[21]:


# Load and register DNN the model
model = load_dnn_model(9, "resources/model/fraud_dnn_weights.pth")
model.eval()

# Define UDF function for prediction
def predict_dnn(features):
    # Convert input features to tensor
    input_tensor = torch.tensor(np.array(features), dtype=torch.float32)
    # Perform prediction
    result = model(input_tensor)
    result = np.argmax(result.detach().numpy(), axis=0)
    return int(result)

# Register the UDF
#concatenate_features_udf = udf(concatenate_features, ArrayType(FloatType()))
spark.udf.register("predict_dnn_udf", predict_dnn, IntegerType())


# In[22]:


# Load and register small xgboost model
xgb_small = load_xgboost_model("resources/model/fraud_xgboost_5_16.json")

def predict_xgb_small(features):
    # Convert input features to DMatrix
    dmatrix = xgb.DMatrix(np.array(features).reshape(1, -1))
    # Perform prediction
    preds = xgb_small.predict(dmatrix)
    return float(preds[0])

# Register the UDF
predict_xgm_small_udf = udf(predict_xgb_small, FloatType())
spark.udf.register("predict_xgb_small_udf", predict_xgb_small, FloatType())


# In[23]:


# Load and register large xgboost model
xgb_large = load_xgboost_model("resources/model/fraud_xgboost_9_1600.json")

def predict_xgb_large(features):
    # Convert input features to DMatrix
    dmatrix = xgb.DMatrix(np.array(features).reshape(1, -1))
    # Perform prediction
    preds = xgb_large.predict(dmatrix)
    return float(preds[0])

# Register the UDF
predict_xgb_large_udf = udf(predict_xgb_large, FloatType())
spark.udf.register("predict_xgb_large_udf", predict_xgb_large, FloatType())


# In[25]:


dfOrder.createOrReplaceTempView("df_order")
dfTransaction.createOrReplaceTempView("df_transaction")
dfCustomer.createOrReplaceTempView("df_customer")


# In[57]:


dfTemp = spark.sql("SELECT o_customer_sk, o_order_id, date_to_timestamp_udf(o_date) as o_timestamp FROM df_order")
dfTemp.createOrReplaceTempView("df_temp")
dfTemp = spark.sql("SELECT o_customer_sk, COUNT(o_order_id) AS total_order, MAX(o_timestamp) AS o_last_order_time FROM df_temp WHERE o_timestamp IS NOT NULL AND is_weekday_udf(o_timestamp) = 1 GROUP BY o_customer_sk")
dfTemp.createOrReplaceTempView("df_temp")
#dfTemp.show()


# In[59]:


dfTemp2 = spark.sql("SELECT transactionID, senderID, amount, datetime_to_timestamp_udf(t_time) AS t_timestamp FROM df_transaction")
dfTemp2.createOrReplaceTempView("df_temp2")
#dfTemp2.show()


# In[61]:


dfJoinTemp = spark.sql("SELECT o_customer_sk, total_order, transactionID, amount, t_timestamp, time_diff_in_days_udf(o_last_order_time, t_timestamp) AS time_diff FROM df_temp INNER JOIN df_temp2 ON o_customer_sk = senderID WHERE t_timestamp IS NOT NULL")
dfJoinTemp.createOrReplaceTempView("df_join_temp")
dfJoinTemp = spark.sql("SELECT o_customer_sk, transactionID, get_transaction_features_udf(total_order, amount, time_diff, t_timestamp) AS transaction_features FROM df_join_temp WHERE time_diff <= 500")
dfJoinTemp.createOrReplaceTempView("df_join_temp")
#dfJoinTemp.show()


# In[63]:


dfTemp3 = spark.sql("SELECT c_customer_sk, get_customer_features_udf(c_current_addr_sk, c_preferred_cust_flag, c_birth_country, calculate_age_udf(c_birth_year)) AS customer_features FROM df_customer")
dfTemp3.createOrReplaceTempView("df_temp3")
#dfTemp3.show()


# In[64]:


dfJoin = spark.sql("SELECT transactionID, concatenate_features_udf(customer_features, transaction_features) AS all_features FROM df_join_temp INNER JOIN df_temp3 ON o_customer_sk = c_customer_sk WHERE predict_xgb_small_udf(transaction_features) >= 0.5")
dfJoin.createOrReplaceTempView("df_join")
#dfJoin.count()
#dfJoin.show()


# In[68]:


tStart = time.time()
dfOutput = spark.sql("SELECT transactionID FROM df_join WHERE predict_xgb_large_udf(all_features) >= 0.5 AND predict_dnn_udf(all_features) = 0")
dfOutput.collect()
tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")
#dfOutput.show()


# In[69]:


#print(dfOutput.count())

