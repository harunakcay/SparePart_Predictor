import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import random
import math
from sklearn import preprocessing
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def initiateDF(fname):
  df = pd.read_excel(fname)
  dict_df = pd.read_excel(fname, sheet_name=1)
  my_dictionary = dict_df.set_index('MATL_GROUP')['Açıklama'].to_dict()

  return df, my_dictionary

def extractTrivialColumns(df):
  for col in df.columns:
    if df[col].nunique() == 1:
      df.drop(col, axis=1, inplace=True)
      #print("Column dropped:", col, " (constant value)")

    elif df[col].nunique() == len(df):
      df.drop(col, axis=1, inplace=True)
      #print("Column dropped:", col, " (unique values for each row)")

    elif df[col].isna().sum() > len(df)*0.5:
      df.drop(col, axis=1, inplace=True)
      #print("Column dropped:", col, " (too much NaN values)")

def fill_empty_rows(group_df,col,weights,values):
  value_counts = group_df[col].value_counts()

  if value_counts.iloc[0] >= 1.5 * value_counts.iloc[1]:
      dominant_value = value_counts.index[0]
      group_df[col].fillna(dominant_value, inplace=True)
  else:
      random_fill = random.choices(values, weights=weights, k=group_df[col].isnull().sum())
      group_df.loc[group_df[col].isnull(), col] = random_fill

  return group_df

def filterDF(target):
  myd = df[df["MATL_GROUP"] == target]
  mydf = myd[["ZSIKAYET","ZPRDHYR8"]]

  pairs = mydf.apply(tuple, axis=1)

  pair_counts = Counter(pairs)
  min_support = 2
  frequent_common_pairs = {pair: count for pair, count in pair_counts.items() if count >= min_support}
  filtered_pairs = list(frequent_common_pairs.keys())

  filtered_df =  myd[mydf.apply(tuple, axis=1).isin(filtered_pairs)]

  return filtered_df

def processDF(df):

  extractTrivialColumns(df)

  df = df.groupby('MATL_GROUP').apply(fill_empty_rows,col='PRICE_GRP',weights = [0.5,0.5],values = [14, 1])
  df["PRD_YEAR"] = df["ZURTMONTH"]//100

  df["USAGE"] = (df["CRMPOSTDAT"]//10000 - df["CRM_WYBEGD"]//10000)*12 + (df["CRMPOSTDAT"]//100)%100 - (df["CRM_WYBEGD"]//100)%100
  df["USAGE"] = df["USAGE"]//12

  column_order = list(df.columns)
  column_order.remove('MATL_GROUP')
  column_order.append('MATL_GROUP')
  df = df[column_order]
  df = df[['PRICE_GRP','ZCRMPRD','ZPRDHYR8','ZRPRGRP','ZSIKAYET','ZURNTIP','ZZMARKA','PRD_YEAR','USAGE','MATL_GROUP']]

  encode_dict = {}

  obj = df.select_dtypes(include="object")
  for column in obj.columns:
    le = preprocessing.LabelEncoder()
    df[column] = le.fit_transform(df[column])
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    encode_dict[column] = class_mapping

  return df,encode_dict

def deleteRows(df):
  result_df = pd.DataFrame()

  for target in [320001295,303113320,320001053]:
      sub_df = filterDF(target)
      result_df = pd.concat([result_df,sub_df],axis=0)

  for target in [303113250, 320001057, 303113360, 320001071, 303113170, 303113190]:
      myd = df[df["MATL_GROUP"] == target]
      result_df = pd.concat([result_df,myd],axis=0)

  return result_df

def getModel(df):

  le = preprocessing.LabelEncoder()
  X = df.drop(columns=["MATL_GROUP"])
  y = df["MATL_GROUP"]
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
  # y_train = le.fit_transform(y_train)
  # y_test = le.fit_transform(y_test)
  # xgb = XGBClassifier(base_score= 0.2, booster= 'gbtree', gamma= 1, learning_rate= 0.05, n_estimators= 500, reg_alpha= 0, reg_lambda= 0.5)
  # xgb.fit(X_train, y_train)
  # y_pred = xgb.predict(X_test)
  # y_pred = le.inverse_transform(y_pred)
  # y_test = le.inverse_transform(y_test)
  # y_train = le.inverse_transform(y_train)

  rf = RandomForestClassifier()
  rf.fit(X_train,y_train)

  return rf
  
def getReport(encode_dict):
  PRICE_GRP = input("PRICE_GRP: ")

  ZCRMPRD = input("ZCRMPRD: ")
  if ZCRMPRD in encode_dict['ZCRMPRD'].keys():
    ZCRMPRD = encode_dict['ZCRMPRD'][ZCRMPRD]
  else:
    ZCRMPRD = len(encode_dict['ZCRMPRD'].values())

  ZPRDHYR8 = input("ZPRDHYR8: ")
  if ZPRDHYR8 in encode_dict['ZPRDHYR8'].keys():
    ZPRDHYR8 = encode_dict['ZPRDHYR8'][ZPRDHYR8]
  else:
    ZPRDHYR8 = len(encode_dict['ZPRDHYR8'].values())

  ZRPRGRP = input("ZRPRGRP: ")

  ZSIKAYET = input("ZSIKAYET: ")
  if ZSIKAYET in encode_dict['ZSIKAYET'].keys():
    ZSIKAYET = encode_dict['ZSIKAYET'][ZSIKAYET]
  else:
    ZSIKAYET = len(encode_dict['ZSIKAYET'].values())

  ZURNTIP = input("ZURNTIP: ")

  ZZMARKA = input("ZZMARKA: ")
  if ZZMARKA in encode_dict['ZZMARKA'].keys():
    ZZMARKA = encode_dict['ZZMARKA'][ZZMARKA]
  else:
    ZZMARKA = len(encode_dict['ZZMARKA'].values())


  PRD_YEAR = input("PRD_YEAR: ")

  USAGE = input("USAGE: ")




  data = {'PRICE_GRP': [PRICE_GRP], 'ZCRMPRD': [ZCRMPRD], 'ZPRDHYR8': [ZPRDHYR8],
          'ZRPRGRP': [ZRPRGRP], 'ZSIKAYET': [ZSIKAYET], 'ZURNTIP': [ZURNTIP],
          'ZZMARKA': [ZZMARKA], 'PRD_YEAR': [PRD_YEAR], 'USAGE': [USAGE]}
  
  input_df = pd.DataFrame(data)
  return input_df


#classes = [303113170,303113190,303113250,303113320,303113360,320001053,320001057,320001071,320001295]

df, my_dict = initiateDF('UC01_TV_raw_mini_dataset.xlsx')
df,encode_dict = processDF(df)
df = deleteRows(df)

model = getModel(df)

input_df = getReport(encode_dict)

pred = model.predict_proba(input_df)
proba_df = pd.DataFrame(pred, columns=my_dict.keys())
top2 = proba_df.iloc[0].nlargest(2)


print('\n******PREDICTIONS:******')
for column_name, value in top2.items():
  value = "{:.2f}".format(value)
  print(f"{my_dict[column_name]}: {value}")
print()
