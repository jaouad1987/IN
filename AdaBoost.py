# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 23:26:24 2018

@author: Jaouad
"""

# La clasificacion AdaBoost 
import pandas
import numpy as np
import category_encoders as ce
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

# establecemos todos los nombres de columna para el conjunto de datos de entrenamiento
names1 = ['id','amount_tsh','date_recorded','funder','gps_height','installer','longitude','latitude','wpt_name',
          'num_private','basin','subvillage','region','region_code','district_code','lga','ward','population',
          'public_meeting','recorded_by','scheme_management','scheme_name','permit','construction_year',
          'extraction_type','extraction_type_group','extraction_type_class','management','management_group',
          'payment','payment_type','water_quality','quality_group','quantity','quantity_group','source','source_type',
          'source_class','waterpoint_type','waterpoint_type_group']

#   establecemos todos los nombres de columna para conjunto de etiquetas
names2 = ['id', 'status_group']

# Leemos el conjudto de datos de entrenamiento
df1 = pandas.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv', names=names1)

# Leemos el conjudto de etiquetas
df2 = pandas.read_csv('Labels.csv', names=names2)

# concatenados los dos datos que hemos leido 

df = pandas.concat([df1, df2], axis=1)

  
df.drop(df.index[0], inplace=True)

# Eliminamos las celdas blancas
df['funder'].replace('', np.nan, inplace=True)
df.dropna(subset=['funder'], inplace=True)

df['public_meeting'].replace('', np.nan, inplace=True)
df.dropna(subset=['public_meeting'], inplace=True)

df['scheme_management'].replace('', np.nan, inplace=True)
df.dropna(subset=['scheme_management'], inplace=True)

df['permit'].replace('', np.nan, inplace=True)
df.dropna(subset=['permit'], inplace=True)

# Replazamos la caracteristica 'date_recorded' con la nueva caracteristica (date_recorded - construction_year)
df['date_recorded'].replace('', np.nan, inplace=True)
df.dropna(subset=['date_recorded'], inplace=True)
df['date_recorded'] = pandas.to_datetime(df['date_recorded']).dt.year
df['date_recorded'] = df['date_recorded'].astype('int32')
df['construction_year'] = df['construction_year'].astype('int32')
df['construction_year'] = df['construction_year'].replace(0,np.nan)
df = df.dropna(subset=['construction_year'])
df['date_recorded'] = df['date_recorded']- df['construction_year']

 
df.drop(df.columns[[0,8,9,11,12,13,14,15,16,19,21,23,25,26,28,30,34,36,37,39]], axis=1, inplace=True)

#Trasnformamos la categoria variables a numerica

encoder = ce.OrdinalEncoder(cols=['status_group'])
df = encoder.fit_transform(df)
df = df.apply(pandas.to_numeric, errors='ignore')
encoder = ce.BinaryEncoder()
df = encoder.fit_transform(df)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Guardamos el contenido en un array
array = df.values
X = array[:,0:67]
Y = array[:,68]

# Ejecutamos nuestro algoritmo
seed = 7
k = 10
num_trees = 150
kFold = model_selection.KFold(n_splits=k, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kFold)
print(results.mean())