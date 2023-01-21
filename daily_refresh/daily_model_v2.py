#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:21:16 2022

@author: jinishizuka
"""

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import sqlite3
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, date
from collections import defaultdict

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from xgboost import XGBRegressor
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample

import pickle
from IPython.display import clear_output 

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


#convert fractional betting spread strings into floats

def convert_to_float(frac_str):
    if frac_str == 'PK':
        return 0.0
    try:
        return float(frac_str)
    except ValueError:
        output = frac_str[:-1]
        
        if output=='+':
            return 0.5
        elif output=='-':
            return -0.5
        
        if output[0]=='-':
            output = float(output) - 0.5
        elif output[0]=='+':
            output = float(output) + 0.5
        return output
    
start_time = datetime.now()

#import data
training_df = pd.read_csv('training_data.csv')
training_df.rename(columns={'HOME_TEAM_ID':'TEAM_ID_HOME', 'HOME_TEAM_CITY':'TEAM_CITY_HOME'}, inplace=True)


######### new code ##########

#Create standardized dataset
training_df.drop(columns=['Unnamed: 0',
                 'TEAM_ID_HOME',
                 'GAME_DATE',
                 'TEAM_CITY_HOME'], inplace=True)

non_feature_cols = {'GAME_ID', 'SPREAD', 'SEASON'}
feature_cols = set(training_df.columns) - non_feature_cols

#standardize features
sclr = StandardScaler()
training_df[list(feature_cols)] = sclr.fit_transform(training_df[list(feature_cols)])

#PCA to minimize multicollinearity
pca = PCA().fit(training_df[list(feature_cols)])
principal_components = np.arange(pca.n_components_)+1

#apply dimensionality reduction to data, keeping n components
n_components = 20
pca=PCA(n_components=n_components)

df_ids_spreads = training_df[list(non_feature_cols)].copy()

df_pca = pd.DataFrame(pca.fit_transform(training_df[list(feature_cols)]))
df_pca['GAME_ID'] = df_ids_spreads['GAME_ID']
df_pca['SPREAD'] = df_ids_spreads['SPREAD']
df_pca['SEASON'] = df_ids_spreads['SEASON']


############### check model on performance over last 5 seasons #############

print('Testing model on last 5 seasons...')

def season_string(season):
    return str(season) + '-' + str(season+1)[-2:]

date = date.today()
year = date.year
month = date.month

if month >= 9:
    cur_season = year
else:
    cur_season = year-1

test_seasons = []
season = cur_season
for _ in range(5):
    season_str = season_string(season)
    test_seasons.append(season_str)
    season -= 1

safetycheck_train_df = df_pca[~df_pca['SEASON'].isin(test_seasons)]
safetycheck_test_df = df_pca[df_pca['SEASON'].isin(test_seasons)]

safetycheck_y_train = safetycheck_train_df['SPREAD'].copy()
safetycheck_X_train = safetycheck_train_df.drop(columns=['SPREAD', 'GAME_ID', 'SEASON'])

safetycheck_y_test = safetycheck_test_df['SPREAD'].copy()
safetycheck_X_test = safetycheck_test_df.drop(columns=['SPREAD', 'GAME_ID', 'SEASON'])

safetycheck_xgb = pickle.load(open('model_latest.pkl', 'rb'))
safetycheck_xgb.fit(safetycheck_X_train, safetycheck_y_train)

test_score = safetycheck_xgb.score(safetycheck_X_test, safetycheck_y_test)

#NOTE: Best R-squared obtained during original model tuning was 13.2
print('R-squared score on last 5 seasons: ', test_score)

###############################################################

print('Retraining model on full data...')

#split data into X_train and y_train
y_train = df_pca['SPREAD'].copy()
X_train = df_pca.drop(columns=['SPREAD', 'GAME_ID', 'SEASON'])

#XGBoost implementation
xgb = pickle.load(open('model_latest.pkl', 'rb'))
xgb.fit(X_train, y_train)

#pickle updated model
pickle.dump(xgb, open('model.pkl', 'wb'))

#load data for evaluation
eval_df = pd.read_csv('eval_data.csv')
eval_df.drop(columns=['Unnamed: 0',
                 'GAME_DATE',
                 'SEASON',
                 'HOME_TEAM_ID',
                 'HOME_TEAM_CITY'], inplace=True)

#standardize features for eval data
eval_df[list(feature_cols)] = sclr.transform(eval_df[list(feature_cols)])

#pca on eval data
eval_df = pd.DataFrame(pca.transform(eval_df[list(feature_cols)]))

#generate spread predictions
y_pred = xgb.predict(eval_df)

#import betting data
spreads_df = pd.read_csv('daily_spreads_df.csv')
spreads_df.drop(columns=['Unnamed: 0'], inplace=True)

#compare predicted spreads against betting spreads

def eval_spreads(spreads, preds):
    
    output = spreads[['GM_DATE', 'HOME_TEAM', 'AWAY_TEAM']].copy()
    book_spread = np.empty(len(output))
    book_num = np.empty(len(output))
    
    for i, row in spreads.iterrows():
        
        away_spreads = np.empty(4)
        
        away_spreads[0] = convert_to_float(row['SPREAD_1_AWAY'])
        away_spreads[1] = convert_to_float(row['SPREAD_2_AWAY'])
        away_spreads[2] = convert_to_float(row['SPREAD_3_AWAY'])
        away_spreads[3] = convert_to_float(row['SPREAD_4_AWAY'])
      
        pred = preds[i]
        
        spread_diffs = np.empty(4)
        
        for j, spread in enumerate(away_spreads):
            spread_diffs[j] = abs(pred - spread)
        
        book_w_max_diff = np.nanargmax(spread_diffs)
        book_num[i] = book_w_max_diff
        book_spread[i] = away_spreads[book_w_max_diff]
        
    output['BOOK_NUM'] = book_num
    output['BOOK_SPREAD'] = book_spread
    output['PRED_SPREAD'] = preds
    output['SPREAD_DIFF'] = abs(output['PRED_SPREAD'] - output['BOOK_SPREAD'])
    
    return output

#generate dataframe for bet evaluation
bets = eval_spreads(spreads_df, y_pred)

#dictionary defining book names
book_dict = {0:'Bet MGM', 1:'Draft Kings', 2:'Fanduel Sportsbook', 3:'Caesars Sportsbook'}

#function to send email notifications
username = 'jnshzk@gmail.com'
password = 'weuuuoxmqwtgzrax'

def send_notification(bets_df, recipients, confidence_thresh=20):

    #filter df for only bets that meet confidence threshold criteria
    ##########UNCOMMENT LINE BELOW BEFORE USE#########
    #bets_df = bets_df[bets_df['SPREAD_DIFF'] >= confidence_thresh]
    
    
    #replace book numbers with book names
    bets_df['BOOK_NUM'].replace(to_replace=book_dict, inplace=True)
    bets_df.rename(columns={'BOOK_NUM':'BOOK_NAME'}, inplace=True)
    
    if len(bets_df)==0:
        print('No recommended bets available')
        return
    
    recommended_bet_on = []
    
    for i, row in bets_df.iterrows():
        home_team = row['HOME_TEAM']
        away_team = row['AWAY_TEAM']
        
        if row['PRED_SPREAD'] >= row['BOOK_SPREAD']:
            recommended_bet_on.append(home_team)
        else:
            recommended_bet_on.append(away_team)
    
    bets_df['RECOMMENDED_BET_ON'] = recommended_bet_on
    
    #create email notification
    #emaillist = [elem.strip().split(',') for elem in recipients]
    msg = MIMEMultipart('mixed')
    msg['Subject'] = 'NBA Bet Evaluations for {}'.format(date.today())
    msg['From'] = username
    
    text = 'Yo,\n Bet evaluations for today ({}) are shown below.\n  NOTE: The tested model only recommends placing bets where the spread difference is greater than 20, but evaluations for all games are shown for your convenience.'.format(date.today())
    part1 = MIMEText(text, 'plain')
    msg.attach(part1)
    
    html = """\
    <html>
      <head></head>
      <body>
        {0}
      </body>
    </html>
    """.format(bets_df.to_html())
    
    part2 = MIMEText(html, 'html')
    msg.attach(part2)
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    #removed password for public repo
    server.login(username, password)
    #server.sendmail(msg['From'], emaillist, msg.as_string())
    server.sendmail(msg['From'], recipients, msg.as_string())
    server.quit()

print('Sending notifications...')
#send notifications
mailing_list = ['jnshzk@gmail.com', 'victoreyo15@gmail.com']
send_notification(bets, mailing_list)

print('Complete')
print('Runtime: ', datetime.now() - start_time)




