#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:03:32 2022

@author: jinishizuka
"""

from nba_api.stats.static import players, teams
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import boxscorescoringv2

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import time as time
from time import sleep
from datetime import date
import datetime
from IPython.core.display import clear_output
import sqlite3

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.headless = True

################ helper functions ###############

def season_string(season):
    return str(season) + '-' + str(season+1)[-2:]


def update_team_basic_boxscores(conn, season):
    table_name = 'team_basic_boxscores'
    season_str = season_string(season)
        
    dfs = []
    for season_type in ['Regular Season', 'Playoffs']:
        team_gamelogs = leaguegamelog.LeagueGameLog(season=season_str, season_type_all_star=season_type).get_data_frames()[0]
        dfs.append(team_gamelogs)
        
    team_gamelogs_updated = pd.concat(dfs)
    team_gamelogs_updated['SEASON'] = season_str
    team_gamelogs_updated.drop(columns = ['SEASON_ID', 'VIDEO_AVAILABLE'], inplace=True)
    
    team_gamelogs_updated.to_sql(table_name, conn, if_exists='append', index=False)

    cur = conn.cursor()
    cur.execute('DELETE FROM {} WHERE rowid NOT IN (SELECT min(rowid) FROM {} GROUP BY TEAM_ID, GAME_ID)'.format(table_name, table_name))
    conn.commit()
    
    return None


def update_team_advanced_boxscores(conn, season, dates):
    table_name = 'team_advanced_boxscores'
    
    season_str = season_string(season)
    
    game_ids_not_added = []
    
    # Pull the GAME_IDs from my data
    game_ids_in_db = pd.read_sql('''SELECT DISTINCT team_basic_boxscores.GAME_ID FROM team_basic_boxscores
                INNER JOIN team_advanced_boxscores 
                ON team_basic_boxscores.GAME_ID = team_advanced_boxscores.GAME_ID
                AND team_basic_boxscores.TEAM_ID = team_advanced_boxscores.TEAM_ID
                WHERE SEASON = "{}" '''.format(season_str), conn)

    game_ids_in_db = game_ids_in_db['GAME_ID'].tolist()
    
    missing_game_ids = []
    if len(dates) != 0:
        for date in dates:
            gamelogs = leaguegamelog.LeagueGameLog(
                season=season_str, date_from_nullable=date, date_to_nullable=date).get_data_frames()[0]
            missing_game_ids.extend(gamelogs['GAME_ID'].unique())
            
    else:        
        # get up to date GAME_IDs
        to_date_game_ids = []
        for season_type in ['Regular Season', 'Playoffs']:
            to_date_gamelogs = leaguegamelog.LeagueGameLog(season=season_str, season_type_all_star=season_type).get_data_frames()[0]
            to_date_game_ids.extend(to_date_gamelogs['GAME_ID'].unique())
        
        # See which game_ids are missing
        missing_game_ids = set(to_date_game_ids) - set(game_ids_in_db)
        
    num_games_updated = len(missing_game_ids)
    print("num_games_updated:", num_games_updated)
    
    if num_games_updated == 0:
        print("All team advanced boxscores up to date in season {}".format(season_str))
        return None
    
    for game_id in tqdm(missing_game_ids, desc='progress'):
        try:
            boxscores = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id).get_data_frames()[1]
            boxscores.to_sql(table_name, conn, if_exists='append', index=False)
            sleep(2)
        except:
            game_ids_not_added.append(game_id)  
    
    cur = conn.cursor()
    cur.execute('DELETE FROM {} WHERE rowid NOT IN (SELECT max(rowid) FROM {} GROUP BY TEAM_ID, GAME_ID)'.format(table_name, table_name))
    conn.commit()
    
    return game_ids_not_added, missing_game_ids


#Define today's date (for testing setting date manually)
start_time = datetime.datetime.now()

test_mode = False

if test_mode==True:
    date = date.today() - datetime.timedelta(days=122)
else:
    date = date.today()

############### test section deleting most recent data ##################
con = sqlite3.connect('nba_refresh.db')
cur = con.cursor()

if test_mode==True:
    
    #delete data after date
    test_game_ids = pd.read_sql('select GAME_ID from team_basic_boxscores where GAME_DATE >= \'{}\''.format(str(date)),con)
    test_game_ids_tuple = [(x,) for x in test_game_ids['GAME_ID']]
    
    cur.executemany('DELETE FROM team_basic_boxscores WHERE GAME_ID=?', test_game_ids_tuple)
    con.commit()
    
    cur.executemany('DELETE FROM team_advanced_boxscores WHERE GAME_ID=?', test_game_ids_tuple)
    con.commit()
    
    team_basic_boxscores_df = pd.read_sql('select * from team_basic_boxscores', con)
    team_advanced_boxscores_df = pd.read_sql('select * from team_advanced_boxscores', con)

    team_boxscores_df = team_basic_boxscores_df.merge(team_advanced_boxscores_df, how='inner', on=['GAME_ID', 'TEAM_ID'])
    
    print('Data deleted after ', max(team_boxscores_df['GAME_DATE']))
    

################# Data refresh continued ###########################

#update boxscores with any missing data

year = date.year
month = date.month

if month >= 9:
    season = year
else:
    season = year-1
    
con = sqlite3.connect('nba_refresh.db')

######## TEMP DELETE IF NOT USING ANYMORE #############
'''
cur = con.cursor()
cur.execute('DELETE FROM team_advanced_boxscores WHERE GAME_ID == \'0022200515\'')
con.commit()

cur = con.cursor()
cur.execute('DELETE FROM team_basic_boxscores WHERE GAME_ID == \'0022200515\'')
con.commit()

cur = con.cursor()
cur.execute('DELETE FROM team_advanced_boxscores WHERE GAME_ID == \'0022200516\'')
con.commit()

cur = con.cursor()
cur.execute('DELETE FROM team_basic_boxscores WHERE GAME_ID == \'0022200516\'')
con.commit()
'''
#######################################################

update_team_basic_boxscores(con, season)
try:
    game_ids_not_added, game_ids_added = update_team_advanced_boxscores(con, season, [])
    print('Number of games missing: ', len(game_ids_not_added))
except:
    print('Database already up to date')


################### Data preprocessing ##########################

#pull in full updated datasets
team_basic_boxscores_df = pd.read_sql('select * from team_basic_boxscores', con)
team_advanced_boxscores_df = pd.read_sql('select * from team_advanced_boxscores', con)

######TEMP#######
#print('team advanced boxscores: ', team_advanced_boxscores_df[team_advanced_boxscores_df['GAME_ID']=='0022200515'])

team_boxscores_df = team_basic_boxscores_df.merge(team_advanced_boxscores_df, how='inner', on=['GAME_ID', 'TEAM_ID'])

#add home team flag
team_boxscores_df['HOME_TEAM'] = team_boxscores_df['MATCHUP'].str[4] == 'v'
team_boxscores_df['HOME_TEAM']

#rename and drop columns
team_boxscores_df.drop(columns=['TEAM_ABBREVIATION_x',
                                'TEAM_NAME_x',
                                'MATCHUP',
                                'TEAM_NAME_y',
                                'TEAM_ABBREVIATION_y',
                                'MIN_y'], inplace=True)
team_boxscores_df.rename(columns={'MIN_x':'MIN'}, inplace=True)

#manually calculate estimation of missing rebound percentage stats
oreb_pct_calc = np.empty(len(team_boxscores_df))
dreb_pct_calc = np.empty(len(team_boxscores_df))
reb_pct_calc = np.empty(len(team_boxscores_df))

print('estimating missing rebound percentage stats...')
for i, row in tqdm(team_boxscores_df.iterrows()):
    game_id = row['GAME_ID']
    team_id = row['TEAM_ID']
    
    opp_row = team_boxscores_df[team_boxscores_df['GAME_ID'] == game_id]
    opp_row = opp_row[opp_row['TEAM_ID'] != team_id]
    
    oreb_pct_calc[i] = row['OREB'] / (row['OREB'] + opp_row['DREB'])
    dreb_pct_calc[i] = row['DREB'] / (row['DREB'] + opp_row['OREB'])
    reb_pct_calc[i] = row['REB'] / (row['REB'] + opp_row['REB'])

team_boxscores_df['OREB_PCT_CALC'] = oreb_pct_calc
team_boxscores_df['DREB_PCT_CALC'] = dreb_pct_calc
team_boxscores_df['REB_PCT_CALC'] = reb_pct_calc

#fill in missing rebound percentage stats with calculated values
rebound_pct_cols = ['OREB_PCT', 'DREB_PCT', 'REB_PCT']

for col in rebound_pct_cols:
    team_boxscores_df[col].fillna(value=team_boxscores_df[col + '_CALC'], inplace=True)

team_boxscores_df.drop(columns=['OREB_PCT_CALC',
                                'DREB_PCT_CALC',
                                'REB_PCT_CALC'], inplace=True)

#store actual point spread for each game
game_ids = team_boxscores_df['GAME_ID'].unique()
spreads = np.empty(len(game_ids))

print('storing actual point spreads for added games...')
for i, game_id in tqdm(enumerate(game_ids)):
    spread = team_boxscores_df[(team_boxscores_df['GAME_ID']==game_id) &
                               (team_boxscores_df['HOME_TEAM']==True)]['PLUS_MINUS']
    spreads[i] = spread

spreads_df = pd.DataFrame(data={'GAME_ID':game_ids, 'SPREAD':spreads})

#change W/L column to 1's and 0's
team_boxscores_df['WL'] = team_boxscores_df['WL'].map({'W':1, 'L':0})

#convert game date to datetime
team_boxscores_df['GAME_DATE'] = pd.to_datetime(team_boxscores_df['GAME_DATE'])

#elo rating helper functions
#credit to rogerfitz

def get_K(MOV, elo_diff):
    """This K multiplier """
    K_0 = 20    

    if MOV > 0:
        multiplier = (MOV+3)**(0.8)/(7.5+0.006*(elo_diff))
    else:
        multiplier = (-MOV+3)**(0.8)/(7.5+0.006*(-elo_diff))
        
    return K_0*multiplier, K_0*multiplier

def get_S(team_score, opp_score):
    """S is 1 if the team wins, and 0 if the team loses"""
    S_team, S_opp = 0, 0
    if team_score > opp_score:
        S_team = 1
    else:
        S_opp = 1
    return S_team, S_opp


def elo_prediction(team_rating, opp_rating):
    """Generate the probability of a home victory based on the teams' elo ratings"""
    E_team = 1.0/(1 + 10 ** ((opp_rating - team_rating) / (400.0)))
    return E_team

def elo_update(team_score, opp_score, team_rating, opp_rating):
    # Add 100 to the home_rating for home court advantage   
    team_rating += 100
    
    E_team = elo_prediction(team_rating, opp_rating)
    E_opp = 1.0 - E_team
    
    MOV = team_score - opp_score
    if MOV > 0:
        elo_diff = team_rating - opp_rating
    else:
        elo_diff = opp_rating - team_rating
            
    S_team, S_opp = get_S(team_score, opp_score)
    
    K_team, K_opp = get_K(MOV, elo_diff)

    return K_team*(S_team-E_team), K_opp*(S_opp-E_opp)
    

def season_reset(rating):
    new_rating = 0.75*rating + 0.25*1505
    return new_rating

def add_elo_ratings(df):
    df.sort_values(['GAME_DATE', 'GAME_ID', 'HOME_TEAM'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    elo_col = np.empty(df.shape[0])
    elo_dict = {}
    cur_season = df.iloc[0]['SEASON']
    
    print('adding elo ratings...')
    for i, row in tqdm(df.iterrows()):
        
        if i%2 != 0:
            continue
        
        if row['SEASON'] != cur_season:
            cur_season = row['SEASON']
            elo_dict = {team_id:season_reset(elo) for team_id, elo in elo_dict.items()}
        
        away_row = row
        home_row = df.iloc[i+1]
        
        #check both rows are from same game
        if away_row['GAME_ID'] != home_row['GAME_ID']:
            print('game ids do not match')
            print('home game id: ', home_row['GAME_ID'])
            print('away game id: ', away_row['GAME_ID'])
            print('iteration: ', i)
            break
        
        home_team_id = home_row['TEAM_ID']
        away_team_id = away_row['TEAM_ID']
        
        if home_team_id not in elo_dict:
            elo_dict[home_team_id] = 1300
        if away_team_id not in elo_dict:
            elo_dict[away_team_id] = 1300

        home_elo = elo_dict[home_team_id]
        away_elo = elo_dict[away_team_id]

        elo_col[i+1] = home_elo
        elo_col[i] = away_elo
        
        home_pts = home_row['PTS']
        away_pts = away_row['PTS']

        home_elo_update, away_elo_update = elo_update(home_pts, away_pts, home_elo, away_elo)
        
        new_home_elo = home_elo + home_elo_update
        new_away_elo = away_elo + away_elo_update
        
        elo_dict[home_team_id] = new_home_elo
        elo_dict[away_team_id] = new_away_elo
    
    df['ELO'] = elo_col
        
    return df, elo_dict

#add elo ratings
team_boxscores_df, elo_dict = add_elo_ratings(team_boxscores_df)

#sort dataframe by date
team_boxscores_df.sort_values(by=['GAME_DATE'], ascending=True, inplace=True)

#take exponentially weighted moving average of stats for each game
num_games = 50

non_feature_cols = {'SEASON', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'TEAM_CITY', 'ELO'}
feature_cols = set(team_boxscores_df.columns) - non_feature_cols

weighted_avgs = []

print('computing weighted moving average of stats for each game...')
for i, row in tqdm(team_boxscores_df.iterrows()):
    team_id = row['TEAM_ID']
    game_date = row['GAME_DATE']

    temp_df = team_boxscores_df[(team_boxscores_df['TEAM_ID'] == team_id) &
                                (team_boxscores_df['GAME_DATE'] < game_date)].copy()
    temp_df.sort_values(by=['GAME_DATE'], ascending=True, inplace=True)
    temp_df = temp_df.tail(num_games)
    
    if len(temp_df) < num_games:
        continue
    
    temp_df[list(feature_cols)] = temp_df[list(feature_cols)].ewm(span=num_games).mean()
    
    row[list(feature_cols)] = temp_df.iloc[-1]
    
    weighted_avgs.append(row)

weighted_avg_df = pd.DataFrame(weighted_avgs)
weighted_avg_df.reset_index(drop=True, inplace=True)

#add number of rest days
rest_days = np.empty(weighted_avg_df.shape[0])

print('adding rest days...')
for i, row in tqdm(weighted_avg_df.iterrows()):
    game_date = row['GAME_DATE']
    team_id = row['TEAM_ID']
    rest_days_df = weighted_avg_df[(weighted_avg_df['TEAM_ID'] == team_id) &
                                     (weighted_avg_df['GAME_DATE'] < game_date)].copy()
    if len(rest_days_df) == 0:
        #assuming earliest game for each team was at the start of the 2000 season, so will assume 4 months rest since their last game of the 1999-2000 season
        rest_days[i] = 120
        continue
    
    rest_days_df.sort_values(by=['GAME_DATE'], ascending=False, inplace=True, ignore_index=True)
    last_game_date = rest_days_df.iloc[0]['GAME_DATE']
    
    delta = game_date - last_game_date
    rest_days[i] = delta.days
    
weighted_avg_df['REST_DAYS'] = rest_days

#reformat so each game is represented by a single row which is the difference between each team's stats
game_ids = weighted_avg_df['GAME_ID'].unique()

revised_rows = []
missing_game_ids = []
missing_game_count = 0

feature_cols.add('ELO')
feature_cols = list(feature_cols)

print('reformatting each games data into single row...')
for game_id in tqdm(game_ids):
    
    home_team_row = weighted_avg_df[(weighted_avg_df['GAME_ID']==game_id) &
                                      (weighted_avg_df['HOME_TEAM']==True)]
    away_team_row = weighted_avg_df[(weighted_avg_df['GAME_ID']==game_id) &
                                      (weighted_avg_df['HOME_TEAM']==False)]
    
    try:
        stats_diff = home_team_row[feature_cols].subtract(np.array(away_team_row[feature_cols]))
        stats_diff[['SEASON','GAME_DATE','GAME_ID','HOME_TEAM_ID','HOME_TEAM_CITY']] = home_team_row[['SEASON',
                                                                                                      'GAME_DATE',
                                                                                                      'GAME_ID',
                                                                                                      'TEAM_ID',
                                                                                                      'TEAM_CITY']]
        revised_rows.append(stats_diff)
    
    except:
        missing_game_ids.append(game_id)
        missing_game_count += 1

final_df = pd.concat(revised_rows)
print('Number of missing games: ', missing_game_count)

#add spread actuals
final_df = final_df.merge(spreads_df, how='inner', on=['GAME_ID'])

def check_missing_vals(df):
    cols_w_missing_vals = []
    for col in df.columns:
        if df[col].isna().sum() != 0:
            cols_w_missing_vals.append(col)
    #print('cols with missing vals: ', cols_w_missing_vals)
    #print('columns: ', df.columns)
    #print('rows with missing vals: ', df[df['WL'].isna()])
    #print(df.head())
    return cols_w_missing_vals

#check for missing values
if len(check_missing_vals(team_boxscores_df)) > 0:
    raise(ValueError('Data has missing values'))
    
#For test mode, filter updated data to only include up to the test date
if test_mode==True:
    final_df = final_df[final_df['GAME_DATE'] < str(date)]
    max(final_df['GAME_DATE'])

#save updated data as csv
final_df.to_csv('training_data.csv')



#################### Pull betting spreads and moneylines for current day's games #####################

#######REMOVE#########

from nba_api.stats.static import players, teams
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.endpoints import boxscorescoringv2

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import time as time
from time import sleep
from datetime import date
import datetime
from IPython.core.display import clear_output
import sqlite3

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

test_mode = False
date = date.today()
final_df = pd.read_csv('/Users/jinishizuka/nba_beating_the_spread/daily_refresh/training_data.csv')

#######REMOVE#########


#pull spreads and moneylines for the day
def pull_spreads(date):
    
    dates_with_no_data = []
    
    seasons = []
    gm_dates = []
    away_teams = []
    home_teams = []
    away_scoreboards = []
    home_scoreboards = []
    away_spreads = []
    home_spreads = []
    
    web = 'https://www.sportsbookreview.com/betting-odds/nba-basketball/?date={}'.format(date)
    #path = '../../Downloads/chromedriver'
    #path = '/Users/jinishizuka/Downloads/chromedriver'
    #driver = webdriver.Chrome(path)
    #driver.get(web)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(web)
    
    sleep(random.randint(1,2))

    try:
        #single_row_events = driver.find_elements_by_class_name('eventMarketGridContainer-3QipG')
        #single_row_events = driver.find_elements(By.CLASS_NAME, 'eventMarketGridContainer-3QipG')
        single_row_events = driver.find_elements(By.CLASS_NAME, 'GameRows_eventMarketGridContainer__GuplK')

    except:
        print("No Data for {}".format(date))
        dates_with_no_data.append(date)

    #num_postponed_events = len(driver.find_elements_by_class_name('eventStatus-3EHqw'))
    num_postponed_events = len(driver.find_elements(By.CLASS_NAME, 'eventStatus-3EHqw'))

    num_listed_events = len(single_row_events)
    cutoff = num_listed_events - num_postponed_events
    
    #print('single row events: ', single_row_events)
    #print('num_listed_events: ', num_listed_events)
    #print('cutoff: ', cutoff)

    for event in single_row_events[:cutoff]:
        #away_team = event.find_elements_by_class_name('participantBox-3ar9Y')[0].text
        #home_team = event.find_elements_by_class_name('participantBox-3ar9Y')[1].text
        away_team = event.find_elements(By.CLASS_NAME, 'GameRows_participantBox__0WCRz')[0].text
        home_team = event.find_elements(By.CLASS_NAME, 'GameRows_participantBox__0WCRz')[1].text
        away_teams.append(away_team)
        home_teams.append(home_team)
        gm_dates.append(date)

        #scoreboard = event.find_elements_by_class_name('scoreboard-1TXQV')
        scoreboard = event.find_elements(By.CLASS_NAME, 'scoreboard-1TXQV')
        
        #print('scoreboard: ', scoreboard)
        
        home_score = []
        away_score = []

        for score in scoreboard:
            #quarters = score.find_elements_by_class_name('scoreboardColumn-2OtpR')
            quarters = score.find_elements(By.CLASS_NAME, 'scoreboardColumn-2OtpR')
            for i in range(len(quarters)):
                scores = quarters[i].text.split('\n')
                away_score.append(scores[0])
                home_score.append(scores[1])

            home_score = ",".join(home_score)
            away_score = ",".join(away_score)

            away_scoreboards.append(away_score)
            home_scoreboards.append(home_score)


        if len(away_scoreboards) != len(away_teams):
            num_to_add = len(away_teams) - len(away_scoreboards)
            for i in range(num_to_add):
                away_scoreboards.append('')
                home_scoreboards.append('')

        #spreads = event.find_elements_by_class_name('pointer-2j4Dk')
        spreads = event.find_elements(By.CLASS_NAME, 'OddsCells_pointer___xLMm')
        
        away_lines = []
        home_lines = []
        for i in range(len(spreads)):    
            if i % 2 == 0:
                away_lines.append(spreads[i].text)
            else:
                home_lines.append(spreads[i].text)

        away_lines = ",".join(away_lines)
        home_lines = ",".join(home_lines)

        away_spreads.append(away_lines)
        home_spreads.append(home_lines)

        if len(away_spreads) != len(away_teams):
            num_to_add = len(away_teams) - len(away_spreads)
            for i in range(num_to_add):
                away_scoreboards.append('')
                home_scoreboards.append('')

    driver.quit()
    clear_output(wait=True)

    df = pd.DataFrame({'GM_DATE':gm_dates,
                      'AWAY_TEAM':away_teams,
                      'HOME_TEAM':home_teams,
                      'AWAY_SCOREBOARD':away_scoreboards,
                      'HOME_SCOREBOARD':home_scoreboards,
                      'AWAY_SPREAD':away_spreads,
                      'HOME_SPREAD':home_spreads})

    df = df.sort_values(['GM_DATE']).reset_index(drop=True)
    
    #print('game dates: ', gm_dates)
    #print('away team: ', away_teams)
    #print('home team: ', home_teams)
    #print('away scoreboard: ', away_scoreboards)
    #print('home scoreboard: ', home_scoreboards)
    #print('away spread: ', away_spreads)
    #print('home_spread: ', home_spreads)
    #print('betting df head: ', df.head())
    return df

#pull betting data
spreads_df = pull_spreads(date)

#THIS IS FOR TESTING
if test_mode==True:
    team_boxscores_df = team_boxscores_df[team_boxscores_df['GAME_DATE'] < str(date)]
    max(team_boxscores_df['GAME_DATE'])


##################### Create model inputs for prediction #######################3

#function to create model input for a given game
#i.e. turns any given game into the difference of each team's average stats and elo and adds rest days

def create_model_input(home_team, away_team, game_date):
    
    if home_team == 'L.A. Clippers':
        home_team = 'LA'
    elif home_team == 'L.A. Lakers':
        home_team = 'Los Angeles'
        
    if away_team == 'L.A. Clippers':
        away_team = 'LA'
    elif away_team == 'L.A. Lakers':
        away_team = 'Los Angeles'
    
    non_feature_cols = {'SEASON', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'TEAM_CITY', 'ELO'}
    feature_cols = set(team_boxscores_df.columns) - non_feature_cols
    
    team_boxscores_df.sort_values(by=['GAME_DATE'], ascending=True, inplace=True)
    
    #pull each team's last n games
    home_team_df = team_boxscores_df[team_boxscores_df['TEAM_CITY']==home_team].tail(num_games).copy()
    away_team_df = team_boxscores_df[team_boxscores_df['TEAM_CITY']==away_team].tail(num_games).copy()
    
    #compute weighted average of each team's stats over the last n games
    home_team_df[list(feature_cols)] = home_team_df[list(feature_cols)].ewm(span=num_games).mean()
    away_team_df[list(feature_cols)] = away_team_df[list(feature_cols)].ewm(span=num_games).mean()
    
    home_row = home_team_df.iloc[-1].copy()
    away_row = away_team_df.iloc[-1].copy()
    
    #add rest days
    home_row['REST_DAYS'] = game_date - home_row['GAME_DATE'].to_pydatetime().date()
    away_row['REST_DAYS'] = game_date - away_row['GAME_DATE'].to_pydatetime().date()

    non_feature_cols.remove('ELO')
    feature_cols = list(set(team_boxscores_df.columns) - non_feature_cols)
    
    #compute difference between home team stats and away team stats
    output_row = home_row[feature_cols].subtract(np.array(away_row[feature_cols]))
    
    return output_row

#create model inputs for current day's games

test_data = []

for i, game in spreads_df.iterrows():
    home_team = game['HOME_TEAM']
    away_team = game['AWAY_TEAM']
    game_date = game['GM_DATE']

    
    test_row = create_model_input(home_team, away_team, game_date)
    test_data.append(test_row)
    
test_df = pd.DataFrame(test_data)

#reformat test_df to be same format as training data
test_df['SEASON'] = np.nan
test_df['GAME_ID'] = np.nan
test_df['GAME_DATE'] = np.nan

test_df = test_df.reindex(columns=list(final_df.columns))

#save test data
test_df.to_csv('eval_data.csv')


############################## Reformat spreads data ###########################

spreads_df.drop(columns=['AWAY_SCOREBOARD', 'HOME_SCOREBOARD'], inplace=True)

#REMOVE
#print(spreads_df.head())

spreads_df[['BOOK_1_AWAY', 'BOOK_2_AWAY', 'BOOK_3_AWAY', 'BOOK_4_AWAY', 'DISCARD_AWAY']] = spreads_df['AWAY_SPREAD'].str.split(pat=',', expand=True, n=4)
spreads_df[['BOOK_1_HOME', 'BOOK_2_HOME', 'BOOK_3_HOME', 'BOOK_4_HOME', 'DISCARD_HOME']] = spreads_df['HOME_SPREAD'].str.split(pat=',', expand=True, n=4)


spreads_df.drop(columns=['AWAY_SPREAD', 'HOME_SPREAD','DISCARD_AWAY','DISCARD_HOME'], inplace=True)

spreads_df['SPREAD_1_AWAY'] = spreads_df['BOOK_1_AWAY'].astype(str).str[:-4]
spreads_df['ODDS_1_AWAY'] = spreads_df['BOOK_1_AWAY'].astype(str).str[-4:]
spreads_df['SPREAD_2_AWAY'] = spreads_df['BOOK_2_AWAY'].astype(str).str[:-4]
spreads_df['ODDS_2_AWAY'] = spreads_df['BOOK_2_AWAY'].astype(str).str[-4:]
spreads_df['SPREAD_3_AWAY'] = spreads_df['BOOK_3_AWAY'].astype(str).str[:-4]
spreads_df['ODDS_3_AWAY'] = spreads_df['BOOK_3_AWAY'].astype(str).str[-4:]
spreads_df['SPREAD_4_AWAY'] = spreads_df['BOOK_4_AWAY'].astype(str).str[:-4]
spreads_df['ODDS_4_AWAY'] = spreads_df['BOOK_4_AWAY'].astype(str).str[-4:]

spreads_df['SPREAD_1_HOME'] = spreads_df['BOOK_1_HOME'].astype(str).str[:-4]
spreads_df['ODDS_1_HOME'] = spreads_df['BOOK_1_HOME'].astype(str).str[-4:]
spreads_df['SPREAD_2_HOME'] = spreads_df['BOOK_2_HOME'].astype(str).str[:-4]
spreads_df['ODDS_2_HOME'] = spreads_df['BOOK_2_HOME'].astype(str).str[-4:]
spreads_df['SPREAD_3_HOME'] = spreads_df['BOOK_3_HOME'].astype(str).str[:-4]
spreads_df['ODDS_3_HOME'] = spreads_df['BOOK_3_HOME'].astype(str).str[-4:]
spreads_df['SPREAD_4_HOME'] = spreads_df['BOOK_4_HOME'].astype(str).str[:-4]
spreads_df['ODDS_4_HOME'] = spreads_df['BOOK_4_HOME'].astype(str).str[-4:]

spreads_df.drop(columns=['BOOK_1_AWAY','BOOK_2_AWAY','BOOK_3_AWAY','BOOK_4_AWAY','BOOK_1_HOME','BOOK_2_HOME','BOOK_3_HOME','BOOK_4_HOME'], inplace=True)

spreads_df.drop_duplicates(inplace=True)

spreads_df.to_csv('daily_spreads_df.csv')

print('Runtime : ', datetime.datetime.now() - start_time)







