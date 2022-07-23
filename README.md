# Winning NBA Spread Betting

For this project, I wanted to create a model to determine which bets to place on NBA games in order to maximize winnings.  My basic approach for this task was to fist create a regression model to predict the point spread of a given NBA game based on historic boxscore statistics for each team and then compare that predicted point spread against the betting spreads for that game to determine which team to bet on.

## Data Collection and Preparation

I began by pulling historic NBA boxscore data via the NBA.com API as well as historic betting spreads and moneylines from www.sportsbookreview.com.  The collected data spanned from the 2000-01 season through the 2021-22 season.  The majority of the data scraping code was provided by https://github.com/jnish23 who worked on a similar project.  The data collection process took several days as frequent sleep statements had to be inserted into the code to avoid timeout errors.  

After consolidating the collected data, the dataset consisted of 56,304 rows and 50 columns.  From here, the data exploration and feature engineering portion of the project began.  This started with checking for features with missing values of which there were only 3 (offensive rebound percentage, defensive rebound percentage, and rebound percentage).  Since each of these statistics is computed using rebound statistics, which were available, I was able to replicate these missing values manually.  After imputing the missing values, I did some simple data preparation (e.g. computed the point spread for each game, converted the Win/Loss column to 1's and 0's, reformtted the data so each game was represented by a single row instead of a row for each team etc.).

Next, I experimented with different methods of representing the data for each game.  Since I obviously would not have the actual statistics of a game prior placing bets on a game, I knew I would have to represent each team as some sort of average of their past performance. After experimenting with different representations (simple moving average, weighted average, etc.), I settled on taking an exponentially weighted moving average of each team's statistics over their last 50 games.  This average gave a higher weight to each team's most recent performance.  Additionally, in order to minimize the number of columns, I took the difference between the home and away teams' statistics (so if the home team was averaging 5 more points per game than the away team, there would only be a single 'points' feature whose value would be 5).

I also added a couple of engineered features including number of rest days and ELO ratings.  The number of rest days was a simple addition and consists of computing the number of days since a team's last game.  Adding ELO ratings was a bit more involved.  ELO ratings are often used to evaluate the relative strength of different teams in sports (more background can be found here: https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/).  Since ELO ratings take into account the strength of a team's opponents, I felt that this would be a valuable addition to my data.  Teams may go on stretches where they face consistently strong or weak opponents which may skew the performance statistics for that team.  It was my hope that ELO ratings would help account for the difficulty of teams' past schedules. Credit to https://github.com/rogerfitz for the code to calculate ELO ratings.

After the data prep and feature engineering were completed, I quickly checked the distributions of my final dataset to make sure there were no anomolies in the data.  At this stage, the dataset consisted of 27,363 rows and 52 columns.

As a final step in the data preparation, I wanted to conduct some dimensionality reduction through PCA in order to minimize multicollinearity concerns.  To begin, I plotted the cumulative explained variance and explained variance as a function of the number of principal components:

![Alt text](https://github.com/jnshzk/nba_beating_the_spread/images/pca_cum_exp_var_plot.png?raw=true)

![Alt text](https://github.com/jnshzk/nba_beating_the_spread/images/pca_scree_plot.png?raw=true)
