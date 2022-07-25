# Winning NBA Spread Betting

For this project, I wanted to create a model to determine which bets to place on NBA games in order to maximize winnings.  My basic approach for this task was to first create a regression model to predict the point spread of a given NBA game based on historic boxscore statistics for each team and then compare that predicted point spread against the betting spreads to determine which which team to bet on.

## Data Collection

I began by pulling historic NBA boxscore data via the NBA.com API as well as historic betting spreads and moneylines from www.sportsbookreview.com.  The collected data spanned from the 2000-01 season through the 2021-22 season.  The majority of the data scraping code was provided by https://github.com/jnish23 who worked on a similar project.  The data collection process took several days as frequent sleep statements had to be inserted into the code to avoid timeout errors.  

## Data Pre-processing
After consolidating the collected data, the dataset consisted of 56,304 rows and 50 columns.  From here, the data exploration and feature engineering portion of the project began.  This started with checking for features with missing values of which there were only 3 (offensive rebound percentage, defensive rebound percentage, and rebound percentage).  Since each of these percentages is computed using rebound statistics, which were available, I was able to replicate these missing values manually.  After imputing the missing values, I did some simple data preparation (e.g. computed the point spread for each game, converted the Win/Loss column to 1's and 0's, reformtted the data so each game was represented by a single row instead of a row for each team etc.).

***Feature Engineering***

Next, I experimented with different methods of representing the data for each game.  Since I obviously would not have the actual statistics of a game prior to placing bets on the game, I knew I would have to represent each team as some sort of average of their past performance. After experimenting with different representations (simple moving average, weighted average, etc.), I settled on taking an exponentially weighted moving average of each team's statistics over their last 50 games.  This average gave a higher weight to each team's most recent performance.  Additionally, in order to minimize the number of columns, I took the difference between the home and away teams' statistics (so if the home team was averaging 5 more points per game than the away team, there would be a single 'points' feature whose value would be 5).

I also added a couple of engineered features including number of rest days and ELO ratings.  The number of rest days was a simple addition and consisted of computing the number of days since a team's last game.  Adding ELO ratings was a bit more involved.  ELO ratings are often used to evaluate the relative strength of different teams in sports (more background can be found here: https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/).  Since ELO ratings take into account the strength of a team's opponents, I felt that this would be a valuable addition to my data.  Teams may go on stretches where they face consistently strong or weak opponents which may skew the performance statistics for that team.  It was my hope that ELO ratings would help account for the difficulty of teams' past schedules. Credit to https://github.com/rogerfitz for the code to calculate ELO ratings.

Next, I quickly checked the distributions of my dataset to make sure there were no anomolies in the data.  At this stage, the dataset consisted of 27,363 rows and 52 columns.

***Principal Component Analysis***

As a final step in the data preparation, I wanted to conduct dimensionality reduction through PCA in order to minimize multicollinearity concerns.  Many of the features in my dataset seemed likely to be highly correalted (e.g. rebounds, defensive rebounds, offensive rebounds, rebound percentage, etc.), so dimensionnality reduction seemed to be a logical next step.  To begin, I plotted the explained variance as a function of the number of principal components:

![pca cumulative explained variance](/images/pca_cum_exp_var_plot.png?raw=true)

![pca scree plot](/images/pca_scree_plot.png?raw=true)

Here, we can see that the majority of explained variance can be represented using 20 principal components.  Accordingly, I reduced the number of features in my dataset to 20 via PCA.

## Model Implementation

For the model implementation portion of this project, I experimented with several models.  These included linear and polynomial regressions (of varying degrees), k-nearest neighbors, random forest, and xgboost.  I trained each model on data from the 2000-01 season through 2016-17 season, reserving the 2017-18 through 2021-22 seasons as test data. Each model was trained to predict the point spread of a given game.  This predicted spread was then compared against the betting spreads for that game.  Since the data I pulled consisted of spreads and moneylines from 4 different books, I selected whichever book had a betting spread with the largest absolute difference with my predicted spread.  All spreads used in this model were away team spreads.  So if my model predicted an away team spread greater than the book's away team spread, I bet on the home team.  Otherwise, I bet on the away team.  

In order to evaluate the models, I wrote a function to evaluate the winnings and losses of each model using the historic moneylines from the test seasons.  I set a constant bet amount of $10.  The primary metric for evaluation I used was ROI.  Since the initial investment could be any arbitrary amount, I defined the initial investment to be the amount needed to cover any losses incured while betting with the model.  For example, if the selected model initially placed a couple losing bets and incured a debt of $20 but then remained profitable from that point forward, then my initial investment would be set at $30 (the debt + the bet amount).  If the model ended the test seasons with $60 in winnings, then the ROI for that model would be 100% ( (60-30) / 30 = 1).

Initially, none of my models were profitable when evaluated on the test seasons.  This led to some of the feature engineering described in the previous section (adding features, re-weighting the average statistics, etc.).  However, even after experimenting with these adjustments, I still struggled to create a model that would generate a profit.  I finally tried implementing a confidence threshold for my betting.  This confidence threshold would only allow bets when there was a large enough difference between the model's predicted spread and the betting spread.  After some trial and error, I eventually settled on a confidence threshold of 20.  This means that I would only place bets on games where there was at least a 20 point discrepancy beetween the predicted spread and the betting spread.  This is obviously a high threshold with only 1-3% of all possible bets meeting this criteria.  However, with this threshold, my models saw a dramatic improvement in profitability.  Results at this stage are shown below:

| Model | Win Percentage | Percent of Bets Placed | Profit | ROI |
| ----- | -------------- | ---------------------- | ------ | --- |
| Linear Regression | 61.6% | 1.2% | 127.25 | 324% |
| Polynomial Regression (degree=2) | 61.1% | 1.5% | 137.22 | 357% |
| KNN | 51.4% | 3.5% | -77.87 | -157% |
| Random Forest | 57.7% | 1.3% | 76.98 | 6% |
| XGBoost | 57.7% | 2.1% | 125.31 | 330% |

Surprisingly, a simple linear regression combined with a high confidence threshold saw very strong performance, and a polynomial regression (with degree=2) saw the highest ROI out of all tested models.  However, since both linear and polynomial regression offer very few options when it comes to hyperparameter tuning, I concentrating my tuning efforts on the XGBoost model which also recorded a strong ROI.

***Model Tuning and Final Results***

In order to tune my XGBoost model, I used the hyperopt library.  This method of hyperparameter tuning uses a form of Bayesian optimization.  We first defined a hyperparameter space as well as an objective function, which in this case was the coefficient of determinateion (R-squared) of the model.  Hyperopt then generates a random initial point in the parameter space and evaluates the value of the objective function at this point.  Using results from this trial and past trials, hyperopt will then attempt to build a conditional probability model to determine another point in the parameter space that will likely yeild a better result.  This process is repeated until the stop criteria (in this case number of iterations=100) is satisfied.  After tuning my XGBoost model, I obtained the following results:


<img src="/images/xgboost_tuned_results.png" width="500">

Here, we can see that I was able to beat the ROI scores of the polynomial regression model and generate a nearly 4x ROI over the course of the last 5 NBA seasons.  There is still a fair amount of fluctuation in winnings due to the inherent unpredictability of NBA games, but there is still a clear and consistent upward trend in the simulated bankroll.  Additionally, due to the high confidence threshold, we can see that there are stretches where no bets are placed.

## Next Steps

There are a number of improvements that can be made to this model.  First is the inclusion of player-level data.  The current iteration of this model only leverages team-level statistics.  Therefore, factors such as player injury, player rest, and player trades are not taken into account.  Team-level boxscore statistics and ELO ratings will eventually adapt to roster changes, but this can take several games.  Incorporating real-time roster changes could potentially lead to a significant improvement in performance.

Productionalizing this model to send alerts about potential bets would also be a potential next step to take.  I would eventually like to set up an automated script to run during the NBA season.  This script would refresh the training data with the latest statistics and would evaluate the available bets for that day's games.  If the model identifies a bet that meets the threshold criteria, an email alert could be sent that would alert users to the bet.
