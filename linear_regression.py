import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# load and investigate the data 
df = pd.read_csv('tennis_stats.csv')
print(df.head())
columns_names = list(df.columns)
# series = {}
# for col in columns_names:
#   series[col] = df[col]
offensive = []
off_num = np.array([2, 3, 5, 7, 9, 11, 12, 16, 17, 19])
defensive = []
def_num = np.array([4, 6, 8, 10, 13, 14, 15, 18])
for num in off_num:
  offensive.append(columns_names[num])
for num in def_num:
  defensive.append(columns_names[num])
joined_list = offensive + defensive
#plot each feature to winning:
for feature in joined_list:
  df.plot(kind = 'scatter', x = feature, y = 'Winnings')
  plt.show()
  plt.clf()

#single feature Linear Regression
#BreakPointsFaces
x1 = df.BreakPointsFaced
x1 = x1.values.reshape(-1,1)
x = df[joined_list]
y = df['Winnings']
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, train_size = 0.8, test_size = 0.2)
regrs = LinearRegression()
regrs.fit(x1_train, y1_train)
y1_predicted = regrs.predict(x1_test)
print(regrs.score(x1_test, y1_test))
plt.scatter(x1_test, y1_test, label = 'data')
plt.plot(x1_test, y1_predicted, color = 'orange', label = 'fit')
plt.legend()
plt.show()
plt.clf()

#Aces
x2 = df.Aces
x2 = x2.values.reshape(-1, 1)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, train_size = 0.8, test_size = 0.2)
regrs.fit(x2_train, y2_train)
y2_predicted = regrs.predict(x2_test)
print(regrs.score(x2_test, y2_test))
plt.scatter(x2_test, y2_test, label = 'data')
plt.plot(x2_test, y2_predicted, color = 'orange', label = 'fit')
plt.legend()
plt.show()
plt.clf()

#Two Features Linear Regression
#BreakPointsOpportunites, ReturnGamesPlayed
x3 = df[['BreakPointsOpportunities', 'ReturnGamesPlayed']]
x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y, train_size = 0.8, test_size = 0.2)
regrs.fit(x3_train, y3_train)
y3_predicted = regrs.predict(x3_test)
print(regrs.score(x3_test, y3_test))

#Multiple Label Linear Regression
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
regrs.fit(x_train, y_train)
y_predicted = regrs.predict(x_test)
print(regrs.score(x_test, y_test))

#Maximizing the score
print(joined_list)
x_good = df[['FirstServe', 'SecondServePointsWon', 'Aces', 'BreakPointsSaved', 'DoubleFaults', 'TotalServicePointsWon', 'SecondServeReturnPointsWon', 'BreakPointsOpportunities', 'TotalPointsWon']]
x_train_good, x_test_good, y_train_good, y_test_good = train_test_split(x_good, y, train_size = 0.8, test_size = 0.2, random_state = 42)
regrs.fit(x_train_good, y_train_good)
y_good_predicted = regrs.predict(x_test_good)
print(regrs.score(x_test_good, y_test_good))
