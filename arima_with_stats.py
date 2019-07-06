from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import preprocess
import stats


def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


def getSizeOf(X):
    return int(len(X) * 0.8)


def getTrainTestData(X):
    return X[0:size], X[size:len(X)]


def getModel(history):
    return ARIMA(history, order=(10, 1, 0))


def addIntoTuple(tupelToChange, valueToAdd):
    temp = list(tupelToChange)
    temp[2] = valueToAdd
    return tuple(temp)


def getTuplesToArrayOfLists(data):
    return np.array([[i[0], i[1], i[2]] for i in data])


def getPredictions():
    return finalData[:, 2]


def getTestData():
    return finalData[:, 0]


series = read_csv('Dataset/TITAN.NS.csv')
series = preprocess.getBuySellCalls(series, "Adj Close")
X = series["Adj Close"]
y = series["Expected"]
X.dropna(inplace=True)
X = X.tolist()
y = y.tolist()
zipped = list(zip(X, y, [None] * len(y)))
size = getSizeOf(zipped)
train, test = getTrainTestData(zipped)
history = [x[0] for x in train]
print(history)
for t in range(len(test)):
    model = getModel(history)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    test[t] = addIntoTuple(test[t], yhat[0])
    obs = test[t][0]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

finalData = getTuplesToArrayOfLists(test)
predictions = getPredictions()
plotTest = getTestData()
error = mean_squared_error(plotTest, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(plotTest)
pyplot.plot(predictions, color='red')
pyplot.show()

statsObject = stats.stats(finalData[:-1], plotTest[:-1], predictions[:-1])
accuracy, precision, recall = statsObject.getStats()
print(accuracy, precision, recall)
