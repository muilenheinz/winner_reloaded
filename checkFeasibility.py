from prepareData import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

useData = dataWithWeatherInformation # from the prepareData script

def plotTimeSeries(dataY, dataX=None, _label="scatter"):
    # if no xAxis is given mock the axis
    if dataX is None:
        dataX = np.array(list(np.arange(1, dataY.size+1)))
    else:
        # append date range as info to the legend
        beginDatetimeValue = datetime.fromtimestamp(dataX[0] / 1000)
        endDatetimeValue = datetime.fromtimestamp(dataX[dataX.size - 1] / 1000)
        printableBegin = beginDatetimeValue.strftime("%d/%m/%Y")
        printableEnd = endDatetimeValue.strftime("%d/%m/%Y")
        _label = _label + " of " + printableBegin + "-" + printableEnd

    plt.scatter(dataX, dataY, marker=".", label=_label)

    plt.legend(loc='upper right')
    plt.show()

# given the data np array it calcs the autocorrelation of the values of the col with the Index colIndex
def checkAutocorrelation(dataY: np.array, dataX: np.array,  _chartLabel=None):
    result = np.correlate(dataY, dataY, mode='full')
    result = result[result.size//2:]
    normalizedResult = result / float(result.max())

    plotTimeSeries(normalizedResult, dataX, "normalized Autocorrelation")


# given an 1D-np.array calcs the deviation of a value from its predecessor
def calcDifferenceSeries(dataY: np.array, dataX: np.array):
    result = np.diff(dataY)
    plotTimeSeries(result, dataX[:-1], "differential data")

# get sample data
threeDaysDataX = np.array(useData[:4320,0], dtype=float)
threeDaysDataY = np.array(useData[:4320,1], dtype=float) #three days in a row
threeDaysData = np.array(useData[:4320,1], dtype=float)
allDaysDataX = np.array(useData[:,0], dtype=float) # all available data
allDaysDataY = np.array(useData[:,1], dtype=float) # all available data
allDaysData = np.array(useData[:,1], dtype=float)

# draw the plain production data
plotTimeSeries(threeDaysDataY, threeDaysDataX, "production data")
plotTimeSeries(allDaysDataY, allDaysDataX, "production data")

# check the autocorrelations
checkAutocorrelation(threeDaysDataY, threeDaysDataX)
checkAutocorrelation(allDaysDataY, allDaysDataX)

# check the difference values
calcDifferenceSeries(threeDaysDataY, threeDaysDataX)
calcDifferenceSeries(allDaysDataY, allDaysDataX)

print("debug checkFeasibility")