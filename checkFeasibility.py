from prepareData import *
import numpy as np
import matplotlib.pyplot as plt

useData = dataWithWeatherInformation # from the prepareData script

def plotTimeSeries(dataY, dataX=None, _label="scatter"):
    # if no xAxis is given mock the axis
    if dataX is None:
        dataX = np.array(list(np.arange(1, dataY.size+1)))

    plt.scatter(dataX, dataY, marker="x", label=_label)

    plt.legend(loc='upper right')
    plt.show()

# given the data np array it calcs the autocorrelation of the values of the col with the Index colIndex
def checkAutocorrelation(data: np.array, colIndex = 0, _chartLabel=None):
    # calc the autocorrelation
    column = np.array(data[:, colIndex], dtype=float)
    result = np.correlate(column, column, mode='full')
    result = result[result.size//2:]

    # plot the results
    chartLabel = "normalized Autocorrelation of " + _chartLabel
    plt.plot(result / float(result.max()), color="red", label=chartLabel)
    plt.legend()
    plt.show()


# draw the plain production data
dataY = np.array(useData[:4320,1], dtype=float) #three days in a row
dataX = np.array(useData[:4320,0], dtype=float)
plotTimeSeries(dataY, dataX, "production data of 05/04/20-08/04/20")

dataX = np.array(useData[:,0], dtype=float) # all available data
dataY = np.array(useData[:,1], dtype=float) # all available data
plotTimeSeries(dataY, dataX, "production data of 05/04/20-15/05/20")

# check the autocorrelations
checkAutocorrelation(useData, 1, " 05/04/20-15/05/20")
data = useData[:4320,:]
checkAutocorrelation(data, 1, "05/04/20-08/04/20")

print("debug checkFeasibility")