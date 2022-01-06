from prepareData import *
import numpy as np
import matplotlib.pyplot as plt

useData = finalData # from the prepareData script

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


checkAutocorrelation(useData, 0, "all input data")
data = useData[:4320,:]
checkAutocorrelation(data, 0, "three days")

print("debug checkFeasibility")