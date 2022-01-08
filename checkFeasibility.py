from prepareData import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import kendalltau
import matplotlib.dates as md

plotColor = "blue"

# plot the dataY over the given dataX. If no dataX is given the axis is mocked as integer values with step 1
# @_label: describes the series, will be printed on the chart
# @_type: charttype, options are scatter (single dots), line and bar
def plotTimeSeries(dataY, dataX=None, _label="timeseries", _type="scatter"):
    global plotColor
    isTimestampXAxis = checkIsTimestampArray(dataX)

    if isTimestampXAxis:
        # append date range as info to the legend
        beginDatetimeValue = datetime.fromtimestamp(dataX[0] / 1000)
        endDatetimeValue = datetime.fromtimestamp(dataX[dataX.size - 1] / 1000)
        printableBegin = beginDatetimeValue.strftime("%d/%m/%Y")
        printableEnd = endDatetimeValue.strftime("%d/%m/%Y")
        _label = _label + " (" + printableBegin + "-" + printableEnd + ")"

        # convert timestamp axis to datetime axis
        datetimeXAxis = np.empty(dataX.shape, dtype='datetime64[s]')
        for key in range(dataX.shape[0]):
            # convert timestamp in microsenconds to ts in seconds and then to datetime object
            datetimeXAxis[key] = datetime.fromtimestamp(dataX[key] / 1000)
        dataX = datetimeXAxis

    # if no xAxis is given mock the axis
    if dataX is None:
        dataX = np.array(list(np.arange(1, dataY.size+1)))

    if _type == "scatter":
        plt.scatter(dataX, dataY, marker=".", label=_label, color=plotColor)
    elif _type == "bar":
        plt.bar(dataX, dataY, label=_label, color=plotColor)
        plt.xticks(rotation=45, ha="right")
    elif _type == "line":
        plt.plot(dataX, dataY, label=_label, color=plotColor)

    # format timestamp values on the xAxis
    if isTimestampXAxis:
        dtFmt = md.DateFormatter('%d/%m/%Y')
        plt.gca().xaxis.set_major_formatter(dtFmt)
        plt.xticks(rotation=15, ha="right")

    plt.legend(loc='upper right')
    plt.show()


# check if the given array contains timestamps
def checkIsTimestampArray(dataX: np.array):
    if dataX is None:
        return False

    # check if axis contains float values
    try:
        float(dataX[0])
    except:
        return False

    # check for a appropriate value range (timestamp given in microsensonds)
    if int(dataX[0]) < 1000000000000:
        return False

    return True

# calcs the autocorrelation of the values of dataY,
# dataX contains the corresponding timestamps used later for plotting
def calcAutocorrelation(dataY: np.array, dataX: np.array, _chartLabel=None):
    result = np.correlate(dataY, dataY, mode='full')
    result = result[result.size//2:]
    normalizedResult = result / float(result.max())
    if _chartLabel is None:
        _chartLabel = "normalisierte Autokorrelation"
    else:
        _chartLabel = "normalisierte Autokorrelation " + _chartLabel
    plotTimeSeries(normalizedResult, dataX, _chartLabel, "line")

# for the 1D-np.array dataY calcs the deviation of a value from its predecessor
def calcDifferenceSeries(dataY: np.array, dataX: np.array, _label = ""):
    result = np.diff(dataY)
    label = "Differenzdaten " + _label
    plotTimeSeries(result, dataX[:-1], label)

# calculate the kendalCoefficient, comparing the production data to every other column in the given np.array
# @_compareWithColIndex: index of the column, the Kendall coefficients shall be calculated for
def calcKendallCoefficients(data: np.array, _compareWithColIndex = 0, _label = "", _colnames = np.empty((0, 0))):
    productionData = data[:,_compareWithColIndex]
    kendallCoefficients = np.zeros(shape=data.shape[1] - 1)

    # iterate all available candidate factors
    for i in range(0, data.shape[1]):
        if i != _compareWithColIndex:   # do not compare column  to itself
            candidateData = data[:,i]
            corr, _ = kendalltau(productionData, candidateData)
            kendallCoefficients[i - 1] = corr

    addedColNames = np.array((
        "time",
        "dayOfWeek",
        "isWeekend",
        "weekNumber",
        "isHoliday (Feiertag)",
        "isSchoolHoliday",
        "diffuse Himmelstrahlung 10min (DS_10)",
        "globalstrahlung joule (GS_10)",
        "sonnenscheindauer (SD_10)",
        "Langwellige Strahlung (LS_10)",
        "Niederschlagsdauer 10min (RWS_DAU_10)",
        "Summe der Niederschlagsh. der vorangeg.10Min (RWS_10)",
        "Niederschlagsindikator  10min (RWS_IND_10)"
     ))
    colNames = np.append(_colnames, addedColNames)

    label = "Kendall Rank Korrelationen " + _label

    # plot coefficients as barchart
    plotTimeSeries(kendallCoefficients, colNames, label, "bar")

def executeFeasibilityAnalysisalfonsPechStr(_data: np.array, _color):
    global plotColor
    plotColor = _color

    # the first three days
    threeDaysTimestampData = np.array(_data[:4320,0], dtype=float)
    threeDaysMeasuremntData = np.array(_data[:4320,1], dtype=float)
    # all available data
    allDaysTimestampData = np.array(_data[:,0], dtype=float)
    allDaysMeasurementData = np.array(_data[:,1], dtype=float)

    # draw the plain production data
    plotTimeSeries(threeDaysMeasuremntData, threeDaysTimestampData, "production data")
    plotTimeSeries(allDaysMeasurementData, allDaysTimestampData, "production data")

    # check the autocorrelations
    calcAutocorrelation(threeDaysMeasuremntData, None)
    calcAutocorrelation(allDaysMeasurementData, None)

    # check the difference values
    calcDifferenceSeries(threeDaysMeasuremntData, threeDaysTimestampData)
    calcDifferenceSeries(allDaysMeasurementData, allDaysTimestampData)

    dataWithoutTimestamp = deleteColFromNpArray(_data, 0)
    calcKendallCoefficients(dataWithoutTimestamp) # from prepareData.py

def executeFeasibilityAnalysistanzendeSiedlung(
        _data: np.array,
        _plotPlainTimeSeries = False,
        _plotAutocorrelations=False,
        _plotDifferenceValues = False,
        _plotKendalCoefficients = False,
        _color = "blue"
    ):
    global plotColor
    plotColor = _color

    labels = {
        "networkObtainanceQuarter": "Netzbezug Quartier",
        "networkFeedInQuarter": "Netzeinspeisung Quartier",
        "PVConsumption": "Bezug PV-Analge",
        "PVFeedIn": "PV-Einspeisung Quartier"
    }

    # the first three days
    threeDayData = {}
    threeDayData["timestamp"] = np.array(_data[:96,0], dtype=float)
    threeDayData["networkFeedInQuarter"] = np.array(_data[:96,1], dtype=float)
    threeDayData["networkObtainanceQuarter"] = np.array(_data[:96,2], dtype=float)
    threeDayData["PVConsumption"] = np.array(_data[:96,3], dtype=float)
    threeDayData["PVFeedIn"] = np.array(_data[:96,4], dtype=float)

    # all available data
    allData = {}
    allData["timestamp"] = np.array(_data[:,0], dtype=float)
    allData["networkFeedInQuarter"] = np.array(_data[:,1], dtype=float)
    allData["networkObtainanceQuarter"] = np.array(_data[:,2], dtype=float)
    allData["PVConsumption"] = np.array(_data[:,3], dtype=float)
    allData["PVFeedIn"] = np.array(_data[:,4], dtype=float)

    # plot time series of the plain data
    if _plotPlainTimeSeries:
        plotTimeSeries(threeDayData["networkObtainanceQuarter"], threeDayData["timestamp"], labels["networkObtainanceQuarter"])
        plotTimeSeries(threeDayData["networkFeedInQuarter"], threeDayData["timestamp"], labels["networkFeedInQuarter"])
        plotTimeSeries(threeDayData["PVConsumption"], threeDayData["timestamp"], labels["PVConsumption"])
        plotTimeSeries(threeDayData["PVFeedIn"], threeDayData["timestamp"], labels["PVFeedIn"])

        plotTimeSeries(allData["networkObtainanceQuarter"], allData["timestamp"], labels["networkObtainanceQuarter"], "line")
        plotTimeSeries(allData["networkFeedInQuarter"], allData["timestamp"], labels["networkFeedInQuarter"], "line")
        plotTimeSeries(allData["PVConsumption"], allData["timestamp"], labels["PVConsumption"], "line")
        plotTimeSeries(allData["PVFeedIn"], allData["timestamp"], labels["PVFeedIn"], "line")

    # calc the autocorrelations
    if _plotAutocorrelations:
        calcAutocorrelation(allData["networkObtainanceQuarter"], None, labels["networkObtainanceQuarter"])
        calcAutocorrelation(allData["networkFeedInQuarter"], None, labels["networkFeedInQuarter"])
        calcAutocorrelation(allData["PVConsumption"], None, labels["PVConsumption"])
        calcAutocorrelation(allData["PVFeedIn"], None, labels["PVFeedIn"])

    # calc the difference values
    if _plotDifferenceValues:
        calcDifferenceSeries(allData["networkObtainanceQuarter"], allData["timestamp"], labels["networkObtainanceQuarter"])
        calcDifferenceSeries(allData["networkFeedInQuarter"], allData["timestamp"], labels["networkFeedInQuarter"])
        calcDifferenceSeries(allData["PVConsumption"], allData["timestamp"], labels["PVConsumption"])
        calcDifferenceSeries(allData["PVFeedIn"], allData["timestamp"], labels["PVFeedIn"])

    # plot the Kendall coefficients for every col of the data
    if _plotKendalCoefficients:
        dataWithoutTimestamp = deleteColFromNpArray(_data, 0)
        colnames = np.array(("networkObtainanceQuarter", "networkFeedInQuarter", "PVConsumption", "PVFeedIn"))

        calcKendallCoefficients(dataWithoutTimestamp, 1, labels["networkObtainanceQuarter"], colnames)
        calcKendallCoefficients(dataWithoutTimestamp, 2, labels["networkFeedInQuarter"], colnames)
        calcKendallCoefficients(dataWithoutTimestamp, 3, labels["PVConsumption"], colnames)
        calcKendallCoefficients(dataWithoutTimestamp, 4, labels["PVFeedIn"], colnames)
