from prepareData import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import kendalltau
import matplotlib.dates as md
import pandas as pd

plotColor = "blue"

# plot the dataY over the given dataX. If no dataX is given the axis is mocked as integer values with step 1
# @_label: describes the series, will be printed on the chart
# @_type: charttype, options are scatter (single dots), line and bar
def plotTimeSeries(dataY, dataX=None, _label="timeseries", _type="scatter", _indicesOfValesToPlot=None, _yAxisLabel=None):
    global plotColor

    # cut the axis to show only the requested area of the data
    if _indicesOfValesToPlot is not None:
        dataY = dataY[_indicesOfValesToPlot[0]:_indicesOfValesToPlot[1]]
        if dataX is not None:
            dataX = dataX[_indicesOfValesToPlot[0]:_indicesOfValesToPlot[1]]

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
            # convert timestamp in microsenconds to ts in seconds
            datetimeXAxis[key] = datetime.fromtimestamp(dataX[key] / 1000)
        dataX = datetimeXAxis

    # if no xAxis is given mock the axis
    if dataX is None:
        dataX = np.array(list(np.arange(1, dataY.size+1)))

    if _type == "scatter":
        plt.scatter(dataX, dataY, marker=".", label=_label, color=plotColor)
    elif _type == "bar":
        # sort the bars by absolute size
        dataY = dataY.astype("float64")
        mergedData = np.vstack((dataX, dataY)).T
        mergedData = np.array(sorted(mergedData, key=lambda row: abs(float(row[1])), reverse=True))

        dataY = np.array(mergedData[:,1].reshape(1, mergedData.shape[0]), dtype=("f8"))[0]
        dataX = np.array(mergedData[:,0].reshape(1, mergedData.shape[0]))[0]

        chart = plt.barh(dataX, dataY, label=_label, color=plotColor)
        plt.bar_label(chart, fmt="%.2f", label_type="center")
    elif _type == "line":
        plt.plot(dataX, dataY, label=_label, color=plotColor)

    # format timestamp values on the xAxis
    if isTimestampXAxis:
        dtFmt = md.DateFormatter('%d/%m/%Y \n %H:%M')
        plt.gca().xaxis.set_major_formatter(dtFmt)
        plt.xticks(rotation=45, ha="right")

    if _yAxisLabel is not None:
        plt.ylabel(_yAxisLabel)
    plt.tight_layout()
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
def calcAutocorrelation(dataY: np.array, dataX: np.array, _chartLabel=None, _indicesOfValesToPlot=None):
    result = np.correlate(dataY, dataY, mode='full')
    result = result[result.size//2:]
    normalizedResult = result / float(result.max())
    if _chartLabel is None:
        _chartLabel = "normalisierte Autokorrelation"
    else:
        _chartLabel = "normalisierte Autokorrelation " + _chartLabel

    # only show the first n values
    if _indicesOfValesToPlot is not None:
        normalizedResult = normalizedResult[_indicesOfValesToPlot[0]:_indicesOfValesToPlot[1]]
        if dataX is not None:
            dataX = dataX[_indicesOfValesToPlot[0]:_indicesOfValesToPlot[1]]

    plotTimeSeries(normalizedResult, dataX, _chartLabel, "line")

# for the 1D-np.array dataY calcs the deviation of a value from its predecessor
def calcDifferenceSeries(dataY: np.array, dataX: np.array, _label = "", _indicesOfValesToPlot=None):
    if _indicesOfValesToPlot is not None:
        dataY = dataY[_indicesOfValesToPlot[0]:_indicesOfValesToPlot[1]]
        dataX = dataX[_indicesOfValesToPlot[0]:_indicesOfValesToPlot[1]]

    result = np.diff(dataY)
    label = "Differenzdaten " + _label
    plotTimeSeries(result, dataX[:-1], label, "line")

# calculate the kendalCoefficient, comparing the production data to every other column in the given np.array
# @_compareWithColIndex: index of the column, the Kendall coefficients shall be calculated for
# @_preColnames: names of columns to be added before the calculated cols like time, weekday etc.
# @_postColnames: names of columns to be added after the calculated cols like time, weekday etc.
def calcKendallCoefficients(data: np.array, _compareWithColIndex = 0, _label = "", _preColnames = np.empty((0, 0)), _postColnames = np.empty((0, 0))):
    productionData = data[:,_compareWithColIndex]
    kendallCoefficients = np.zeros(shape=data.shape[1] - 1)

    filteredData = np.delete(data, _compareWithColIndex, 1)

    kendallNanIndices = []

    # iterate all available candidate factors
    for i in range(0, filteredData.shape[1]):
        candidateData = filteredData[:, i]
        corr, _ = kendalltau(productionData, candidateData)
        if pd.isna(corr):
            corr = 0
            kendallNanIndices.append(i)
        kendallCoefficients[i] = corr

    addedColNames = np.array((
        "Zeit (cos)",
        "Tag der Woche",
        "Wochenende",
        "Wochennummer",
        "Feiertag",
        "Schulferien",
        "DS_10", # diffuse Himmelstrahlung 10min
        "GS_10", #globalstrahlung joule
        "SD_10", #sonnenscheindauer
        "LS_10",# Langwellige Strahlung
        "RWS_DAU_10", #Niederschlagsdauer 10min
        "RWS_10", #Summe der Niederschlagsh. der vorangeg.10Min
        "RWS_IND_10" #Niederschlagsindikator  10min
     ))
    colNames = np.append(_preColnames, addedColNames)
    colNames = np.append(colNames, _postColnames)

    # remove col to compare from names
    colNames = np.delete(colNames, _compareWithColIndex)

    for index in kendallNanIndices:
        print("got Kendall = nan for ", colNames[index], "(index ", index, ")")

    label = "Kendall Ranks \n" + _label

    # plot coefficients as barchart
    plotTimeSeries(kendallCoefficients, colNames, label, "bar")

    return kendallCoefficients

def executeFeasibilityAnalysisalfonsPechStr(_data: np.array, _color):
    global plotColor
    plotColor = _color

    # all available data
    allDaysTimestampData = np.array(_data[:,0], dtype=float)
    allDaysMeasurementData = np.array(_data[:,1], dtype=float)

    # draw the plain production data
    plotTimeSeries(allDaysMeasurementData, allDaysTimestampData, "Produktionsdaten", "line")

    # check the autocorrelations
    calcAutocorrelation(allDaysMeasurementData, None)

    # check the difference values
    calcDifferenceSeries(allDaysMeasurementData, allDaysTimestampData)

    dataWithoutTimestamp = deleteColFromNpArray(_data, 0)
    calcKendallCoefficients(dataWithoutTimestamp, 0, "Erzeugung", np.array(("Erzeugung"))) # from prepareData.py

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
        "networkObtainanceQuarter": "Netzbezug",
        "networkFeedInQuarter": "Netzeinspeisung",
        "PVConsumption": "Bezug PV-Analge",
        "PVFeedIn": "PV-Einspeisung",
        "overallConsumption": "Gesamtverbrauch"
    }

    # the first three days
    threeDayData = {}
    threeDayData["timestamp"] = np.array(_data[:96,0], dtype=float)
    threeDayData["networkFeedInQuarter"] = np.array(_data[:96,1], dtype=float)
    threeDayData["networkObtainanceQuarter"] = np.array(_data[:96,2], dtype=float)
    threeDayData["PVConsumption"] = np.array(_data[:96,4], dtype=float)
    threeDayData["PVFeedIn"] = np.array(_data[:96,3], dtype=float)
    threeDayData["overallConsumption"] = np.array(_data[:96,-1], dtype=float)

    # all available data
    allData = {}
    allData["timestamp"] = np.array(_data[:,0], dtype=float)
    allData["networkFeedInQuarter"] = np.array(_data[:,1], dtype=float)
    allData["networkObtainanceQuarter"] = np.array(_data[:,2], dtype=float)
    allData["PVConsumption"] = np.array(_data[:,3], dtype=float)
    allData["PVFeedIn"] = np.array(_data[:,4], dtype=float)
    allData["overallConsumption"] = np.array(_data[:,-2], dtype=float)

    indicesOfValesToPlot = np.array((17376, 17760)) #
    # indicesOfValesToPlot = None

    # plot time series of the plain data
    if _plotPlainTimeSeries:
        yAxisUnit = "kWh"
        plotTimeSeries(allData["PVFeedIn"], allData["timestamp"], labels["PVFeedIn"], "line", indicesOfValesToPlot, yAxisUnit)
        plotTimeSeries(allData["overallConsumption"], allData["timestamp"], labels["overallConsumption"], "line", indicesOfValesToPlot, yAxisUnit)

    # calc the autocorrelations
    if _plotAutocorrelations:
        calcAutocorrelation(allData["PVFeedIn"], None, labels["PVFeedIn"], indicesOfValesToPlot)
        calcAutocorrelation(allData["overallConsumption"], None, labels["overallConsumption"], indicesOfValesToPlot)

    # calc the difference values
    if _plotDifferenceValues:
        calcDifferenceSeries(allData["PVFeedIn"], allData["timestamp"], labels["PVFeedIn"], indicesOfValesToPlot)
        calcDifferenceSeries(allData["overallConsumption"], allData["timestamp"], labels["overallConsumption"], indicesOfValesToPlot)

    # plot the Kendall coefficients for every col of the data
    if _plotKendalCoefficients:
        dataWithoutTimestamp = deleteColFromNpArray(_data, 0)
        colnames = np.array((labels["networkObtainanceQuarter"], labels["networkFeedInQuarter"], labels["PVFeedIn"], labels["PVConsumption"]))
        postColNames = np.array(("Gesamtverbrauch", "Mietverträge"))

        calcKendallCoefficients(dataWithoutTimestamp, 2, labels["PVFeedIn"], colnames, postColNames)
        calcKendallCoefficients(dataWithoutTimestamp, 17, labels["overallConsumption"], colnames, postColNames)

apsData = prepareAlfonsPechStrasseData()
executeFeasibilityAnalysisalfonsPechStr(apsData.values, "red")

tasData = prepareTanzendeSiedlungData()
executeFeasibilityAnalysistanzendeSiedlung(tasData.values, True, True, True, True, "blue")
