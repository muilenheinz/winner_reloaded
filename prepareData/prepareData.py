import os
import numpy as np
from datetime import datetime

path = '../../data/PV/APS_PV/'

# order of the np array:
# (timestamp), measurement, time, dayOfWeek, isWeekend, weekNumber

# returns a np.array of all data given in the csv files in the directory indicated by path parameter
# @path directory in which the csv files are placed
def loadData(path):
    list_of_files = getListOfAvailableFiles(path)
    data = loadCSVDataFromFiles(list_of_files)
    return data

# load list of files available in the given directory into the global var list_of_files
# @path directory in which csv files shall be searched for
def getListOfAvailableFiles(_path):
    list_of_files = []
    for root, dirs, files in os.walk(_path):
        for file in files:
            if file.find(".csv") != -1:
                list_of_files.append(os.path.join(root,file))

    return list_of_files

# @list_of_files: list of all files which shall be loaded into the resulting array
# @return concatenated np array of all data in the files indicated by list_of_files
def loadCSVDataFromFiles(list_of_files):
    result = []
    for name in list_of_files:
        f = open(name)
        for i, line in enumerate(f):
            if i > 1: # exclude the header row
                data = line.strip().split(",")
                result.append([data[0], data[1]])

    return np.array(result)

# from the timestamp information of each line in the np.array extract
# time, weekday, isWeekend, weekNumber, isHoliday (Feiertag) and add it to a new column
def convertTimestampToDateInformation(data: np.array):

    data = addNewColToNpArray(data)  # for the time
    data = addNewColToNpArray(data)  # for the weekday
    data = addNewColToNpArray(data)  # for isWeekend
    data = addNewColToNpArray(data)  # for weeknumber

    for line in data:
        timestamp = float(line[0])

        # get the time
        datetimeValue = datetime.fromtimestamp(timestamp / 1000)
        time = datetimeValue.strftime("%H:%M")
        line[2] = time

        # get the day of week as index, where Monday = 0 and Sunday = 6
        dayOfWeek = datetimeValue.weekday()
        line[3] = dayOfWeek

        # get isWeekend
        if dayOfWeek == 5 or dayOfWeek == 6:
            line[4] = 1  # otherwise, 0, which is the standard value

        # get the week number
        weeknumber = datetimeValue.isocalendar()[1]
        line[5] = weeknumber

    return data

# adds a column of zeroes to the input np.array
def addNewColToNpArray(array: np.array):
    dataLength = array.shape[0]
    newCol = np.zeros((dataLength, 1))
    newArray = np.append(array, newCol, axis=1)

    return newArray


# call all necessary steps
data = loadData(path)
dataWithDateInformation = convertTimestampToDateInformation(data)