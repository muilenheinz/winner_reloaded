import os
import numpy as np
from datetime import datetime
import requests

path = '../../data/PV/APS_PV/'

# order of the np array:
# (timestamp), measurement, time, dayOfWeek, isWeekend, weekNumber, isHoliday (Feiertag)

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

# adds the information to the given array, whether a day is a holiday (Feiertag) or not
def getIsHoliday(data: np.array):
    # add a new col to hold the new values
    data = addNewColToNpArray(data)

    year = None
    holidays = []

    for line in data:
        # when the year changes in contrast to the last line download the holiday-values for this "new" year
        timestamp = float(line[0])
        datetimeValue = datetime.fromtimestamp(timestamp / 1000)
        lineYear = datetimeValue.year
        if year != lineYear:
            year = lineYear
            holidays = getHolidays(year, "TH")

        # check if the current date is listed as holiday
        lineDate = datetimeValue.strftime("%Y-%m-%d")
        if lineDate in holidays:
            line[6] = 1  # otherwise zero, which is the standard

    return data

# calling an API returns a list of dates, which are holidays (Feiertage) in the given year and the given state
# @_state: String, abbreviation for the state, so e.g. Thüringen would be "TH"
# @return list of dates in format YYY-MM-DD, which are holidays
def getHolidays(_year, _state):
    holidays = []
    formatUrl = "https://feiertage-api.de/api/?jahr={year}".format(year=_year)
    r = requests.get(formatUrl)
    APIResult = r.json()[_state]

    # for better performance and since the names of the holidays are not required for this task
    # get a list which only contains the dates of the vacations
    for holiday in APIResult:
        holidays.append(APIResult[holiday]["datum"])

    return holidays

# adds a column of zeroes to the input np.array
def addNewColToNpArray(array: np.array):
    dataLength = array.shape[0]
    newCol = np.zeros((dataLength, 1))
    newArray = np.append(array, newCol, axis=1)

    return newArray

# call all necessary steps
data = loadData(path)
dataWithDateInformation = convertTimestampToDateInformation(data)
dataWithHolidayInformation = getIsHoliday(dataWithDateInformation)