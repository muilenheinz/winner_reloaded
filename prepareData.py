import os
import numpy as np
from datetime import datetime
import requests
from datetimerange import DateTimeRange
from datetime import date, datetime, timedelta, time
import csv
import urllib.request
import zipfile
import time
import re
import datetime as dt
import pandas as pd
from scipy.stats import kendalltau

weatherStationId = None

# returns a np.array of all data given in the csv files in the directory indicated by path parameter
# @path directory in which the csv files are placed
# @_csvSeparator: sign with which the csv values are separated, typically , or ;
# @headerRows: number of lines which do only contain headlines
def loadData(_path, _csvSeparator, _headerRows):
    print("load data from csv files")
    # get List of available Files
    list_of_files = []
    for root, dirs, files in os.walk(_path):
        for file in files:
            if file.find(".csv") != -1:
                list_of_files.append(os.path.join(root,file))

    list_of_files.sort()

    # iterate these files and load all available data into np.array
    result = []
    for name in list_of_files:
        f = open(name, encoding="utf-8")
        for i, line in enumerate(f):
            if i > _headerRows:     # exclude the header row(s)
                if line != "" and line != "\n" and line != '""':
                    data = line.replace('"', '').strip().split(_csvSeparator)
                    data = convertDecimalCommaToDecimalPointInDict(data)
                    result.append(data)

    return np.array(result)

# checks all fields in the given dict whether they contain numbers with a decimal comma and
# converts them to decimal point format when necessary
def convertDecimalCommaToDecimalPointInDict(dict):
    for key, value in enumerate(dict):
        # check if it is numeric in the form x,xxx
        if re.search("[0-9]*,[0-9]", value):
            dict[key] = value.replace(",", ".")
    return dict

# from the timestamp information of each line in the np.array extract
# time, weekday, isWeekend, weekNumber, isHoliday (Feiertag) and add it to a new column
def convertTimestampToDateInformation(data: np.array):
    print("get date information")

    data = addNewColToNpArray(data)  # for the time
    data = addNewColToNpArray(data)  # for the weekday
    data = addNewColToNpArray(data)  # for isWeekend
    data = addNewColToNpArray(data)  # for weeknumber

    for line in data:
        timestamp = float(line[0])

        datetimeValue = datetime.fromtimestamp(timestamp / 1000)

        # get the time as midnight-timestamp (seconds elapsed since midnight)
        hourVal = int(datetimeValue.strftime('%H'))
        minuteVal = int(datetimeValue.strftime('%M'))
        secondVal = int(datetimeValue.strftime('%S'))
        timeValue = dt.time(hourVal, minuteVal, secondVal)
        td = datetime.combine(datetime.min, timeValue) - datetime.min
        midnightTimestamp = td // timedelta(seconds=1)

        seconds_in_day = 24*60*60
        cosine_time = np.cos(2*np.pi*midnightTimestamp/seconds_in_day)

        # store the time value in the dataframe
        line[-4] = cosine_time

        # get the day of week as index, where Monday = 0 and Sunday = 6
        dayOfWeek = datetimeValue.weekday()
        weekdayVector = [0, 0, 0, 0, 0, 0, 0]
        weekdayVector[dayOfWeek] = 1
        line[-3] = str(weekdayVector)

        line[-3] = dayOfWeek

        # get isWeekend
        if dayOfWeek == 5 or dayOfWeek == 6:
            line[-2] = 1  # otherwise, 0, which is the standard value

        # get the week number
        weeknumber = datetimeValue.isocalendar()[1]
        line[-1] = weeknumber

    return data

# adds the information to the given array, whether a day is a holiday (Feiertag) or not
def addIsHolidayInformation(data: np.array):
    print("add Holiday Information")
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
            holidays = getHolidayList(year, "TH")

        # check if the current date is listed as holiday
        lineDate = datetimeValue.strftime("%Y-%m-%d")
        if lineDate in holidays:
            line[-1] = 1  # otherwise zero, which is the standard

    return data

# calling an API returns a list of dates, which are holidays (Feiertage) in the given year and the given state
# @_state: String, abbreviation for the state, so e.g. Thüringen would be "TH"
# @return list of dates in format YYY-MM-DD, which are holidays
def getHolidayList(_year, _state):
    holidays = []
    formatUrl = "https://feiertage-api.de/api/?jahr={year}".format(year=_year)
    r = requests.get(formatUrl)
    APIResult = r.json()[_state]

    # for better performance and since the names of the holidays are not required for this task
    # get a list which only contains the dates of the vacations
    for holiday in APIResult:
        holidays.append(APIResult[holiday]["datum"])

    return holidays

# calling an api returns a list of DateTimeRanges, which are vacation in the given state and year
def getSchoolHolidayList(_year, _state):
    url = "https://ferien-api.de/api/v1/holidays/{state}/{year}".format(state=_state, year=_year)

    r = requests.get(url)
    APIResult = r.json()
    schoolHolidays = []

    for schoolHoliday in APIResult:
        # schoolHolidaySpan = [schoolHoliday["start"], schoolHoliday["end"]]
        schoolHolidaySpan = DateTimeRange(schoolHoliday["start"], schoolHoliday["end"])
        schoolHolidays.append(schoolHolidaySpan)

    return schoolHolidays

# for each entry of the given data array it checks whether there are schoolHolidays and sets the data-value accordingly
def addIsSchoolHolidayInformation(data: np.array):
    print("add school holiday information")
    # add a new col to hold the new values
    data = addNewColToNpArray(data)
    year = None
    isHoliday = 0
    day = None
    schoolHolidays = []

    for line in data:
        # when the year changes in contrast to the last line download the holiday-values for this "new" year
        timestamp = float(line[0])
        datetimeValue = datetime.fromtimestamp(timestamp / 1000)
        lineYear = datetimeValue.year
        if year != lineYear:
            year = lineYear
            schoolHolidays = getSchoolHolidayList(year, "TH")

        # when the day changed against the previous one check if the new day is in some Holiday, otherwise just copy
        # the value of the last loop-passage
        lineDay = datetimeValue.day
        if lineDay != day:
            day = lineDay
            isHoliday = 0

            # check if the current date lies in a schoolHoliday
            lineDate = datetimeValue.strftime("%Y-%m-%dT%H:%M:%S+0000")
            for schoolHolidayRange in schoolHolidays:
                if lineDate in schoolHolidayRange:
                    line[7] = 1
                    isHoliday = 1
                    break
        else:
            line[7] = isHoliday
    return data

# downloads weatherdata from the dwd
# @_type: "solar" or "precipitation"
# @_mode: (for internal use) in "historical" mode it will search for data of past years in the /historical folder
#       however, the case might occur that at the start of the new year the data of the old year are not yet uploaded
#       to the /historical folder yet. In this case the mode "recent" will load data from the /recent folder
def downloadDWDWeatherData(_year, _type, _mode="historical"):
    global weatherStationId
    current_year = date.today().year

    if _type == "precipitation":
        internalType = "nieder"
    else:
        internalType = "SOLAR"

    url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/{type}/".format(type = _type)
    if current_year != _year and _mode == "historical":
        url += "historical/"
        next_year = _year + 1
        foldername = "10minutenwerte_{type}_{weatherstation_id}_{year}0101_{next_year}1231_hist.zip"
        foldername = foldername.format(weatherstation_id = weatherStationId, year = _year, type=internalType, next_year = next_year)
    else:
        url += "recent/"
        foldername = "10minutenwerte_{type}_{station}_akt.zip".format(station = weatherStationId, type=internalType)

    weatherdata = downloadAndUnzipContent(url + foldername)
    if weatherdata == False:
        print("not found, retry with recent data")
        return downloadDWDWeatherData(_year, _type, "recent")

    return weatherdata

# given the url to a zip file it downloads the folder, decompresses it and returns the content of the first file
def downloadAndUnzipContent(_url):
    print("download ", _url)
    try:
        filehandle, _ = urllib.request.urlretrieve(_url)
    except urllib.error.HTTPError as exception:
        return False    # most likely a 404-error

    zip_file_object = zipfile.ZipFile(filehandle, 'r')
    first_file = zip_file_object.namelist()[0]
    file = zip_file_object.open(first_file)

    content = file.read()
    content = content.decode("utf-8") # convert byte-object to text

    result = {}
    reader = csv.reader(content.split('eor\r\n'), delimiter=';')
    for row in reader:
        if len(row) > 0 and row[0] != "MESS_DATUM" and row[0] != '   0.000':
            result[row[1]] = row

    return result

# adds weatherdata given from the dwd to the given array, currently only the LongwaveRadiation
def addWeatherdata(data: np.array):
    print("add weatherdata")
    year = None
    data = addNewColToNpArray(data) # DS_10; diffuse Himmelstrahlung 10min
    data = addNewColToNpArray(data) # GS_10; Globalstrahlung 10min
    data = addNewColToNpArray(data) # SD_10; Sonnenscheindauer 10min
    data = addNewColToNpArray(data) # LS_10; Langwellige Strahlung 10min

    data = addNewColToNpArray(data) # RWS_DAU_10; Niederschlagsdauer 10min
    data = addNewColToNpArray(data) # RWS_10; Summe der Niederschlagsh. der vorangeg.10Min
    data = addNewColToNpArray(data) # RWS_IND_10; Niederschlagsindikator  10min

    for line in data:
        timestamp = float(line[0])
        datetimeValue = datetime.fromtimestamp(timestamp / 1000)
        lineYear = datetimeValue.year

        # when the year changes against the last loop, load new weatherdata
        if lineYear != year:
            year = lineYear
            solarWeatherdata = downloadDWDWeatherData(year, "solar")
            precipitationWeatherdata = downloadDWDWeatherData(year, "precipitation")

        # get the date for the weather by rounding the current minute to the next 10-minute, since weatherdata are
        # given in 10-minute steps
        currentMinute = float(datetimeValue.strftime("%M"))
        remainder = currentMinute % 10
        roundedMinute = currentMinute - remainder
        roundedDate = str(datetimeValue.strftime("%Y%m%d%H")) + ("0" + str(int(roundedMinute)))[-2:]

        # get the weatherdata for the current situation

        setSolarDataToRow(solarWeatherdata, roundedDate, line)
        setprecipitationDataToRow(precipitationWeatherdata, roundedDate, line)

    return data

# helper function for addWeatherdata; saves the given solarWeatherdata to the given row
def setSolarDataToRow(solarWeatherdata, roundedDate, line):
    DS_10 = solarWeatherdata[roundedDate][3] # diffuse Himmelstrahlung 10min
    if DS_10 != -999:
        line[-7] = DS_10
    else:
        line[-7] = 0

    GS_10 = solarWeatherdata[roundedDate][4] # Globalstrahlung 10min
    if GS_10 != -999:
        line[-6] = GS_10
    else:
        line[-6] = 0

    SD_10 = solarWeatherdata[roundedDate][5] # Sonnenscheindauer 10min
    if SD_10 != -999:
        line[-5] = SD_10
    else:
        line[-5] = 0

    LS_10 = solarWeatherdata[roundedDate][6] # Langwellige Strahlung 10min
    if LS_10 != "-999":
        line[-4] = LS_10
    else:
        line[-4] = 0

# helper function for addWeatherdata; saves the given precipitationWeatherdata to the given row
def setprecipitationDataToRow(precipitationWeatherdata, roundedDate, line):
    if roundedDate in precipitationWeatherdata:
        RWS_DAU_10 = precipitationWeatherdata[roundedDate][3] # Niederschlagsdauer 10min
        if RWS_DAU_10 != -999:
            line[-3] = RWS_DAU_10
        else:
            line[-3] = 0

        RWS_10 = precipitationWeatherdata[roundedDate][4] # Summe der Niederschlagsh. der vorangeg.10Min
        if RWS_10 != -999:
            line[-2] = RWS_10
        else:
            line[-2] = 0

        RWS_IND_10 = precipitationWeatherdata[roundedDate][5] # Niederschlagsindikator  10min
        if RWS_IND_10 != -999:
            line[-1] = RWS_IND_10
        else:
            line[-1] = 0
    else:
        line[-3] = 0
        line[-2] = 0
        line[-1] = 0

# adds a column of zeroes to the input np.array
def addNewColToNpArray(array: np.array):
    dataLength = array.shape[0]
    newCol = np.zeros((dataLength, 1))
    newArray = np.append(array, newCol, axis=1)

    return newArray

def deleteColFromNpArray(array: np.array, index):
    array = np.delete(array, index, 1)
    return array

def convertDateColToTimestampCol(data: np.array, _index):
    def convertDateToTimestamp(input):
        timestamp = time.mktime(time.strptime(input[0], '%d.%m.%Y %H:%M:%S'))
        input[0] = int(timestamp * 1000)
        return input

    output = np.apply_along_axis(convertDateToTimestamp, 1, data)
    return output

def dataCleaning(data: np.array):
    def replaceMinusWithZero(input):
        if input == " - ":
            return 0
        else:
            return input
    replaceMinusWithZeroVectorized = np.vectorize(replaceMinusWithZero)
    result = replaceMinusWithZeroVectorized(data)
    return result

# gets all data for the given path and enriches it with various information
# @_path: path of the directory of the desired files as string relative to this folder
# @_weatherStationId: dwd-id of the station the weather shall be loaded from
# @_withTimestamp: boolean if the timestamp col shall be deleted or not
def prepareData(_path, _weatherStationId, _withTimestamp, _csvSeparator, _headerRows):
    global weatherStationId
    weatherStationId = _weatherStationId

    data = loadData(_path, _csvSeparator, _headerRows)

    # if date is given in datetime format (e.g. 12.12.21 23:00) convert it to timestamp
    if not data[0][0].isnumeric():
        data = convertDateColToTimestampCol(data, 0)

    dataWithDateInformation = convertTimestampToDateInformation(data)
    dataWithHolidayInformation = addIsHolidayInformation(dataWithDateInformation)
    dataWithSchoolHolidayInformation = addIsSchoolHolidayInformation(dataWithHolidayInformation)
    dataWithWeatherInformation = addWeatherdata(dataWithSchoolHolidayInformation)
    dataCleaned = dataCleaning(dataWithWeatherInformation)

    if _withTimestamp:
        return dataCleaned
    else:
        return deleteColFromNpArray(dataCleaned, 0)


def calcOverallEnergyConsuption(data: np.array):
    data = addNewColToNpArray(data)

    def calcConsumption (line):
        LG_wirk_plus_trafo = line[13]
        LG_wirk_minus_trafo = line[16]
        LG_wirk_plus_pv = line[19]

        overall = float(LG_wirk_plus_trafo) + float(LG_wirk_plus_pv) - float(LG_wirk_minus_trafo)
        line[-1] = overall

    for line in data:
        calcConsumption(line)

    return data

def dataCleaningAlfonsPechStrasse(_data: np.array):
    dropindices = []
    for index, row in enumerate(_data):
        # filter unrealistic production data
        if float(row[1]) > 1.0:
            dropindices.append(index)

    dropindices.reverse()
    for index in dropindices:
        _data = np.delete(_data, index, axis=0)

    return _data

# in 2021, tanzende siedlung was not fully occupied by January, but instead the tenants moved in over the year
# so add the information, how many people lived there each month
def addOccupanyNumbersTanzendeSiedlung(data: np.array):
    data = addNewColToNpArray(data)

    newContracts = {
        "Jan": 9,
        "Mar": 1,
        "Apr": 1,
        "May": 1,
        "Sep": 28
    }

    oldMonth = None
    numberOfContracts = 0

    for row in data:
        timestamp = float(row[0]) / 1000
        datetimeValue = datetime.fromtimestamp(timestamp)
        month = datetimeValue.strftime("%b")
        year = datetimeValue.strftime("%Y")

        # check if the month border was exceeded and if so if new contracts were signed that month
        if int(year) == 2021 and oldMonth != month:
            oldMonth = month
            if month in newContracts:
                numberOfContracts += newContracts[month]
        row[-1] = numberOfContracts

    return data

def convertArrayToDataFrame(data: np.array, _colNames):
    dataFrame = pd.DataFrame(data)
    dataFrame.astype(float)
    dataFrame.columns = _colNames

    return dataFrame

def filterDataBasedOnKendallRanks(data: pd.DataFrame, KendallRanksForCol = "Messwert", limit = 0.3):

    # calc the kendall ranks for the col given as parameter
    targetData = data[KendallRanksForCol]

    del data["Zeitstempel"]

    for column in data:
        corr, _ = kendalltau(targetData, data[column])
        if pd.isna(corr):
            corr = 0
        # filter all columns with kendall smaller than limit
        if abs(corr) < limit:
            del data[column]

    return data







