import os
import numpy as np
from datetime import datetime
import requests
from datetimerange import DateTimeRange
from datetime import date
import csv
import urllib.request
import zipfile
from ftplib import FTP

path = '../data/PV/APS_PV/'

# Overview of weatherstations available here:  https://opendata.dwd.de/climate_environment/CDC/help/CS_Stundenwerte_Beschreibung_Stationen.txt
weatherStationId = "00853" # wheatherstationId Chemnitz
# weatherStationId = "02444" # wheatherstationID Jena Sternwarte

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

    list_of_files.sort()
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
        time = datetimeValue.strftime("%H:%M:%S")
        line[2] = time

        # get the day of week as index, where Monday = 0 and Sunday = 6
        dayOfWeek = datetimeValue.weekday()
        weekdayVector = [0, 0, 0, 0, 0, 0, 0]
        weekdayVector[dayOfWeek] = 1
        line[3] = str(weekdayVector)

        # get isWeekend
        if dayOfWeek == 5 or dayOfWeek == 6:
            line[4] = 1  # otherwise, 0, which is the standard value

        # get the week number
        weeknumber = datetimeValue.isocalendar()[1]
        line[5] = weeknumber

    return data

# adds the information to the given array, whether a day is a holiday (Feiertag) or not
def addIsHolidayInformation(data: np.array):
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
            line[6] = 1  # otherwise zero, which is the standard

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
def downloadDWDWeatherData(_year, _type):
    global weatherStationId
    current_year = date.today().year

    if _type == "precipitation":
        internalType = "nieder"
    else:
        internalType = "SOLAR"

    url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/{type}/".format(type = _type)
    if current_year != _year:
        url += "historical/"
        foldername = "10minutenwerte_{type}_{weatherstation_id}_{year}0101_{year}1231_hist.zip"
        foldername = foldername.format(weatherstation_id = weatherStationId, year = _year, type=internalType)
    else:
        url += "recent/"
        foldername = "10minutenwerte_{type}_{station}_akt.zip".format(station = weatherStationId, type=internalType)

    weatherdata = downloadAndUnzipContent(url + foldername)
    return weatherdata

# given the url to a zip file it downloads the folder, decompresses it and returns the content of the first file
def downloadAndUnzipContent(_url):
    print("download ", _url)
    filehandle, _ = urllib.request.urlretrieve(_url)
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
        line[8] = DS_10
    else:
        line[8] = 0

    GS_10 = solarWeatherdata[roundedDate][4] # Globalstrahlung 10min
    if GS_10 != -999:
        line[9] = GS_10
    else:
        line[9] = 0

    SD_10 = solarWeatherdata[roundedDate][5] # Sonnenscheindauer 10min
    if SD_10 != -999:
        line[10] = SD_10
    else:
        line[10] = 0

    LS_10 = solarWeatherdata[roundedDate][6] # Langwellige Strahlung 10min
    if LS_10 != -999:
        line[11] = LS_10
    else:
        line[11] = 0

# helper function for addWeatherdata; saves the given precipitationWeatherdata to the given row
def setprecipitationDataToRow(precipitationWeatherdata, roundedDate, line):
    RWS_DAU_10 = precipitationWeatherdata[roundedDate][3] # Niederschlagsdauer 10min
    if RWS_DAU_10 != -999:
        line[12] = RWS_DAU_10
    else:
        line[12] = 0

    RWS_10 = precipitationWeatherdata[roundedDate][4] # Summe der Niederschlagsh. der vorangeg.10Min
    if RWS_10 != -999:
        line[13] = RWS_10
    else:
        line[13] = 0

    RWS_IND_10 = precipitationWeatherdata[roundedDate][5] # Niederschlagsindikator  10min
    if RWS_IND_10 != -999:
        line[14] = RWS_IND_10
    else:
        line[14] = 0

# adds a column of zeroes to the input np.array
def addNewColToNpArray(array: np.array):
    dataLength = array.shape[0]
    newCol = np.zeros((dataLength, 1))
    newArray = np.append(array, newCol, axis=1)

    return newArray

def deleteColFromNpArray(array: np.array, index):
    array = np.delete(array, index, 1)
    return array

# call all necessary steps
data = loadData(path)
dataWithDateInformation = convertTimestampToDateInformation(data)
dataWithHolidayInformation = addIsHolidayInformation(dataWithDateInformation)
dataWithSchoolHolidayInformation = addIsSchoolHolidayInformation(dataWithHolidayInformation)
dataWithWeatherInformation = addWeatherdata(dataWithSchoolHolidayInformation)
finalData = deleteColFromNpArray(dataWithWeatherInformation, 0)

print("debug prepareData")