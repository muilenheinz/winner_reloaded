# winner_reloaded

## prepareData.py
load data from csv files and add information on candidate values like the weather or holidays to them

order of the appended data in the np array:
- (values from the csv files)
- time (as seconds since midnight of that day)
- dayOfWeek (as vector, e.g. monday = [1,0,0,0,0,0,0])
- isWeekend
- weekNumber
- isHoliday (Feiertag)
- isSchoolHoliday
- diffuse Himmelstrahlung 10min (DS_10)
- globalstrahlung joule (GS_10)
- sonnenscheindauer (SD_10)
- Langwellige Strahlung (LS_10)
- Niederschlagsdauer 10min (RWS_DAU_10)
- Summe der Niederschlagsh. der vorangeg.10Min (RWS_10)
- Niederschlagsindikator  10min (RWS_IND_10)

## checkFeasibility.py

do some calculations to prove the feasibility of the given data for machine learning

- plain plotting of the data
- (normalized) autocorrelation
- differential data (measurement value - measurement value of predecessor in data)
- Kendal rank correlations
