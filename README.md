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

## checkModuleParameters.py

checks how good a config is suited by running it three times. Average RMSE and runtime  of these three runs are then 
stored  in a file called regressionModels.csv accompanied by all other parameters. Plots of losses and predictions 
done by this model are stored in the same folder. Call this script with params

checkModuleParameters.py 001 110001

The first three digits indicate whether to calc the models for aps:
1. 60-Min  
2. 24hours 
3. 7days model 

The last 6 digits indicate whether to try the params for tanzende Siedlung: 
4. 60min FeedIn
5. 24hours FeedIn
6. 7days FeedIn
7. 60min Usage
8. 24hours Usage
9. 7days Usage