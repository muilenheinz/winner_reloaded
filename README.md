# winner_reloaded

Scripts to predict solar-energy-production and usage for houses in the winner & winner reloaded projects. For further information see Ausarbeitung.pdf  

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
4. 60-min FeedIn
5. 24-hours FeedIn
6. 7-days FeedIn
7. 60-min Usage
8. 24-hours Usage
9. 7-days Usage

## createFinalModels.py 

Calculates the models derived from the parameters determined in the script checkModuleParameters.py. Weights of the models, the plots for val_loss and loss functions and the plotted predictions are then stored in the folder results/finalModels

##  plotResults.py

Plots the results of the Parameter check as barcharts by comparing the RMSEs of the tests for the single parameters.