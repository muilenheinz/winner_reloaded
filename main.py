from prepareData import *
from checkFeasibility import *
from doPredictions import *
import pandas as pd

# Overview of weatherstations available here:
# https://opendata.dwd.de/climate_environment/CDC/help/CS_Stundenwerte_Beschreibung_Stationen.txt
weatherStationIdChemnitz = "00853"  # wheatherstationId Chemnitz
weatherStationIdJena = "02444"      # wheatherstationID Jena Sternwarte


# check and prepare Data for Alfons-Pech-Straße, Chemnitz
def calcAlfonsPechStrasse(_feasibilityAnalysis = True, _predictions = True):
    print("######################## calculations for Alfons-Pech-Strasse #######################")

    alfonsPechStrData = prepareData('../data/PV/APS_PV/', weatherStationIdChemnitz, True, ",", 1)
    alfonsPechStrData = dataCleaningAlfonsPechStrasse(alfonsPechStrData)

    if _feasibilityAnalysis:
        executeFeasibilityAnalysisalfonsPechStr(alfonsPechStrData, "red")

    if _predictions:
        colNames = np.array((
            "Zeitstempel",
            "Messwert",
            "Zeit (cos)",
            "Tag der Woche",
            "ist Wochenende",
            "Wochennummer",
            "ist Feiertag",
            "sind Schulferien",
            "DS_10",                # diffuse Himmelstrahlung 10min
            "GS_10",                # globalstrahlung joule
            "SD_10",                # sonnenscheindauer
            "LS_10",                # Langwellige Strahlung
            "RWS_DAU_10",           # Niederschlagsdauer 10min
            "RWS_10",               # Summe der Niederschlagsh. der vorangeg.10Min
            "RWS_IND_10",           # Niederschlagsindikator  10min
        ))

        alfonsPechStrData = convertArrayToDataFrame(alfonsPechStrData, colNames)
        determineOptimalParametersForAlfonsPechStrasse(alfonsPechStrData)

# chak and prepare Data for tanzende Siedlung, Chemnitz
def calcTanzendeSiedlung(_feasibilityAnalysis = True, _predictions = True):
    print("######################## calculations for tanzende Siedlung #######################")
    tanzendeSiedlungData = prepareData('../data/TAS/inetz/', weatherStationIdChemnitz, True, ";", 2)

    tanzendeSiedlungData = calcOverallEnergyConsuption(tanzendeSiedlungData)
    tanzendeSiedlungData = addOccupanyNumbersTanzendeSiedlung(tanzendeSiedlungData)

    # remove cols for blinddata, Unit and state from the data
    removeCols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24]
    removeCols.reverse()
    for i in removeCols:
        tanzendeSiedlungData = np.delete(tanzendeSiedlungData, i, 1)

    if _feasibilityAnalysis:
        executeFeasibilityAnalysistanzendeSiedlung(tanzendeSiedlungData, True, True, True, True, "cornflowerblue")

    if _predictions:
        colNames = np.array((
            "Zeitstempel",
            "Netzbezug",
            "Netzeinspeisung",
            "Bezug PV-Analge",
            "PV-Einspeisung",
            "Zeit (cos)",
            "Tag der Woche",
            "ist Wochenende",
            "Wochennummer",
            "ist Feiertag",
            "sind Schulferien",
            "DS_10", # diffuse Himmelstrahlung 10min
            "GS_10", #globalstrahlung joule
            "SD_10", #sonnenscheindauer
            "LS_10",# Langwellige Strahlung
            "RWS_DAU_10", #Niederschlagsdauer 10min
            "RWS_10", #Summe der Niederschlagsh. der vorangeg.10Min
            "RWS_IND_10", #Niederschlagsindikator  10min
            "Gesamtverbrauch",
            "Mieteranzahl"
        ))

        tanzendeSiedlungData = convertArrayToDataFrame(tanzendeSiedlungData, colNames)
        determineOptimalParametersForTanzendeSiedlung(tanzendeSiedlungData)
        print("debug")

calcAlfonsPechStrasse(False, True)
calcTanzendeSiedlung(False, True)
