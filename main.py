from prepareData import *
from checkFeasibility import *

# Overview of weatherstations available here:
# https://opendata.dwd.de/climate_environment/CDC/help/CS_Stundenwerte_Beschreibung_Stationen.txt
weatherStationIdChemnitz = "00853"  # wheatherstationId Chemnitz
weatherStationIdJena = "02444"      # wheatherstationID Jena Sternwarte


# Data for Alfons-Pech-Straße, Chemnitz
alfonsPechStrData = prepareData('../data/PV/APS_PV/', weatherStationIdChemnitz, True, ",", 1)
executeFeasibilityAnalysis(alfonsPechStrData)


# Data for tanzende Siedlung, Chemnitz
tanzendeSiedlungData = prepareData('../data/TAS/inetz/', weatherStationIdChemnitz, True, ";", 2)

print("debug")
