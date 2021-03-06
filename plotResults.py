import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec

def createReusltChart(sourceFilePath, targetFilePath, color="b"):
        # load and format data
        data = pd.read_csv(sourceFilePath, delimiter=";")
        data_types_dict = {
                'LSTM Units': str,
                'batch_size': str,
                'dropout': str,
                'lossFunction': str,
                'steps_into_past': str
        }
        data = data.astype(data_types_dict)

        # create plot
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3)

        # plot lstm units
        axis = plt.subplot(gs[0, 0])
        axis.bar(data.loc[0:3, "LSTM Units"], data.loc[0:3, "Durchschnitt"], color=color)
        axis.set_title("LSTM Units")

        # plot batch_size
        axis = plt.subplot(gs[0, 1])
        axis.bar(data.loc[4:6, "batch_size"], data.loc[4:6, "Durchschnitt"], color=color)
        axis.set_title("Batch size")

        # plot dropout, overlap with batch_size
        axis = plt.subplot(gs[0, 2])
        axis.bar(data.loc[6:8, "dropout"], data.loc[6:8, "Durchschnitt"], color=color)
        axis.set_title("Dropout")

        # plot steps into past
        axis = plt.subplot(gs[1, 0])
        axis.bar(data.loc[[16,17,18], "steps_into_past"], data.loc[[16,17,18], "Durchschnitt"], color=color)
        axis.set_title("Schritte in die Vergangenheit")


        # plot optimization function results
        lossFunctions = data.loc[8:14, "lossFunction"]
        replaceDict = {
                "mae": "Mean \n average \n error",
                "<keras.losses.CosineSimilarity object ": "Cosine \n similarity",
                "<keras.losses.Huber object ": "Huber loss",
                "<keras.losses.MeanAbsolutePercentageError ": "Mean \n Absolute \n Percentage \nError",
                "<keras.losses.MeanSquaredLogarithmicError ": "Mean \n Squared \n Logarithmic \nError",
                "<keras.losses.LogCosh object": "Log Cosh",
                "mse": "Mean \n squared \n error"
        }

        # replace keys of the function with readable names
        lossFunctions.reset_index()
        for index, entry in enumerate(lossFunctions):
                for replaceKey, replaceEntry in enumerate(replaceDict):
                        if entry.find(replaceEntry) != -1:
                                lossFunctions.iloc[index] = replaceDict[replaceEntry]

        axis = plt.subplot(gs[1, 1:])
        axis.bar(data.loc[8:14, "lossFunction"], data.loc[8:14, "Durchschnitt"], color=color)
        axis.set_title("Optimierungsfunktionen")

        plt.tight_layout

        # save the plot as file
        plt.savefig(targetFilePath, bbox_inches='tight', dpi=150)
        plt.show()

# sourceFilePath = "../results_remote/aps_regression_60minutes/aps_60min_regressionModels.csv"
# targetFilePath = "../results_remote/aps_regression_60minutes/"

sourceFilePath = "../results_remote/ts_regression_feedin_60minutes/ts_feedIn_60min_regressionModels.csv"
targetFilePath = "../results_remote/ts_regression_feedin_60minutes/ts_feedIn_60min_regressionModels_results.jpg"




createReusltChart(sourceFilePath, targetFilePath, None)