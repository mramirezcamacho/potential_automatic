from datetime import datetime
from predict_potential_functions import *


def part1():
    print("Let's print the first 5 cities and how many are there")
    print(list(divideDataPerCity().keys())[:5])
    print("There're:", len(list(divideDataPerCity().keys())), "cities")
    print(divideDataPerCity()['Medellin'].columns)


def unitaryTest(city: str = "Monterrey"):
    data = divideDataPerCity(downSample=False)
    randomForestAproach(data[city], 1)
    GBMAproach(data[city], 1)
    SVMAproach(data[city], 1)


def part3():
    cali = cityModel('Cali', df=divideDataPerCity(downSample=False)['Cali'])
    print(cali.nameOfModel)
    cali.saveModel()
    models = loadModels()
    caliLoaded = models['Cali']
    randoms = divideDataPerCity(downSample=False)['Cali']
    tries = 0
    victories = 0
    for index, random in randoms.iterrows():
        real = random['order_class']
        X = random.drop('order_class').to_frame().T
        prediction = caliLoaded.predict(X)
        if prediction == real:
            victories += 1
        tries += 1
    print(victories / tries)


def createModelPerCity():
    findBestModelPerCity(divideDataPerCity(downSample=False), save=True)


def part5():
    data = divideDataPerCity(downSample=False)
    data.keys()
    testExistingModels(data)


def get_most_recent_csv(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")

    csv_files = [
        os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')
    ]

    if not csv_files:
        raise ValueError

    most_recent_file = max(csv_files, key=os.path.getctime)

    return os.path.basename(most_recent_file)


def joinFiles(fileNameOfPredictions):
    dtype_dict = {'shop_id': str}
    addDF = pd.read_csv(f'ML/{PREDICTIONS}/{fileNameOfPredictions}', encoding='unicode_escape',
                        low_memory=False, dtype=dtype_dict)
    addDF = addDF.dropna(subset=['new_rank'])
    addDF = addDF.rename(columns={'new_rank': 'new_potential'})
    addDF = addDF.reset_index(drop=True)  # Reiniciar los índices
    df_filtrado = addDF
    df_filtrado[["country_code", "shop_id", "new_potential"]].to_csv(
        f"""gattaran_files/data_new_old_priority/data_{datetime.today().strftime('%Y-%m-%d')}.csv""", index=False)


def predictFile(fileToPredict):
    fileLocation = f'ML/{BASEDATA}/{fileToPredict}'
    saveLocation = f'ML/{PREDICTIONS}'
    fileNameOfPredictions = fileToPredict.split(
        '.')[0]+'_prediction.'+fileToPredict.split('.')[1]
    newData = divideDataPerCityForPrediction(fileLocation=fileLocation)
    print("I'm going to predict the potential of each Rs")
    predictUsingModels(newData, saveLocation,
                       fileNameOfPredictions, fileLocation)
    print("I'm finished with the the predictions file")
    joinFiles(fileNameOfPredictions)
    print("All ready and set")


if __name__ == '__main__':
    predictFile('NewRankDataset_cristobalnavarro_20241129002848.csv')
