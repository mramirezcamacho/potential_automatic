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


def predictFile(fileToPredict):
    fileLocation = f'ML/{BASEDATA}/{fileToPredict}'
    saveLocation = f'ML/{PREDICTIONS}'
    fileNameOfPredictions = fileToPredict.split(
        '.')[0]+'_prediction.'+fileToPredict.split('.')[1]
    newData = divideDataPerCityForPrediction(fileLocation=fileLocation)
    predictUsingModels(newData, saveLocation,
                       fileNameOfPredictions, fileLocation)
