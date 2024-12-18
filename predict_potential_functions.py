from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import glob
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
warnings.filterwarnings('ignore')


MODELS = 'models'
PREDICTIONS = 'predictions'
BASEDATA = 'baseData'

BASICFILE = f'ML/{BASEDATA}/NewSegmentationDataset_V2.csv'


def getData(fileLocation: str):
    data = pd.read_csv(
        fileLocation, encoding='unicode_escape', low_memory=False)
    data = data.rename(columns=lambda x: x.strip())

    data = data[data['days_in_data'] == 14]
    data.drop(['days_in_data', 'complete_orders', 'set_hours', 'active_days', 'traffic', 'shop_enter_uv', 'online_hours', 'b_p1', 'b_p2',  # 'order_class_2',
               'r_burn_per_order', 'b_burn_per_order', 'b2c_burn_per_order', 'p2c_burn_per_order', 'shop_id', 'country_code',
               # 'b_p1p2'
               ], axis=1, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    return data


def prettify_city_names(city_name: str):
    if 'medell' in city_name.lower():
        return 'Medellin'
    elif 'cal' in city_name.lower()[:3]:
        return 'Cali'
    elif 'canc' in city_name.lower()[:4]:
        return 'Cancun'
    elif 'bogot' in city_name.lower()[:6]:
        return 'Bogota D.C'
    else:
        return city_name.replace("¨¢_", "a").replace('¨ª', 'i').replace("¨²", 'u').replace("¨¦", "e").replace("¨®", "o")


def downSampling(dataPerCity: dict) -> dict:
    downsampledData: dict = {}
    for city, data in dataPerCity.items():
        # Find the minimum class count in this city's dataset
        min_rows = data['order_class'].value_counts().min()

        # Downsampling each class to have count equal to min_rows
        downsampled_data = pd.concat([data[data['order_class'] == group].sample(
            min_rows, random_state=42) for group in data['order_class'].unique()])

        # Store the downsampled data for each city
        downsampledData[city] = downsampled_data
    return downsampledData


def divideDataPerCity(fileLocation: str = BASICFILE, downSample: bool = False):
    data = getData(fileLocation)
    data['city_name'] = data['city_name'].apply(prettify_city_names)
    cities = data['city_name'].unique()
    try:
        city_datasets = {city: data[data['city_name'] == city].drop(
            ['city_name', 'potential', 'r_performance'], axis=1) for city in cities}
    except:
        city_datasets = {city: data[data['city_name'] == city].drop(
            ['city_name'], axis=1) for city in cities}
    if downSample:
        city_datasets = downSampling(city_datasets)
    return city_datasets


# Assuming X contains your numerical features and y contains your ratings (SS, AA, BB)
def randomForestAproach(data, showData: bool = False):
    Y = data["order_class"]
    X = data.copy().drop('order_class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    if showData:
        print("Random Forest Performance:")
        print(classification_report(y_test, rf_pred))
    return (accuracy_score(y_test, rf_pred), rf)


def GBMAproach(data, showData: bool = False):
    # Encode target variable
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(data["order_class"])

    X = data.copy().drop('order_class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    gbm = xgb.XGBClassifier(random_state=42)
    gbm.fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)

    if showData:
        print("Gradient Boosting Performance:")
        print(classification_report(y_test, gbm_pred))
    return (accuracy_score(y_test, gbm_pred), gbm)


# Scale your numerical features (important for SVMs)

def SVMAproach(data, showData: bool = False):
    Y = data["order_class"]
    X = data.copy().drop('order_class', axis=1)

    # Crear el pipeline con el escalador y el clasificador SVM
    svm_model = make_pipeline(StandardScaler(), SVC(random_state=42))

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    if showData:
        print("SVM Performance:")
        print(classification_report(y_test, svm_pred))
    return accuracy_score(y_test, svm_pred), svm_model


class cityModel:

    def __init__(self, name, df=None, model=None, nameOfModel=None):
        self.name = name
        if df is not None:
            self.df = df
            self.bestModel, self.bestAccuracy, self.nameOfModel = self.getBestModel()
        elif model is not None:
            self.bestModel = model
            if nameOfModel != None:
                self.nameOfModel = nameOfModel

        else:
            raise ValueError("Either 'df' or 'model' must be provided")

    def evaluateRandomForest(self):
        return randomForestAproach(self.df)

    def evaluateSVM(self):
        return SVMAproach(self.df)

    def evaluateGBM(self):
        return GBMAproach(self.df)

    def getBestModel(self):
        bestModel = None
        bestName = ''
        bestAcurracy = 0
        RandomForest = self.evaluateRandomForest()
        SVM = self.evaluateSVM()
        GBM = self.evaluateGBM()
        if RandomForest[0] > bestAcurracy:
            bestAcurracy = RandomForest[0]
            bestModel = RandomForest[1]
            bestName = 'Random_Forest'
        if SVM[0] > bestAcurracy:
            bestAcurracy = SVM[0]
            bestModel = SVM[1]
            bestName = 'SVM'
        if GBM[0] > bestAcurracy:
            bestAcurracy = GBM[0]
            bestModel = GBM[1]
            bestName = 'GBM'
        return bestModel, bestAcurracy, bestName

    def renewFolder(self):
        carpeta = f'ML/{MODELS}/{self.name}'
        archivos = glob.glob(os.path.join(carpeta, '*'))

        for archivo in archivos:
            if os.path.isfile(archivo):
                os.remove(archivo)

    def saveModel(self):
        self.renewFolder()
        folder_path = f'''ML/{
            MODELS}/{self.name}/'''
        os.makedirs(folder_path, exist_ok=True)
        fileName = folder_path + f'Model_{self.name}_{self.nameOfModel}.joblib'
        try:
            joblib.dump(self.bestModel, fileName)
            print(f'{self.name} file was saved successfuly')
        except Exception as e:
            print('We had the error:', e)

    def predict(self, X_parametro):
        X = X_parametro.copy()
        if 'order_class' in X:
            X = X.drop('order_class', axis=1)
        else:
            X = X_parametro.copy()
        if self.nameOfModel not in ["Random Forest", 'SVM']:
            X = X.values.reshape(1, -1)
        prediction = self.bestModel.predict(X)
        if hasattr(self.bestModel, 'predict_proba'):
            prediction = self.bestModel.predict_proba(X)
            predicted_class_index = np.argmax(prediction)
        else:
            predicted_class_index = prediction[0]
        if predicted_class_index in ['AA', 'BB', 'SS']:
            translation = {'AA': 'T2', 'BB': 'T3', 'SS': 'T1'}
            return translation[predicted_class_index]
        if predicted_class_index in ['T1', 'T2', 'T3']:
            return predicted_class_index
        # BEFORE
        # predicted_class = ['AA', 'BB', 'SS'][predicted_class_index]
        # NOW
        predicted_class = ['T1', 'T2', 'T3'][predicted_class_index]
        return predicted_class


def list_dirs_in_folder(folder_path: str = f'ML/{MODELS}/', onlyFiles: bool = False):
    try:
        # List all files in the given folder
        if not onlyFiles:
            folders = os.listdir(folder_path)
            folders = [folder_path + folder for folder in folders]
            return folders
        else:
            files = os.listdir(folder_path)
            files = [file for file in files]
            return files

    except Exception as e:
        print('We had the error:', e)
        return []


def loadModel(city: str, modelName: str, modelPath: str = f'ML/{MODELS}/'):
    modelPath += city + '/'
    model = joblib.load(modelPath+modelName)
    return model


def loadModels():
    folders = list_dirs_in_folder()
    modelsPerCity: dict = {}
    for folder in folders:
        files = list_dirs_in_folder(folder, True)
        for fileName in files:
            miniData = fileName[:fileName.rfind('.')].split('_')
            cityName, modelName = miniData[1], miniData[2]
            model = loadModel(cityName, fileName)
            cityObject = cityModel(
                name=cityName, nameOfModel=modelName, model=model)
            modelsPerCity[cityName] = cityObject
    return modelsPerCity


def findBestModelPerCity(data, testSize: int = -1, save: bool = True) -> dict:
    citiesObjects: list = []
    i = 0
    for city, df_city in data.items():
        citiesObjects.append(cityModel(city, df_city))
        i += 1
        if i == testSize:
            break
    if save:
        for city in citiesObjects:
            city.saveModel()


def testExistingModels(data):
    models = loadModels()
    for city, cityObj in models.items():
        dataFromCity = data[city]
        tries = 0
        victories = 0
        for index, cityData in dataFromCity.iterrows():
            real = cityData['order_class']
            X = cityData.drop('order_class').to_frame().T
            prediction = cityObj.predict(X)
            if prediction == real:
                victories += 1
            tries += 1
        print(f'Performance of {city}:', victories / tries)


def getDataToPredict(fileLocation: str):
    dtype_dict = {'shop_id': str, 'city_name': str, 'country_code': str}
    data = pd.read_csv(fileLocation, encoding='unicode_escape',
                       low_memory=False, dtype=dtype_dict)
    data = data.rename(columns=lambda x: x.strip())
    data = data.rename(
        columns=lambda col: 'city_name' if 'city_name' in col.lower() else col)
    # For now will not make this
    data = data[data['days_in_data'] > 13]
    columns_to_keep = [
        'shop_id', 'city_name', 'eff_online_days', 'asp', 'online_rate', 'is_healthy_store',
        'b_cancel_rate', 'activity_days', 'recurrency_rate',
        'commission_per_order', 'ted_per_order', 'photo_rate', 'b_p1p2']
    for col in columns_to_keep:
        if col not in data.columns:
            data[col] = 0
            print(f'la columna {col} no existe D:')
    df = data[columns_to_keep]
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    return df


def divideDataPerCityForPrediction(fileLocation):
    data = getDataToPredict(fileLocation)
    data['city_name'] = data['city_name'].apply(prettify_city_names)
    cities = data['city_name'].unique()
    city_datasets = {city: data[data['city_name'] == city].drop(
        'city_name', axis=1).reset_index() for city in cities}
    return city_datasets


def predictUsingModels(data: dict, saveLocation: str, fileNameOfPredictions: str, baseDataPath: str):
    models = loadModels()
    predictionsDF = pd.DataFrame(columns=['city', 'shop_id', 'new_rank'])
    predictions = []
    for city, df_city in data.items():
        try:
            cityObj = models[city]
        except Exception as e:
            print(f"{city} data can't be predicted", e)
            continue
        for index, row in df_city.iterrows():
            shop_id = row['shop_id']
            X = row.drop('shop_id').drop('index').to_frame().T
            prediction = cityObj.predict(X)
            new_row = {'city': city, 'shop_id': shop_id,
                       'new_rank': prediction}
            predictions.append(new_row)
    predictionsDF = pd.DataFrame(
        predictions, columns=['city', 'shop_id', 'new_rank'])
    predictionsDF['shop_id'] = predictionsDF['shop_id'].astype(str)
    df2 = pd.read_csv(baseDataPath, encoding='unicode_escape',
                      low_memory=False, dtype={'shop_id': str})
    result = pd.merge(
        predictionsDF, df2[['shop_id', 'country_code']], on='shop_id', how='left')
    result[['country_code', 'city', 'shop_id', 'new_rank']].to_csv(
        f'{saveLocation}/{fileNameOfPredictions}', index=False)
