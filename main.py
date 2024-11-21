from makeFilesForGattaranUpload import divide_per_country
from automater import doSelenium
from allMLStuff import predictFile

if __name__ == '__main__':
    predictFile('NewRankDataset_october22.csv')
    divide_per_country()
    doSelenium()
