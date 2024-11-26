from makeFilesForGattaranUpload import divide_per_country
from automater import doSelenium
from allMLStuff import predictFile

if __name__ == '__main__':
    predictFile('NewRankDataset (8).csv')
    divide_per_country()
    doSelenium()
