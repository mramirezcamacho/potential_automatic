from makeFilesForGattaranUpload import divide_per_country
from automater import doSelenium
from allMLStuff import predictFile, createModelPerCity
import sys


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file = sys.argv[1]
        if '.csv' not in file:
            file = file+'.csv'
    else:
        print("No input file provided, using default file.")
        file = 'NewRankDataset_cristobalnavarro_20241129002848.csv'

    if len(sys.argv) > 2:
        if 'train' in sys.argv[2]:
            createModelPerCity()

    predictFile(file)
    divide_per_country()
    doSelenium()
