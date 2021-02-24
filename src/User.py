import time
import sys
from joblib import load

from Params import *
from LRclf import *
from SVMclf import *
from DTclf import *

from Explainer import *

classifiersList = ['SVM', 'LR', 'DT']
classesList = ['HC_AD', 'MCI_AD']

def main(argv):
    # no need to load a model in userpy. it should be in explainer.


    predRows = pd.read_csv(argv[2])

    if argv[0]=='ALL' and argv[1]=='ALL':
        for model in classifiersList:
            for cl in classesList:
                explain(predRows, model, cl)

    elif argv[1]=='ALL':
        for cl in classesList:
            explain(predRows, argv[0], cl)

    elif argv[0]=='ALL':
        for model in classifiersList:
            explain(predRows, model, argv[1])
    else:
        explain(predRows, argv[0], argv[1])


if __name__ == '__main__':
    start_time = time.time()

    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
