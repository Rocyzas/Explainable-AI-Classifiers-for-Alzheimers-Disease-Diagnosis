import time
import sys
from joblib import load

from DataProcessing import *
from Classifiers import *
from Explainer import *

def loadModel(name):
    model = load('Models/' + name + '.joblib')
    return model


def main(argv):
    model = loadModel(argv[0]+"-"+argv[1])
    print(type(model))

    predRows = pd.read_csv(argv[2])
    # print(dataRow)
    predRows = predRows.to_numpy()
    predRows = scaleData(predRows)
    yn = model.predict(predRows)
    # print(argv[0], argv[1])
    print("PREDICTION: ", yn)
    # print(dataRow)
    cols = pd.read_csv("Columns.csv")
    cols = cols.columns.tolist()
    cols.pop(0)

    # explainELI5(model)
    explain(predRows, cols, model, argv[0], argv[1])


if __name__ == '__main__':
    start_time = time.time()

    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
