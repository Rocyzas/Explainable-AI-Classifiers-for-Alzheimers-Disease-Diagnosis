import time
import sys
from joblib import load

from Params import *
from LRclf import *
from SVMclf import *
from DTclf import *

from Explainer import *

def main(argv):
    # model = loadModel(argv[0]+"-"+argv[1])

    predRows = pd.read_csv(argv[2])

    # explain(predRows, model, argv[0], argv[1])
    # explain(predRows, loadModel("LR-HC_AD"), "LR", "HC_AD")
    explain(predRows, loadModel("LR-MCI_AD"), "LR", "MCI_AD")

    # explain(predRows, loadModel("DT-HC_AD"), "DT", "HC_AD")
    # explain(predRows, loadModel("DT-MCI_AD"), "DT", "MCI_AD")

    # explain(predRows, loadModel("SVM-HC_AD"), "SVM", "HC_AD")
    # explain(predRows, loadModel("SVM-MCI_AD"), "SVM", "MCI_AD")

if __name__ == '__main__':
    start_time = time.time()

    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
