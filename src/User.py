import time
import sys
from joblib import load

from Params import *
from LRclf import *
from SVMclf import *
from DTclf import *

from Explainer import *

def main(argv):
    model = loadModel(argv[0]+"-"+argv[1])

    predRows = pd.read_csv(argv[2])

    explain(predRows, model, argv[0], argv[1])


if __name__ == '__main__':
    start_time = time.time()

    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
