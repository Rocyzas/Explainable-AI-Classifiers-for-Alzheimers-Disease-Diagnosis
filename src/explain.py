import time
import sys
import pandas as pd

from params import *
from PythonParser import parserExplainARGV
from Explainer import explainLIME, explainELI5

classifiersList = ['SVM', 'LR', 'DT']
classesList = ['HC_AD', 'MCI_AD', 'HC_MCI']

def main(argv):
    # no need to load a model in userpy. it should be in explainer.

    args = parserExplainARGV(argv)
    predRows = pd.read_csv(args.data)

    if args.classifier=='ALL' and args.classification=='ALL':
        for model in classifiersList:
            for cl in classesList:
                explainLIME(predRows, model, cl)
                explainELI5(predRows, model, cl)

    elif args.classification=='ALL':
        for cl in classesList:
            explainLIME(predRows, args.classifier, cl)
            explainELI5(predRows, args.classifier, cl)

    elif args.classifier=='ALL':
        for model in classifiersList:
            explainLIME(predRows, model, args.classification)
            explainELI5(predRows, model, args.classification)

    else:
        explainLIME(predRows, args.classifier, args.classification)
        explainELI5(predRows, args.classifier, args.classification)



if __name__ == '__main__':
    start_time = time.time()

    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
