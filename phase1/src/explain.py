import time
import sys
import pandas as pd

from params import *
from PythonParser import parserExplainARGV
from Explainer import explainLIME, explainELI5
from DataProcessing import getDf, getXY

classifiersList = ['SVM', 'LR', 'DT']
classesList = ['HC_AD', 'MCI_AD', 'HC_MCI']

def main(argv):
    # no need to load a model in userpy. it should be in explainer.

    args = parserExplainARGV(argv)
    predRows = pd.read_csv(args.data)

    # get dataframe of all the data for explainer
    df = getDf()

    if args.classifier=='ALL' and args.classification=='ALL':
        for model in classifiersList:
            for cl in classesList:
                XY = getXY(df, cl, predRows)
                explainLIME(predRows, model, cl, XY)
                explainELI5(predRows, model, cl, XY)

    elif args.classification=='ALL':
        for cl in classesList:
            XY = getXY(df, cl, predRows)
            explainLIME(predRows, args.classifier, cl, XY)
            explainELI5(predRows, args.classifier, cl, XY)

    elif args.classifier=='ALL':
        for model in classifiersList:
            XY = getXY(df, args.classification, predRows)
            explainLIME(predRows, model, args.classification, XY)
            explainELI5(predRows, model, args.classification, XY)

    else:
        XY = getXY(df, args.classification, predRows)
        explainLIME(predRows, args.classifier, args.classification, XY)
        explainELI5(predRows, args.classifier, args.classification, XY)



if __name__ == '__main__':
    start_time = time.time()

    main(sys.argv[1:])

    print("--- %.2f seconds ---" % (time.time() - start_time))
