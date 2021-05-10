# Explainable-AI-Classifiers-for-Alzheimers-Disease-Diagnosis

*Important Note*
This guidance assumes you have the following already installed.
* Python version 3.7.5
* Pip version 21.0.1
* NumPy version 1.19.2
* Pandas version 0.25.3
* MatPlotLib version 3.2.2
* SciKit-Learn version 0.22.1
* TensorFlow 2.2.0
* And requirements for hippodeep to be executed(for phase2) - https://github.com/bthyreau/hippodeep_pytorch/blob/master/README.md

The versions above match the development environment in which this software was built. Hence, newer versions might also work.

*Phase 1*

Training.
Run the program with "python3 main.py -clf X -cl Y -s Z" command-line argument.
X allows options: SVC, LR, DT. Y allows options: HC_MCI, MCI_AD, HC_AD, ALL or MULTI. Z is 0 or 1. 
Flag -clf indicates an available classifier(Support Vector classifier, gradient boosted decision trees, linear regression available), -cl indicates classes to classify (HC vs. MCI, MCI vs. AD, HC vs. AD or MULTI), and flag -s saves models and templates, models and feature importance if set to 1, does not save if 0.

To change required parameters, modify params.py file.

By default, models and their feature importance are saved in ../Models directory. Template 'Columns.csv' with other produced data files by default are saved in the ../DataFiles directory. 

Predicting the instance.
Once models are trained and saved, explain.py can be run by following "python3 explain.py -clf X -cl Y -d Z" command-line argument. X allows options: SVC, LR, DT. Y allows options: HC_MCI, MCI_AD, HC_AD, ALL, or MULTI. Z is a datafile. Flag -clf indicates a classifier(Support Vector classifier, gradient boosted decision trees, linear regression available), -cl indicates classes to classify (HC vs. MCI, MCI vs. AD, HC vs. AD or multi), and -d is a filled template with patients data, for prediction and explanation. A template might also have missing values, and the program would still run.

Predictions are saved in the predictions.csv file in the same directory as all source files, and both ELI5 and LIME explanations are stored in the ExplainHTML directory.

To run the program from other than phase1/src location requires changing the params file.

*Phase 2*

Training.
Run the program with "python3 main.py" command-line argument. 
The program saves trained models in the Models directory. Metrics are stored in Metrics/log.csv file, and accuracy improvement charts of both training and validation accuracy of MLPC and LeNet are also stored in Metrics/.

Predicting.
Run the program with "python3 predict.py" command-line argument.
'Final predictions' file would only make sense predicting either multi, single binary, all binary or all classifications. If the user requests two binary and multi, two binary, or two binary and multi classifications, 'final predictions' would fail to output sensible results, and they would be stored in a "detailed output" file rather than final.
This happens because for maximising the convenient output, the most common class for each individual is predicted. Thus, if a case is AD, and user specifies HC vs MCI prediction, outcome cannot be AD, whereas, in all binary classes, output would be AD(HC vs MCI = MCI; AD vs HC = AD, MCI vs AD = AD, by maximum value, AD is predicted).

Parameters for selecting which models to train, where to save a log file, images, path to hippodeep package, and intensity projections for training must be specified in params.py file.
Files for prediction must be in ".nii" NIfTI format.

In both phases, the program is expected to be executed in the src directory, or requires changing paths in params.py.
