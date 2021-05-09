# Explainable-AI-Classifiers-for-Alzheimers-Disease-Diagnosis
Phase 1

Run the training.
Compile and run the program with "python3 main.py -clf X -cl Y -s Z" command-line argument, where X is SVC, LR, DT, Y is HC_MCI, MCI_AD, HC_AD, ALL or MULTI, and Z is 0/1. Flag -clf indicates an available classifier(Support Vector classifier, gradient boosted decision trees, linear regression available), -cl indicates classes to classify (HC vs. MCI, MCI vs. AD, HC vs. AD or multi), and flag -s saves models and templates, models and feature importance if set to 1, does not save if 0.

To change model training parameters, directories, file paths, features to remove, which datasets to use, hyperparameters in a search space, metrics training score for Bayesian Optimisation, and other settings, modify params.py file.

By default, models and their feature importance are saved in ../Models directory. Template 'Columns.csv' with other produced data files are saved in the ../DataFiles directory. 

Predicting the instance.
Once models are trained and saved, explain.py can be run by following "python3 explain.py -clf X -cl Y -d Z" command-line argument. X is SVC, LR, DT, Y is HC_MCI, MCI_AD, HC_AD, ALL, or MULTI, and Z is a datafile. Flag -clf indicates a classifier(Support Vector classifier, gradient boosted decision trees, linear regression available), -cl indicates classes to classify (HC vs. MCI, MCI vs. AD, HC vs. AD or multi), and -d is a filled template with patients data, for prediction and explanation. A template might also have missing values, and the program would still run.

Predictions are saved in the predictions.csv file in the same directory as all source files, and both ELI5 and LIME explanations are stored in the ExplainHTML directory.

TO run the program from other location than phase1/src, would require changes in params file.

Phase 2

Run the training.
To train both MLPC and LeNet "python3 main.py" command line should be executed. The program saves trained models in Models directory, metrics are stored in Metrics/log.csv file and accuracy improvement charts of both training and validation accuracy of MLPC and LeNet are also stored in Metrics/.

To predict instances, run "python3 predict.py".
'Final predictions' file would only make sense predicting either multi, single binary or all classifications. Hence, if the user requests two binary and multi, two binary, or two binary and multi classifications, 'final predictions' would fail to output sensible results, and they would be stored in detailed output rather than final. This happens because for maximising the convenient output, the most common class for each individual is predicted. Thus, if case is AD, and user specifies HC vs MCI prediction, outcome cannot be AD, whereas, in all binary classes, output would be AD(HC vs MCI = MCI; AD vs HC = AD, MCI vs AD = AD, by maximum value, AD is predicted).

Parameters for selecting which models to train, where to save a log file, images, path to hippodeep package, and intensity projections for training must be specified in params.py file by the user before running the code.
For predicting instances modify params.py file for a path to save predictions and specify which classifiers and intensity matrices use for prediction. Files for prediction must be in ".nii" format.

In both phases, program for either training or predicting/explaining should be run in the src directory, or alternatively, requires changing paths in params.py.
