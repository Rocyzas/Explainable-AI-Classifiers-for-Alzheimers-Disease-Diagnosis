# Explainable-AI-Classifiers-for-Alzheimers-Disease-Diagnosis
python3 Model.py 'clasisfier' 'classification_method' '0/1(explainability)'

What I did:
Kernel function - linear and poly gives the best results - ~80% with all data, but even 99% without sMRI and ASL.
This is because of 2 reasons.
	First: the latter files has many NaN values and currenly I am just deleting the rows -dropna.
	Second: Neuro file has CDR and MMRE information, that is almost the same as defining wether person has AD or not.
I should propose a better way of dealing with missing values.(and I will also need to think about how to deal with Neuro file.)
MinMAxScaler increased the performance of SVM by 10%.

Using median for NaN values replacement
Fixed the training. Now using cross_val_predict, which for each element in the input returns the prediction, obtained for that element when it was in the test set(each elem was once in test set).
Got confustion matrix.
Accuracies now is between 80-88% for both linear and polynomial classifiers depending on cv.

Gradient Boosted Decision Tree gives precision of 73% at best...

Logistic regression with MinMAx scaler, lbfgs solver, cross validation,  and C=3 gives 87% accuracy.

GridSearchCV for DT
Using BayesSearchCV for hyperparameters.
Warning 'UserWarning: The objective has been evaluated at this point before' means that minimizer keeps suggesting points at which the objective has already been evaluated.
Hence number of iterations should be smaller, or add additional parameters.

There was a bug with suffle. I shuffled X and y separately, and hence labels were missmathced with the correct data row in X.
BayesSearch gave predictions over 50%, because it selected HParams only for that specific data.

Fixed Shuffle by zipping and unzipping values, fixed scaling for explainability by appending X wit filldata and then unconcatinating both.

Fixed the bug with empty fillData when training.

Now it works when the data is not in order
New datasets added, it also works with them.
Explainability now accepts classification mode in all LR, SVM, DT.
There was no need in scaling separately XY[3], since getXY scales it.
Problems: Explainability does not work as intended(inacurate for SVM, LR, and does not make sensible explanation for all models)

Made ALL variants in Exaplainer, but they arent tested yet.

Shuffle fucntion works on Explain and Train, The data file is saved once.

penalty C for polynomial kernel in bayessearchcv reduced. As C gets bigger, the model tries to reduce the penalty, and so takes more time to train.

PART 1 TODO
*Select Specific features, which makes biggest impact on classification | -
*Do a separate classification on different data sets(tests, sMRI, ALS)  | -
*Save all files only once when running program(regardless of options)   | o
*Use other explanations, not only LIME, for example ELI5, or diagrams DT| -
*Test data processing functions, models(on f score, recall, spec, sens) | -
*Make as much functions as possible to accept and return same type      | -
*Handle exceptions                                                      | -
*AUC and ROC curves in % (saving scores into log)                       | o
*Make MCI_HC                                                            | o
*partial volume features(ADNI + Sheffield) and normalise sheffield data | o
*better to predicct False Positive AD than False Negative in AD         | -
*Test argv[] input (maybe inside tests)                                 | o
*If col does not exist in drop list, just skip it, dont exit            | o
*CHeck if the training porcess and MCI data processing is correct, since
	it gives 0 recall and 0 TP                                      | -
*Precision-Recall diagram containing the f-score for each tissue sample?| -
*Rename HC, AD, MCI to GROUPS instead of classes			| -
*Make read once in explain         for eli5 and lime                    | -

PART2
Downloaded ADNI1Screening1.5 data ~1k 3D MRI images
Extracted Left and Right hippcampal with hippodeep-pytorch-master software
Software Extracted eTIV, hipL, hipR cols.
First, I extracted Subject name from it.
Then, from extracted Subject column I merged it with ROSTER file to obtain the RID
Then, From ADNI1_Screening_1.5T csv file by Subject I obtained Group(hc,ad,mci), Age, gender.
Because of duplicate rows (images) I averaged data of a patient (google the reason for adni duplicates).

Feature importance looks great, however, the recall is low for MCI_AD(on every classifier).
Accuracy for all are pretty high

Merged ADNI_sheff 'fulldata.csv' dataset with grouped, named it F01.csv and now model performance slightl increased.


PART 2 TODO

*For evaluation: Dice similarity coefficient(DSC), senisitivity, positive predicted values(PPV), volume error(VE).

Plan Part2
Read nii data with nibabel library and convert it to numpy array.
Train CNN (LeNet5) on that numpy array.
Test it

