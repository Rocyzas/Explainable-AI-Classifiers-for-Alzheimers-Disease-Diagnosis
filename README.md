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

