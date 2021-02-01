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
