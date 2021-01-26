# Explainable-AI-Classifiers-for-Alzheimers-Disease-Diagnosis

What I did:
Kernel function - linear and poly gives the best results - ~80% with all data, but even 99% without sMRI and ASL.
This is because of 2 reasons.
	First: the latter files has many NaN values and currenly I am just deleting the rows -dropna.
	Second: Neuro file has CDR and MMRE information, that is almost the same as defining wether person has AD or not.
I should propose a better way of dealing with missing values.(and I will also need to think about how to deal with Neuro file.)
MinMAxScaler increased the performance of SVM by 10%.

