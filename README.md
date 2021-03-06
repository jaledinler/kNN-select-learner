

CS 760 Homework Assignment

A k nearest neighbor learner for both classification and regression is implemented according to the following guidelines:

For classification tasks, the class attribute is named 'class' and it is the last attribute listed in the header section. 
Similarly, for regression tasks, the response is named 'response' and it is the last attribute listed in the header section.

All features will be numeric. Euclidean distance to compute distances between instances is used. Whether the given data set is a 
classification or regression task depending on whether the last attribute in the trainingset ARFF file is named 'class' or 
'response'. If there is a tie among multiple instances to be in the knearest neighbors, the tie is broken in favor of those 
instances that come first in the ARFF file. If there is a tie in the class predicted by the knearest neighbors, then among the 
classes that have the same number of votes, the tie is broken in favor of the class comes first in the ARFF file.

Program accepts five commandline arguments as follows:
kNN‐select <train‐set‐file> <test‐set‐file> k1 k2 k3.

This program uses leave-one-out cross validation with just the training data to select the value of k to use for the test set 
by evaluating k1 k2 k3 and selecting the one that results in the minimal crossvalidated error within the training set. 
To measure error for regression tasks, you should use mean absolute error.

Output:

The value of k used for the test set on the first line and then the predictions for the test set instances are printed. 
For each instance in the test set, one line of output with spaces separating the fields is printed. For a classification task, 
each output line lists the predicted class label and actual class label. It is followed by a line listing the number of correctly 
classified test instances, and the total number of instances in the test set.

For a regression task, each output line lists the predicted response, and the actual response value. This should beIt is followed 
by a line listing the mean absolute error for the test instances, and the total number of instances in the test set.
