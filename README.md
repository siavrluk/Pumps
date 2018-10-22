# Pumps

The goal of this competition is to predict which of the given pumps are functional, which need repair, and which do not work at all.

The data comes in 3 files:

* Training set values -- contains the independent variables for the training set
* Training set labels -- contains the dependent variable (status_group) for each of the rows in Training set values
* Test set values -- contains the independent variables that need predictions

The original training and test set have 40 features most of which are categorical.
One issue with the data is that there are many missing values. We handle them in the following way. For categorical variables we define new category "uknown" to denote the missing values. Numerical features with more than 35% will not be used. For the rest of the numerical variables we use mean/median to impute the missing entries. 
Another issue is that few of the categorical features have more than 2000 unique values. For these we select the top 20 unique values and group the rest in new category "other".
Since some of the categorical features represent the same information on different level of granularity, we choose to use just one of them.

After dropping some of the features and engineering a few new ones we are left with 13 predictors.

Since some of the algorithms work better when the data satisfies some conditions, we do the following two transformations of the data:

* First we dummify the categorical variables. The resulting datasets have 144 columns.
* Then the dimmified dataset is reduced to 70 dimensions via PCA.

After these transformations, there are three different representations of the data. We refer to those as train, train_dummy and train_pca70.

In the following table we summarize the performing result for different algorithms. Note that in the CV column the score is calculated via 3-fold cross-validation on the training data. The test results are calculated only once and are obtained via submitting the predictions at DrivenData (https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/submissions/). Bolded are the algorighms that performed best on the train and on the test sets. 

When scoring accuracy, the simplest benchmark is to predict the most common class -- in our case, "functional". This benchmark would score an accuracy of 0.5418 on the training set.


| Algorithm | Dataset | CV | Test |
|---------- | ------- | -- | ---- |
| Logistic Regression | train_dummy | 0.7383 | 0.7349 | 
| Neural Network | train_pca70 | 0.7720 | 0.7810 | 
| KNN, k = 10 | train_pca70 | 0.7690 | 0.7746 | 
| KNN, k = 20 | train_pca70 | 0.7621 | 0.7679 | 
| KNN, k = 50 | train_pca70 | 0.7480 | 0.7531 |
| KNN, k = 100 | train_pca70 | 0.7321 | 0.7378 |
| Naive Bayes | train_dummy | 0.3543 | 0.3776 | 
| Decision Tree | train | 0.7607 | 0.7694 | 
| **Random Forest** | train | 0.7979 | **0.8069** | 
| **xgboost** | train | **0.8010** | 0.8045 | 
| SVM, kernel = radial | train_pca70 | 0.7819 | 0.7874 | 
| SVM, kernel = linear | train_pca70 | 0.7251 | 0.7226 | 
| SVM, kernel = polynomial | train_pca70 | 0.7832 | 0.7910 | 
| SVM, kernel = sigmoid | train_pca70 | 0.5939 | 0.6131 | 
