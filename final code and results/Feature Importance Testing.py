import numpy as np
import pandas as pd  # for data loading
import matplotlib.pyplot as plt  # for plotting
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent Classifier
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC  # classifier models for SVM
from imblearn.over_sampling import ADASYN  # for rescaling imbalanced y data
from imblearn.pipeline import Pipeline  # to develop pipelines
import seaborn as sns


# This file is specifically focused on analyzing the importance of features in our LinearSVC and SGDClassifier models.
# Returns a bar chart identifying the absolute value coefficient of each feature in the respective fitted regression.

# load data to dataframe
dset = 'diabetes_012_health_indicators_BRFSS2015.csv'
df = pd.read_csv(dset)  # read csv into DataFrame with Pandas

# create training data
dfx = df.drop(['Diabetes_012'], axis='columns')  # isolate features by removing classification column
dfy = df.Diabetes_012  # isolate results

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=42)  # train/test

# Model Development #
# This file only covers LinearSVC and SGDClassifier as K-Nearest Neighbours classification does not identify feature
# importance in the same way that or linear or tree model would.

models = {  # initialize dictionary
        "LinearSVC": Pipeline([  # initialize pipeline, less complicated workflow compared to original model development
           ('standardize', StandardScaler()),  # standardizes the data to prevent leakage
           ('adasyn', ADASYN(random_state=42)),  # resampling of training data accounting for imbalance of classes
           ('classifier', LinearSVC(dual=False, class_weight='balanced', random_state=42))  # set up classifier model
        ]),
        "SGDClassifier": Pipeline([
            ('standardize', StandardScaler()),
            ('adasyn', ADASYN(random_state=42)),
            ('classifier', SGDClassifier(loss='hinge', class_weight='balanced', random_state=42))  # hinge loss uses SVM
        ])
    }


# Feature Importance
# Identifies the absolute values for the regression features (LinearSVC, SGDClassifier). Absolute values represent the
# magnitude effect a feature variable has on the dependent variable (our target classes).

for model_name, pipe in models.items():  # iterate through dictionary of classifier model pipelines
    pipe.fit(x_train, y_train)  # train model
    feature_names = x_train.columns  # store the features
    coefs = np.abs(pipe.named_steps['classifier'].coef_).flatten()  # get coefficients from the regression function

    feature_importance = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)  # formatting/sorting
    feature_x, feature_y = [],[]

    print(f"\n{model_name} Feature Importance:")
    for feature, importance in feature_importance:  # get x and y values for the plot
        feature_x.append(feature)
        feature_y.append(importance)
        print(f"{feature}: {importance}")

    # Bar plot for visualizing rank of feature importance
    plt.figure(figsize=(10, 6)), plt.barh(feature_x, feature_y)
    plt.xlabel('Feature Importance (Absolute Coefficient Value)'), plt.ylabel('Feature')
    plt.title(f'{model_name} Feature Importance')
    plt.gca().invert_yaxis(), plt.show()  # Invert y-axis to show the most important feature at the top

    # Step 1: Calculate the correlation matrix between features
    correlation_matrix = dfx.corr()

    # Step 2: Visualize the correlation matrix with a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()
