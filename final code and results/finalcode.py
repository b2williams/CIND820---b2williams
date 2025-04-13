import numpy as np
import pandas as pd  # for data loading
import matplotlib.pyplot as plt  # for plotting
from sklearn.feature_selection import SelectKBest, f_classif  # for feature selection
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent Classifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc  # Display Metrics
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbours Classifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC  # classifier models for SVM
from imblearn.over_sampling import ADASYN  # for rescaling imbalanced y data
from imblearn.pipeline import Pipeline  # to develop pipelines


# Data Pre-Processing #
# Loads csv data into dataframe which can be manipulated for analysis
# Separates data into training/testing samples
# Provides alternative training/testing sample set with rescaling for imbalance in class representation

# load data to dataframe
dset = 'diabetes_012_health_indicators_BRFSS2015.csv'
df = pd.read_csv(dset)  # read csv into DataFrame with Pandas

# create training data
dfx = df.drop(['Diabetes_012'], axis='columns')  # isolate features by removing classification column
dfy = df.Diabetes_012  # isolate results

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=42)  # train/test

# Model Development #
# Covers development of LinearSVC, Stochastic Gradient Descent, and K-Nearest Neighbours (5 & 10) Classifiers.
# Develops a dictionary 'models' of pipelines to greatly simplify the code compared to original results. The pipelines
# each use ADASYN to ensure that target training data is rescaled for imbalance in class presence. The use of pipelines
# allow us to call specific steps in the model pipeline with ease. The use of dictionaries allows us to nest our new
# model pipelines into one convenient basket for us to iterate through when fine-tuning, plotting, or printing results.

models = {  # initialize dictionary
        "LinearSVC": Pipeline([  # initialize pipeline, less complicated workflow compared to original model development
            ('standardize', StandardScaler()),  # standardizes the data to prevent leakage
            ('adasyn', ADASYN(random_state=42)),  # resampling of training data accounting for imbalance of classes
            ('feature_selection', SelectKBest(score_func=f_classif, k=10)),  # 10 features, as selected from testing
            ('classifier', LinearSVC(dual=False, class_weight='balanced', random_state=42))  # set up classifier model
        ]),
        "SGDClassifier": Pipeline([
            ('standardize', StandardScaler()),
            ('adasyn', ADASYN(random_state=42)),
            ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
            ('classifier', SGDClassifier(loss='hinge', class_weight='balanced', random_state=42))  # hinge loss uses SVM
        ]),
        "KNN-5": Pipeline([
            ('standardize', StandardScaler()),
            ('adasyn', ADASYN(random_state=42)),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ]),
        "KNN-10": Pipeline([
            ('standardize', StandardScaler()),
            ('adasyn', ADASYN(random_state=42)),
            ('classifier', KNeighborsClassifier(n_neighbors=10))
        ])
    }

# Results #

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
model_scores = {}

for model_name, pipe in models.items():  # iterate through dictionary of classifier model pipelines

    # Cross Validation
    # Uses for loop to iterate through each of the developed pipelines, taking advantage of the models' dictionary
    # format. The cross-validation phase employs Stratified kFold to calculate stats based on 5 separate trials with
    # unique test and train sets. Cross-validation prints test precision, recall, accuracy, and f1 score means for each
    # model. Also prints time taken to process fit and score processing time for each model.

    print("\n", model_name, "Cross Validation")
    stats = cross_validate(pipe, dfx, dfy, cv=5, scoring=scoring)  # cross validation for multi-class using 5 folds
    model_scores[model_name] = [
        np.mean(stats['test_accuracy']),
        np.mean(stats['test_precision_weighted']),
        np.mean(stats['test_recall_weighted']),
        np.mean(stats['test_f1_weighted'])
    ]
    for stat in stats:  # iterate through the scores for the given classifier
        print(stat, ": ", stats[stat].mean())

    # ## Visualizations ## #

    # train and test model for future plotting
    pipe.fit(x_train, y_train)  # fit training data with each ML model
    class_prediction = pipe.predict(x_test)  # predict classes for each x-point in the test set

    # Confusion matrix - maps matrix of model's class predictions
    cm = confusion_matrix(y_test, class_prediction, normalize='true')  # init cm, normalize result into percentages
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.named_steps['classifier'].classes_)
    disp.plot(), plt.title(f"{model_name} Confusion Matrix"), plt.show()  # display plot

    # ROC & AUC curve - recall on y-axis, false positives on x-axis - one vs. rest strategy
    y_binary = label_binarize(y_test, classes=[0, 1, 2])  # Binarize the output for ROC, comparing one target vs. rest
    if hasattr(pipe.named_steps['classifier'], 'decision_function'):  # safe method of calling linearSVC/SGDClassifier
        y_score = pipe.decision_function(x_test)    # get probabilities of each class possible for a given x
    else:  # for KNN-5 and KNN-10
        y_score = pipe.predict_proba(x_test)

    for i in range(y_score.shape[1]):  # loop through each class
        false_pos, true_pos, _ = roc_curve(y_binary[:, i], y_score[:, i])  # get false positive/negative rates
        plt.plot(false_pos, true_pos, label=f"Class {i} (AUC = {auc(false_pos, true_pos):.2f})")

    plt.plot([0, 1], [0, 1], 'k--'), plt.title(f"{model_name} ROC Curve (One-vs-Rest)")
    plt.xlabel("False Positive Rate"), plt.ylabel("True Positive Rate")
    plt.legend(), plt.grid(), plt.show()

    # Learning Curves - Demonstrates how well a model learns over time with increasing training data availability
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, dfx, dfy,
        cv=5,  # number of stratified k folds
        scoring='recall_weighted',  # recall is a strong metric indicator when predicting healthcare diagnosis
        n_jobs=-1,  # maximum parallel processing for computation time
        train_sizes=np.linspace(0.1, 1.0, 10),  # splits up into 10 subsets of data for gradual learning
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)  # training score means
    val_mean = val_scores.mean(axis=1)  # validation score means

    plt.figure()
    plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
    plt.plot(train_sizes, val_mean, label='Validation Score', marker='o')
    plt.title(f"Learning Curve - {model_name}"), plt.xlabel("Training Set Size"), plt.ylabel("Accuracy")
    plt.ylim(0, 1), plt.legend(), plt.grid(), plt.tight_layout(), plt.show()


# Bar Chart comparing all fundamental metrics (Precision, Recall, Accuracy, F1)
score_df = pd.DataFrame(model_scores,  # create DataFrame for plt visualizations of metrics. '.T' flips cols and rows
                        index=['Accuracy', 'Precision', 'Recall', 'F1 Score']).T
score_df.plot(kind='bar', figsize=(10, 6), edgecolor='black')

plt.title("Model Comparison on Cross-Validation Metrics")
plt.ylabel("Score"), plt.ylim(0, 1), plt.xticks(rotation=45)
plt.legend(title="Metric", loc='lower right'), plt.grid(axis='y', linestyle='--', alpha=0.7), plt.tight_layout()
plt.show()

