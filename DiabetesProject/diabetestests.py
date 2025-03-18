import pandas as pd  # for data loading
from sklearn.model_selection import train_test_split  # organizes data into training and testing data
from sklearn.svm import LinearSVC, SVC  # classifier models for SVM
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent Classifier
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbours Classifier
from sklearn.preprocessing import StandardScaler  # Standardizes data for KNN Classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # model performance metrics
import seaborn as sns  # for plotting
import matplotlib.pyplot as plt  # for plotting


# Data Processing #

# load data to dataframe
dset = 'diabetes_binary_health_indicators_BRFSS2015.csv'
df = pd.read_csv(dset)  # read csv into dataset with pandas

# create training data
dfx = df.drop(['Diabetes_binary'], axis='columns')  # only features
dfy = df.Diabetes_binary  # target results
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2, random_state=1998)  # train/test


# Model Development #

# linearSVC model
lsvc = LinearSVC(dual=False)  # create model
lsvc.fit(x_train, y_train)  # fit the training data to model
lsvc_p = lsvc.predict(x_test)  # makes predictions on test set based on training
lsvc_cm = confusion_matrix(y_test, lsvc_p)  # develop confusion matrix comparing predictions to answers
print("LSVC Classification Report:\n", classification_report(y_test, lsvc_p))

# sgdClassifier model
sgdc = SGDClassifier(loss='hinge')  # create model, hinge gives linear SVM
sgdc.fit(x_train, y_train)  # fit training data to model
sgdc_p = sgdc.predict(x_test)  # makes predictions on test set based on training
sgdc_cm = confusion_matrix(y_test, sgdc_p)  # develop confusion matrix comparing predictions to answers
print("SGDC Classification Report:\n", classification_report(y_test, sgdc_p))

# SVC model - has been scaled down to 50000 training samples to reduce load/processing time
xt = x_train.iloc[0:49999]  # scale down training data
yt = y_train.iloc[0:49999]
xtest = x_test[0:999]  # scale down test data
ytest = y_test[0:999]
svc = SVC().fit(xt, yt)  # create model
svc_p = svc.predict(xtest)  # predict on new test set
svc_cm = confusion_matrix(ytest, svc_p)  # develop confusion matrix comparing predictions to answers
print("SVC Classification Report:\n", classification_report(ytest, svc_p))

# KNN model
scaler = StandardScaler()  # standardizing tool
x_t = scaler.fit_transform(x_train)  # standardized training set
x_te = scaler.transform(x_test)  # standardized test set
n = range(1, 11, 3)  # create 1, 3, 7, 10 neighbours
knn_cm = []

for neighbours in n:
    knn = KNeighborsClassifier(n_neighbors=neighbours)  # create model
    knn.fit(x_t, y_train)  # fit standardized training data to model
    knn_p = knn.predict(x_te)
    knn_cm.append(confusion_matrix(y_test,knn_p))
    print("K-NN (" + str(neighbours) +") Classification Report:\n", classification_report(y_test, knn_p))


# Metrics and Plots #

models = ["LinearSVC", "SGDClassifier", "SVC"]
accuracy_scores = [accuracy_score(y_test, lsvc_p), accuracy_score(y_test, sgdc_p), accuracy_score(ytest, svc_p)]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracy_scores, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("Accuracy Score")
plt.title("Model Accuracy Comparison")
plt.ylim(0.7, 0.9)  # Accuracy ranges from 0 to 1
plt.show()
# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, cm, model in zip(axes, [lsvc_cm, sgdc_cm, svc_cm],
                         ["LinearSVC", "SGDClassifier", "SVC"]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title(f"{model} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, cm, model in zip(axes, [knn_cm[1], knn_cm[2], knn_cm[3]],
                         ["K-NN (4 Neighbours)", "K-NN (7 Neighbours)", "K-NN (10 Neighbours)"]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title(f"{model} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()