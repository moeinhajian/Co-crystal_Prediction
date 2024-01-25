# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

# Set random seed for NumPy
np.random.seed(5)

# %%
# Read datasets and scriptors
maindataset = pd.read_excel('Maindatasets.xlsx', sheet_name='main')
descriptors = pd.read_excel('Maindatasets.xlsx', sheet_name='descriptors')
descriptors = descriptors.iloc[: , 2:]
dfmain_2  = descriptors.astype('float')

# %%
# Scale data before applying PCA
std_scaler = StandardScaler()
scaled_main_2 = std_scaler.fit_transform(dfmain_2)

# PCA Use fit and transform method
pca = PCA(n_components=60)
pcamaindata_2 = pca.fit_transform(scaled_main_2)

# %%
# Separate datasets
# independent dataset
xcosmo = pcamaindata_2[0:97]
ycosmo = maindataset['Cosmo-results'][0:97].values
hexcosmo = maindataset['Hex'][0:97].values

xtest = pcamaindata_2[5:34]
ytest = maindataset['expresult'][5:34].values
hextest = maindataset['Hex'][5:34].values

X = pcamaindata_2[97:199]
y = maindataset['expresult'][97:199].values


# %% [markdown]
# ## SVM

# %%
#Split the data into training, validation, and testing sets
X_resampled, X_finaltest, y_resampled, y_finaltest = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.25, random_state=46)
# Apply SMOTE to oversample the minority class
smote = SMOTE(sampling_strategy=1, random_state=5)  # Adjust the sampling strategy as needed
X_first, y_first = smote.fit_resample(X_resampled, y_resampled)

# Initialize a Random Forest classifier
svm_classifier = SVC(kernel='rbf', C=10, gamma=0.001, random_state=5, probability=True)
# Initialize StratifiedKFold with the desired number of folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
# Store accuracy scores for each fold
fold_accuracies = []
fold_bacc = []
print('Results of 5fold cross validation:')
# Iterate through each fold
for fold, (train_index, test_index) in enumerate(skf.split(X_first,y_first)):
    X_set, X_test = X_first[train_index], X_first[test_index]
    y_set, y_test = y_first[train_index], y_first[test_index]

    # Train the model 
    history = svm_classifier.fit(X_set, y_set)
    # Evaluate the model on the test set
    y_pred = (svm_classifier.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    print(f'Fold {fold + 1} Accuracy: {accuracy:.2f}')
    confusion_mtx = confusion_matrix(y_test, y_pred)
    sensitivity = confusion_mtx[1,1] / (confusion_mtx[1,1] + confusion_mtx[1,0])
    specificity = confusion_mtx[0,0] / (confusion_mtx[0,1] + confusion_mtx[0,0])
    bacc = (sensitivity + specificity)/2
    print(f'Sensitivity: {sensitivity:.2f} ,  Specificity:  {specificity:.2f} ,  BACC:  {bacc:.2f} ')
    fold_bacc.append(bacc)
    # print(classification_report(ytest, y_pred))
    print(confusion_mtx )

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(fold_accuracies)
mean_bacc = np.mean(fold_bacc)
print(f'Mean Cross-Validation Accuracy: {mean_accuracy:.2f}, and Mean BACC: {mean_bacc:.2f}')

# Create and compile the model
finalmodel = SVC(kernel='rbf', C=10, gamma=0.001, random_state=5, probability=True)
# Train the model with all training dataset
finalmodel.fit(X_first, y_first)
# Evaluate the model on the test set
y_pred = (finalmodel.predict(X_finaltest) > 0.5).astype(int)
accuracy = accuracy_score(y_finaltest, y_pred)
print('Results of test dataset:')
print(f"Accuracy over test dataset: {accuracy:.2f}")
# Calculate the confusion matrix
confusion_mtx = confusion_matrix(y_finaltest, y_pred)
sensitivity = confusion_mtx[1,1] / (confusion_mtx[1,1] + confusion_mtx[1,0])
specificity = confusion_mtx[0,0] / (confusion_mtx[0,1] + confusion_mtx[0,0])
bacc = (sensitivity + specificity)/2
print(f'Sensitivity: {sensitivity:.2f} ,  Specificity:  {specificity:.2f} ,  BACC:  {bacc:.2f} ')
print(confusion_mtx)

X_first, y_first = smote.fit_resample(X, y)
finalmodel.fit(X_first, y_first)
# Evaluate the model on the 29 experimental data points (Included in the ind. test set)
y_pred = (finalmodel.predict(xtest) > 0.5).astype(int)
accuracy = accuracy_score(ytest, y_pred)
print('Results of 29 experimental data points (Included in the ind. test set):')
print(f"Accuracy experiment 29: {accuracy:.2f}")
# print(classification_report(ytest, y_pred))
confusion_mtx = confusion_matrix(ytest, y_pred)
sensitivity = confusion_mtx[1,1] / (confusion_mtx[1,1] + confusion_mtx[1,0])
specificity = confusion_mtx[0,0] / (confusion_mtx[0,1] + confusion_mtx[0,0])
bacc = (sensitivity + specificity)/2
print(f'Sensitivity: {sensitivity:.2f} ,  Specificity:  {specificity:.2f} ,  BACC:  {bacc:.2f} ')
print(confusion_matrix(ytest, y_pred))



