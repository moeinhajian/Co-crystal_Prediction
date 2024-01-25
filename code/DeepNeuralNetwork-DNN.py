# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

# Set random seed for TensorFlow
tf.random.set_seed(5)
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
# # Neural Network Classification

# %%
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60,)),             # Input layer with 90 features
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
        # Hidden layer with 128 neurons, ReLU activation, and L2 regularization   
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),
# Hidden layer with 128 neurons, ReLU activation, and L2 regularization   
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)),

        # Hidden layer with 64 neurons, ReLU activation, and L2 regularization
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=5)) # Output layer with sigmoid activation for binary classification
    ])
    # Compile the model with learning rate control
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust the learning rate as needed
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# %% [markdown]
# ## Five Fold Cross Validation

# %%
#Split the data into training, validation, and testing sets
X_resampled, X_finaltest, y_resampled, y_finaltest = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.25, random_state=46)
# Apply SMOTE to oversample the minority class
smote = SMOTE(sampling_strategy=1, random_state=5)  # Adjust the sampling strategy as needed
X_first, y_first = smote.fit_resample(X_resampled, y_resampled)
# Define early stopping and model checkpoint callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# Initialize StratifiedKFold with the desired number of folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
# Store accuracy scores for each fold
fold_accuracies = []
fold_bacc = []

print('Results of 5fold Cross validation:')
# Iterate through each fold
for fold, (train_index, test_index) in enumerate(skf.split(X_first,y_first)):
    X_set, X_test = X_first[train_index], X_first[test_index]
    y_set, y_test = y_first[train_index], y_first[test_index]
    X_train, X_val, y_train, y_val = train_test_split(X_set, y_set, stratify=y_set, shuffle=True, test_size=0.2, random_state=5)

    # Create and compile the model
    model = create_model()
    # Train the model with validation data and callbacks
    history = model.fit(X_train, y_train, epochs=50, verbose=0,validation_data=(X_val, y_val), callbacks=[early_stopping])
   
    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.6).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    print(f'Fold {fold + 1} Accuracy: {accuracy:.2f}')
    confusion_mtx = confusion_matrix(y_test, y_pred)
    sensitivity = confusion_mtx[1,1] / (confusion_mtx[1,1] + confusion_mtx[1,0])
    specificity = confusion_mtx[0,0] / (confusion_mtx[0,1] + confusion_mtx[0,0])
    bacc = (sensitivity + specificity)/2
    print(f'Sensitivity: {sensitivity:.2f} ,  Specificity:  {specificity:.2f} ,  BACC:  {bacc:.2f} ')
    fold_bacc.append(bacc)
    print(confusion_mtx)

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(fold_accuracies)
print(f'Mean Cross-Validation Accuracy: {mean_accuracy:.4f}')


# %% [markdown]
# ## Test dataset

# %%
#Split the data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_first, y_first, stratify=y_first, shuffle=True, test_size=0.25, random_state=5)
# Create and compile the model
finalmodel = create_model()
# Train the model with validation data and callbacks
finalmodel.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])
# Evaluate the model on the test set
y_pred = (finalmodel.predict(X_finaltest) > 0.6).astype(int)
accuracy = accuracy_score(y_finaltest, y_pred)
print('Results of test dataset:')
print(f"Accuracy over test dataset: {accuracy:.4f}")
# Calculate the confusion matrix
confusion_mtx = confusion_matrix(y_finaltest, y_pred)
sensitivity = confusion_mtx[1,1] / (confusion_mtx[1,1] + confusion_mtx[1,0])
specificity = confusion_mtx[0,0] / (confusion_mtx[0,1] + confusion_mtx[0,0])
bacc = (sensitivity + specificity)/2
print(f'Sensitivity: {sensitivity:.4f} ,  Specificity:  {specificity:.4f} ,  BACC:  {bacc:.4f} ')
print(confusion_mtx)

# %% [markdown]
# ## 29 experimental data points (Included in the ind. test set)

# %%
X_resampled, y_resampled = smote.fit_resample(X, y)
#Split the data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, stratify=y_resampled, shuffle=True, test_size=0.2, random_state=5)
# Create and compile the model
finalmodel = create_model()
# Train the model with validation data and callbacks
finalmodel.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = (finalmodel.predict(xtest) > 0.6).astype(int)
accuracy = accuracy_score(ytest, y_pred)
print('Results of 29 experimental data points (Included in the ind. test set):')
print(f"Accuracy experiment 29: {accuracy:.2f}")
confusion_mtx = confusion_matrix(ytest, y_pred)
sensitivity = confusion_mtx[1,1] / (confusion_mtx[1,1] + confusion_mtx[1,0])
specificity = confusion_mtx[0,0] / (confusion_mtx[0,1] + confusion_mtx[0,0])
bacc = (sensitivity + specificity)/2
print(f'Sensitivity: {sensitivity:.4f} ,  Specificity:  {specificity:.4f} ,  BACC:  {bacc:.4f} ')
# print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))



