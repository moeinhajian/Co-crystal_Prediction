# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

# %%
# Apply SMOTE to oversample the minority class
smote = SMOTE(sampling_strategy=1, random_state=5)  # Adjust the sampling strategy as needed
X_resampled, y_resampled = smote.fit_resample(X, y)
#Split the data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, stratify=y_resampled, shuffle=True, test_size=0.2, random_state=5)
# Create and compile the model
finalmodel = create_model()
# Define early stopping and model checkpoint callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# Train the model with validation data and callbacks
finalmodel.fit(X_train, y_train, epochs=50, verbose=0, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = (finalmodel.predict(xcosmo) > 0.6).astype(int)
DNN_labels = y_pred.reshape(-1)
DNN_probs = finalmodel.predict(xcosmo) 
DNN_probs = DNN_probs.reshape(-1)
dnntest_prob = finalmodel.predict(xtest).reshape(-1)

# %% [markdown]
# ## Random Forest

# %%
# Apply SMOTE to oversample the minority class
smote = SMOTE(sampling_strategy=1, random_state=5)  # Adjust the sampling strategy as needed
# Create and compile the model
finalmodel = RandomForestClassifier(bootstrap= False, max_depth= None, max_features= 15, min_samples_leaf= 10, min_samples_split= 2, n_estimators= 30, random_state=5)
X_first, y_first = smote.fit_resample(X, y)
finalmodel.fit(X_first, y_first)

# Evaluate the model on the test set
y_pred = finalmodel.predict(xcosmo)
rf_labels = y_pred.reshape(-1)
rf_probs = finalmodel.predict_proba(xcosmo)[:, 1]
rftest_prob = finalmodel.predict_proba(xtest)[:, 1]

# %% [markdown]
# ## SVM

# %%
# Apply SMOTE to oversample the minority class
smote = SMOTE(sampling_strategy=1, random_state=5)  # Adjust the sampling strategy as needed

# Create and compile the model
finalmodel = SVC(kernel='rbf', C=10, gamma=0.001, random_state=5, probability=True)
X_first, y_first = smote.fit_resample(X, y)
finalmodel.fit(X_first, y_first)
# Evaluate the model on the test set
y_pred = (finalmodel.predict(xcosmo) > 0.5).astype(int)
svm_labels = y_pred.reshape(-1)
svm_probs = finalmodel.predict_proba(xcosmo)[:, 1]
svmtest_prob = finalmodel.predict_proba(xtest)[:, 1]

# %% [markdown]
# # Spearman Ranking

# %%
dfcosmo= pd.DataFrame({'Coformer': maindataset['Coformer'][0:97].values, 'cosmo_Hex': -hexcosmo, 'DNN_probs': DNN_probs,
                        'DNN_label': DNN_labels, 'SVM_probs': svm_probs, 'SVM_label': svm_labels, 'RF_probs': rf_probs, 'RF_label': rf_labels,
                         'nHBAcc': descriptors['nHBAcc'][0:97], 'nHBDon': descriptors['nHBDon'][0:97], 'nC': descriptors['nC'][0:97] })
# Sort by COSMO-RS Hex
dfcosmo = dfcosmo.sort_values(by=['cosmo_Hex'], ascending=False)
dfcosmo['cosmo_rank'] = np.arange(1, len(dfcosmo) + 1)
# Sort by SVM probabilities 
dfcosmo = dfcosmo.sort_values(by=['nHBDon'], ascending=False)
dfcosmo = dfcosmo.sort_values(by=['SVM_probs', 'nHBDon','nHBAcc', 'nC'], ascending=False)
dfcosmo['SVM_rank'] = np.arange(1, len(dfcosmo) + 1)
# Sort by RF probabilities
dfcosmo = dfcosmo.sort_values(by=['nHBDon'], ascending=False)
dfcosmo = dfcosmo.sort_values(by=['RF_probs', 'nHBDon', 'nHBAcc', 'nC'], ascending=False)
dfcosmo['RF_rank'] = np.arange(1, len(dfcosmo) + 1)
# Sort by DNN probabilities
dfcosmo = dfcosmo.sort_values(by=['nHBDon'], ascending=False)
dfcosmo = dfcosmo.sort_values(by=['DNN_probs', 'nHBDon', 'nHBAcc', 'nC'], ascending=False)
dfcosmo['DNN_rank'] = np.arange(1, len(dfcosmo) + 1)
# dfcosmo.to_excel('COSMOresults.xlsx')


# %%
# Calculate Spearman's correlation
spearman_corr, _ = spearmanr(dfcosmo['RF_rank'], dfcosmo['cosmo_rank'])

print("Random Forest (RF):")
# Check for monotonicity
if spearman_corr > 0:
    print("The two columns have a positive monotonic relationship.")
elif spearman_corr < 0:
    print("The two columns have a negative monotonic relationship.")
else:
    print("The two columns are not monotonous (no monotonic relationship).")

print("Spearman's Correlation Coefficient For Random Forest:", spearman_corr)

# Calculate Spearman's correlation
spearman_corr, _ = spearmanr(dfcosmo['DNN_rank'], dfcosmo['cosmo_rank'])

print("Deep Neural Network (DNN):")
# Check for monotonicity
if spearman_corr > 0:
    print("The two columns have a positive monotonic relationship.")
elif spearman_corr < 0:
    print("The two columns have a negative monotonic relationship.")
else:
    print("The two columns are not monotonous (no monotonic relationship).")

print("Spearman's Correlation Coefficient For DNN:", spearman_corr)


# Calculate Spearman's correlation
spearman_corr, _ = spearmanr(dfcosmo['SVM_rank'], dfcosmo['cosmo_rank'])

print("Support Vector Machine (SVM):")
# Check for monotonicity
if spearman_corr > 0:
    print("The two columns have a positive monotonic relationship.")
elif spearman_corr < 0:
    print("The two columns have a negative monotonic relationship.")
else:
    print("The two columns are not monotonous (no monotonic relationship).")

print("Spearman's Correlation Coefficient For SVM:", spearman_corr)

# %%
# Scatter plot
plt.figure(figsize=(7, 7))
df= pd.DataFrame({'Coformer': maindataset['Coformer'][5:34].values, 'cosmo_Hex': -hextest, 'EXP-label': ytest, 'DNNtest_probs':dnntest_prob, 'SVMtest_probs': svmtest_prob, 'RFtest_probs': rftest_prob,
                  'nHBAcc': descriptors['nHBAcc'][5:34], 'nHBDon': descriptors['nHBDon'][5:34], 'nC': descriptors['nC'][5:34]})
# Sort by COSMO-RS Hex
df= df.sort_values(by=['cosmo_Hex'], ascending=False)
df['cosmo_rank'] = np.arange(1, len(df) + 1)
# Sort by SVM probabilities 
df = df.sort_values(by=['SVMtest_probs', 'nHBDon', 'nHBAcc', 'nC'], ascending=False)
df['SVM_rank'] = np.arange(1, len(df) + 1)
# Sort by RF probabilities
df = df.sort_values(by=['cosmo_Hex'], ascending=False)
df = df.sort_values(by=['RFtest_probs', 'nHBDon', 'nHBAcc', 'nC'], ascending=False)
df['RF_rank'] = np.arange(1, len(df) + 1)
# Sort by DNN probabilities
df = df.sort_values(by=['cosmo_Hex'], ascending=False)
df = df.sort_values(by=['DNNtest_probs', 'nHBDon', 'nHBAcc', 'nC'], ascending=False)
df['DNN_rank'] = np.arange(1, len(df) + 1)

# Different markers for each column
markers = ['o', 'x', 10]  # Circle, Square, Triangle
lb = ['RF', 'SVM', 'DNN']
# Colors based on binary column
colors = ['green' if val == 1 else 'red' for val in df['EXP-label']]

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# Plotting each column against the target column with different markers and colors
for column, marker, lb in zip(['RF_rank', 'SVM_rank', 'DNN_rank'], markers, lb):
    plt.scatter(df['cosmo_rank'], df[column], label=lb, marker=marker, c=colors)
plt.xlabel('COSMO-RS ranking')
plt.ylabel('ML models ranking')
# plt.title('probabilities of 29 independent data in models vs COSMO-RS ranking')
# Create custom handles for the legend
from matplotlib.lines import Line2D
legend_handles = [Line2D([0], [0], color='black', marker='o', linestyle='', label='RF'),
                  Line2D([0], [0], color='black', marker='x', linestyle='', label='SVM'),
                  Line2D([0], [0], color='black', marker=6, linestyle='', label='DNN')]

# Add the legend to the plot
plt.legend(handles=legend_handles, loc='lower right')

# Display the plot
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('ranking.png', dpi=1200,bbox_inches='tight')


