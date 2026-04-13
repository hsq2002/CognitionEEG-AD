import scipy.io
import numpy as np
import pandas as pd
import os
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


# load participants and keep only AD (A) and Control (C) subjects, drop FTD
df = pd.read_csv('participants.tsv', sep='\t')
df = df[df['Group'].isin(['A', 'C'])]
df['label'] = df['Group'].map({'A': 1, 'C': 0})


# load a subject's .mat file and split the continuous eeg signal into fixed 4 second epochs
def get_epochs(subject_id, data_dir='patient_data', fs=256, epoch_length=4.0):
    path = os.path.join(data_dir, f'{subject_id}_task-eyesclosed.mat')
    mat = scipy.io.loadmat(path)
    eeg = mat['eeg']  # shape: (19 channels, n_samples)

    samples_per_epoch = int(epoch_length * fs)  # 4 seconds * 256 hz = 1024 samples per epoch
    n_epochs = eeg.shape[1] // samples_per_epoch

    # slice the signal into non-overlapping epochs
    epochs = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epochs.append(eeg[:, start:end])

    return np.array(epochs)  # shape: (n_epochs, 19, 1024)


# extract 133 features from a single epoch:
# 95 band power features (19 channels x 5 bands)
# 19 theta/alpha ratio features (one per channel)
# 19 delta/alpha ratio features (one per channel)
def extract_features(epoch, fs=256):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 45)
    }

    # compute power spectral density for each channel across each band
    band_powers = {b: [] for b in bands}
    for ch in epoch:  # loop over 19 channels
        freqs, psd = welch(ch, fs=fs)  # welch's method gives a stable psd estimate
        for band, (low, high) in bands.items():
            idx = np.where((freqs >= low) & (freqs <= high))
            band_powers[band].append(np.mean(psd[idx]))

    # flatten band powers into a single feature list (95 features)
    features = []
    for band in bands:
        features.extend(band_powers[band])

    # add theta/alpha ratio per channel — elevated in AD subjects
    for i in range(19):
        features.append(band_powers['theta'][i] / (band_powers['alpha'][i] + 1e-8))  # 1e-8 avoids division by zero

    # add delta/alpha ratio per channel — another well known AD biomarker
    for i in range(19):
        features.append(band_powers['delta'][i] / (band_powers['alpha'][i] + 1e-8))

    return features  # 133 features total


# build the feature matrix by loading each subject, extracting features per epoch, then averaging across epochs
X = []
y = []

for _, row in df.iterrows():  # loop over each subject in the participants file
    try:
        epochs = get_epochs(row['participant_id'])
        epoch_features = [extract_features(e) for e in epochs]  # extract features from every epoch
        subject_features = np.mean(epoch_features, axis=0)  # average across epochs to get one row per subject
        X.append(subject_features)
        y.append(row['label'])
    except Exception as e:
        print(f"Skipping {row['participant_id']}: {e}")

X = np.array(X)  # shape: (65, 133)
y = np.array(y)


# normalize features — important for consistent model performance across different signal scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 70/30 split — more test data gives more stable evaluation metrics with a small dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42  # stratify ensures balanced AD/CN in each split
)


# randomized search tries 50 random combinations of hyperparameters
# and picks the one with the best cross validated auc score
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),  # 10 fold cross validation
    scoring='roc_auc',
    n_jobs=-1,  # use all available cpu cores
    random_state=42,
    verbose=2
)

random_search.fit(X_train, y_train)
print(f"\nBest params: {random_search.best_params_}")
print(f"Best CV AUC: {random_search.best_score_:.3f}")


# evaluate the best model found by random search on the held out test set
best_rf = random_search.best_estimator_

y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]  # probability scores needed for auc-roc

print("\n--- tuned model results ---")
print(classification_report(y_test, y_pred))
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.3f}")

# confusion matrix shows true vs predicted labels for each class
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'AD'])
disp.plot()
plt.title('Confusion Matrix (Tuned)')
plt.show()


# plot raw feature importances across all 133 features
importances = best_rf.feature_importances_
plt.bar(range(133), importances)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importances (Tuned)')
plt.show()