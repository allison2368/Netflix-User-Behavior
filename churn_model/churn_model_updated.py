"""
churn_model_updated.py — All-in-one Netflix Churn Prediction Training Script

Just run: python churn_model_updated.py
"""
import os
import pickle
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE  # pylint: disable=import-error

# config
# BigQuery settings
PROJECT_ID = 'netflix-user-behavior'
SQL_QUERY = "SELECT * FROM `netflix-user-behavior.kaggle_cleaned.churn_features`"

# RFE columns for customer segmentation
RFE_COLUMNS = [
    "days_since_last_watch",
    "total_sessions",
    "avg_completion_rate"
]

# Model parameters
N_CLUSTERS = 4
CHURN_THRESHOLD = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOTE_RATIO = 0.2
PREDICTION_THRESHOLD = 0.3

# Random Forest parameters
RF_N_ESTIMATORS = 1000
RF_MAX_DEPTH = None
RF_CLASS_WEIGHT = 'balanced'

# Storage
MODEL_SAVE_DIR = './model_outputs'
USE_GCS = False  # Set to True to save to Google Cloud Storage
BUCKET_NAME = 'netflix-churn-models'


# load data from BigQuery

def load_data():
    """
    Load data from BigQuery and return it as a cleaned pandas DataFrame.

    This function executes a predefined SQL query against the configured
    BigQuery project, retrieves the results as a pandas DataFrame, and
    replaces any missing values with zeros.

    Returns:
        pandas.DataFrame: The query results from BigQuery with missing
        values filled with 0.
    """
    client = bigquery.Client(project=PROJECT_ID)
    job = client.query(SQL_QUERY)
    df = job.result().to_dataframe()
    df.fillna(0, inplace=True)
    return df


# customer segmentation using K-Means

def create_segments(df):
    """
    Generate customer segments using K-Means clustering.

    This function scales selected behavioral features and applies
    K-Means clustering to group users into segments. The resulting
    cluster labels are added to the dataset.
    Args:
        df (pandas.DataFrame): Input DataFrame containing the user
            features specified in `RFE_COLUMNS`.

    Returns:
        tuple:
            pandas.DataFrame (df): DataFrame with an added `segment`
            column representing the assigned cluster for each user.
            sklearn.preprocessing.StandardScaler (scaler_rfe): Fitted scaler used
            to normalize the segmentation features.
            sklearn.cluster.KMeans (kmeans): Trained K-Means clustering model.
            pandas.DataFrame (segment_profile): Segment profile table showing the mean
            feature values for each cluster.
    """
    df = df.copy()

    x_rfe = df[RFE_COLUMNS].copy()
    scaler_rfe = StandardScaler()
    x_scaled = scaler_rfe.fit_transform(x_rfe)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    df['segment'] = kmeans.fit_predict(x_scaled)

    segment_profile = df.groupby('segment')[RFE_COLUMNS].mean()

    return df, scaler_rfe, kmeans, segment_profile


# feature engineering and target creation

def prepare_features(df):
    """
    Create the churn target and clean feature matrix for modeling.

    Generates a binary `churn_risk` label based on `watch_decline_ratio`,
    removes non-feature columns, converts all remaining features to numeric,
    and handles NaN or infinite values.

    Args:
        df (pandas.DataFrame): Input dataset containing user behavior features.

    Returns:
        tuple:
            pandas.DataFrame (x): Clean feature matrix `X`.
            pandas.Series (y): Target variable `y` representing churn risk.
            list[str] (feature_names): Names of the feature columns used in the model.
    """
    df = df.copy()

    # Create target
    df['churn_risk'] = (df['watch_decline_ratio'] < CHURN_THRESHOLD).astype(int)

    # Drop columns
    cols_to_drop = ['user_id', 'watch_decline_ratio', 'watch_last_7d',
                    'total_watch_minutes', 'is_active', 'churn_risk']

    x = df.drop(columns=cols_to_drop)
    y = df['churn_risk']

    # Convert to numeric and handle NaN
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.fillna(0)
    x = x.replace([np.inf, -np.inf], 0)
    x = x.astype(float)

    return x, y, x.columns.tolist()


# model training and evaluation
# pylint: disable=too-many-locals
def train_model(df):
    """
    Train a Random Forest churn model and return predictions,
    evaluation metrics, and model artifacts.

    Args:
        df (pandas.DataFrame): Input dataset containing user features.

    Returns:
        dict: Trained model, scaler, test data, predictions, probabilities,
        feature importance, and evaluation metrics.
    """
    # Prepare features
    x, y, feature_names = prepare_features(df)

    # Only stratify if both classes have enough samples
    if y.nunique() > 1 and min(y.value_counts()) > 1:
        stratify = y
    else:
        stratify = None
    x_train, x_test, y_train, y_test = train_test_split(x, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify)

    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # SMOTE
    # SMOTE only if enough minority samples exist
    # (safe fallback if dataset too small)
    class_counts = y_train.value_counts()

    if len(class_counts) > 1 and class_counts.min() > 1:
        try:
            smote = SMOTE(
                sampling_strategy=SMOTE_RATIO,
                random_state=RANDOM_STATE
            )
            x_train_resampled, y_train_resampled = smote.fit_resample(
                x_train_scaled,
                y_train
            )
        except ValueError:
            # fallback if SMOTE fails
            x_train_resampled, y_train_resampled = x_train_scaled, y_train
    else:
        x_train_resampled, y_train_resampled = x_train_scaled, y_train
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=-1
    )
    rf_model.fit(x_train_resampled, y_train_resampled)
    # Predictions
    y_probs = rf_model.predict_proba(x_test_scaled)[:, 0]
    y_pred = [0 if p > PREDICTION_THRESHOLD else 1 for p in y_probs]

    # Metrics
    feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

    # compute accuracy always
    accuracy = accuracy_score(y_test, y_pred)

# only compute recall / f1 if more than one class exists
    if y_test.nunique() > 1:
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
    else:
        recall = 0.0
        f1 = 0.0
        report = {}

    return {
    "model": rf_model,
    "scaler": scaler,
    "feature_cols": feature_names,
    "X_test_scaled": x_test_scaled,
    "X_test": x_test,
    "y_test": y_test,
    "predictions": y_pred,
    "probabilities": y_probs,
    "metrics": {
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "feature_importance": feature_importance,
        "threshold": PREDICTION_THRESHOLD
    }
}


# save model artifacts

def save_artifacts(result, scaler_rfe, kmeans, segment_profile):
    """
    Save trained model artifacts, clustering objects, metrics, and segment profiles
    to a local directory.

    Args:
        result (dict): Output dictionary from `train_model` containing the model,
            scaler, feature names, metrics, and predictions.
        scaler_rfe (StandardScaler): Fitted scaler used for RFE clustering features.
        kmeans (KMeans): Trained K-Means segmentation model.
        segment_profile (pandas.DataFrame): Aggregated feature summary for each segment.
    """
    # Save locally
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Save pickle files
    with open(os.path.join(MODEL_SAVE_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(result['model'], f)
    with open(os.path.join(MODEL_SAVE_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(result['scaler'], f)
    with open(os.path.join(MODEL_SAVE_DIR, 'rfe_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_rfe, f)
    with open(os.path.join(MODEL_SAVE_DIR, 'kmeans.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODEL_SAVE_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(result['feature_cols'], f)
    with open(os.path.join(MODEL_SAVE_DIR, 'metrics.pkl'), 'wb') as f:
        pickle.dump(result['metrics'], f)

    # Save CSVs
    segment_profile.to_csv(os.path.join(MODEL_SAVE_DIR, 'segment_profile.csv'))
    result['metrics']['feature_importance'].to_csv(
        os.path.join(MODEL_SAVE_DIR, 'feature_importance.csv'),
        index=False
    )

    # Save config
    config = {
        'PROJECT_ID': PROJECT_ID,
        'N_CLUSTERS': N_CLUSTERS,
        'CHURN_THRESHOLD': CHURN_THRESHOLD,
        'PREDICTION_THRESHOLD': PREDICTION_THRESHOLD
    }
    with open(os.path.join(MODEL_SAVE_DIR, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print(f"✓ Saved to: {MODEL_SAVE_DIR}/")

def load_artifacts(path: str) -> dict:
    """
    Load all pickle artifacts from a directory.

    Args:
        path (str): Directory containing saved model artifacts.

    Returns:
        dict: Dictionary where keys are artifact names and values are loaded objects.
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Artifact directory not found: {path}")

    artifacts = {}

    for file_name in os.listdir(path):
        if file_name.endswith(".pkl"):
            artifact_name = file_name.replace(".pkl", "")
            file_path = os.path.join(path, file_name)

            with open(file_path, "rb") as file:
                artifacts[artifact_name] = pickle.load(file)

    if not artifacts:
        raise ValueError(f"No .pkl artifacts found in {path}")

    return artifacts

# main function to run everything
def main():
    """Run the complete training pipeline"""

    # Step 1: Load data
    print("\n[1/5] Loading data from BigQuery...")
    print(f"  Project: {PROJECT_ID}")
    df = load_data()
    print(f"✓ Loaded {len(df):,} records with {df.shape[1]} features")

    # Step 2: Create segments
    print("\n[2/5] Creating customer segments...")
    df, scaler_rfe, kmeans, segment_profile = create_segments(df)
    print(f"✓ Created {N_CLUSTERS} customer segments")
    print("\nSegment Profiles:")
    print(segment_profile.round(2))

    # Step 3: Train model
    print("\n[3/5] Training Random Forest model...")
    print(f"  - Trees: {RF_N_ESTIMATORS}")
    print(f"  - SMOTE ratio: {SMOTE_RATIO}")
    print(f"  - Prediction threshold: {PREDICTION_THRESHOLD}")
    result = train_model(df)
    print("✓ Model training complete")

    # Step 4: Display metrics
    print("\n[4/5] Model Performance:")
    print(f"  Test Set Size: {len(result['y_test']):,} samples")
    print(f"  Accuracy: {result['metrics']['accuracy']:.3f}")
    print(f"  Recall:   {result['metrics']['recall']:.3f}")
    print(f"  F1 Score: {result['metrics']['f1_score']:.3f}")

    test_churn_counts = pd.Series(result['y_test']).value_counts()
    print("\n  Test Set Distribution:")
    print(f"    No Churn (0): {test_churn_counts.get(0, 0):,}")
    print(f"    Churn Risk (1): {test_churn_counts.get(1, 0):,}")

    print("\nTop 10 Most Important Features:")
    for idx, row in result['metrics']['feature_importance'].head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:40s} {row['importance']:.4f}")

    # Step 5: Save everything
    print("\n[5/5] Saving model artifacts...")
    print(f"  Storage: Local ({MODEL_SAVE_DIR})")

    save_artifacts(result, scaler_rfe, kmeans, segment_profile)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

    if USE_GCS:
        print(f"\nModel saved to: gs://{BUCKET_NAME}/model_outputs/")
        print("\nNext steps:")
        print("  1. Update dashboard.py to set USE_GCS = True")
        print("  2. Run: streamlit run dashboard.py")
    else:
        print(f"\nModel saved to: {MODEL_SAVE_DIR}/")



if __name__ == "__main__":
    main()
