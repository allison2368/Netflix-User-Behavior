"""
churn_model.py — All-in-one Netflix Churn Prediction Training Script

Just run: python churn_model.py
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

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
    """Load data from BigQuery"""
    client = bigquery.Client(project=PROJECT_ID)
    job = client.query(SQL_QUERY)
    df = job.result().to_dataframe()
    df.fillna(0, inplace=True)
    return df


# customer segmentation using K-Means

def create_segments(df):
    """Create customer segments using K-Means clustering"""
    df = df.copy()
    X_rfe = df[RFE_COLUMNS].copy()
    
    scaler_rfe = StandardScaler()
    X_scaled = scaler_rfe.fit_transform(X_rfe)
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    df['segment'] = kmeans.fit_predict(X_scaled)
    
    segment_profile = df.groupby('segment')[RFE_COLUMNS].mean()
    
    return df, scaler_rfe, kmeans, segment_profile


# feature engineering and target creation

def prepare_features(df):
    """Prepare features and target"""
    df = df.copy()
    
    # Create target
    df['churn_risk'] = (df['watch_decline_ratio'] < CHURN_THRESHOLD).astype(int)
    
    # Drop columns
    cols_to_drop = ['user_id', 'watch_decline_ratio', 'watch_last_7d', 
                    'total_watch_minutes', 'is_active', 'churn_risk']
    
    X = df.drop(columns=cols_to_drop)
    y = df['churn_risk']
    
    # Convert to numeric and handle NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.astype(float)
    
    return X, y, X.columns.tolist()


# model training and evaluation

def train_model(df):
    """Train Random Forest model"""
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE
    smote = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=-1
    )
    rf_model.fit(X_train_resampled, y_train_resampled)
    
    # Predictions
    y_probs = rf_model.predict_proba(X_test_scaled)[:, 0]
    y_pred = [0 if p > PREDICTION_THRESHOLD else 1 for p in y_probs]
    
    # Metrics
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': rf_model,
        'scaler': scaler,
        'feature_cols': feature_names,
        'X_test_scaled': X_test_scaled,
        'X_test': X_test,
        'y_test': y_test,
        'predictions': y_pred,
        'probabilities': y_probs,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': feature_importance,
            'threshold': PREDICTION_THRESHOLD
        }
    }


# save model artifacts

def save_artifacts(result, scaler_rfe, kmeans, segment_profile):
    """Save all model artifacts"""
    
    if USE_GCS:
        # Save to Google Cloud Storage
        from google.cloud import storage
        import io
        
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        print(f"Saving to GCS: gs://{BUCKET_NAME}/model_outputs/")
        
        # Save pickle files
        for filename, obj in [
            ('rf_model.pkl', result['model']),
            ('scaler.pkl', result['scaler']),
            ('rfe_scaler.pkl', scaler_rfe),
            ('kmeans.pkl', kmeans),
            ('feature_names.pkl', result['feature_cols']),
            ('metrics.pkl', result['metrics'])
        ]:
            blob = bucket.blob(f'model_outputs/{filename}')
            blob.upload_from_string(pickle.dumps(obj))
            print(f"  ✓ {filename}")
        
        # Save CSVs
        for filename, data in [
            ('segment_profile.csv', segment_profile),
            ('feature_importance.csv', result['metrics']['feature_importance'])
        ]:
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=(filename == 'segment_profile.csv'))
            blob = bucket.blob(f'model_outputs/{filename}')
            blob.upload_from_string(csv_buffer.getvalue())
            print(f"  ✓ {filename}")
        
        # Save config
        config = {
            'PROJECT_ID': PROJECT_ID,
            'N_CLUSTERS': N_CLUSTERS,
            'CHURN_THRESHOLD': CHURN_THRESHOLD,
            'PREDICTION_THRESHOLD': PREDICTION_THRESHOLD
        }
        blob = bucket.blob('model_outputs/config.pkl')
        blob.upload_from_string(pickle.dumps(config))
        print(f"  ✓ config.pkl")
        
        print(f"\n✓ Saved to: gs://{BUCKET_NAME}/model_outputs/")
        
    else:
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
    print(f"\n  Test Set Distribution:")
    print(f"    No Churn (0): {test_churn_counts.get(0, 0):,}")
    print(f"    Churn Risk (1): {test_churn_counts.get(1, 0):,}")
    
    print("\nTop 10 Most Important Features:")
    for idx, row in result['metrics']['feature_importance'].head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:40s} {row['importance']:.4f}")
    
    # Step 5: Save everything
    print("\n[5/5] Saving model artifacts...")
    if USE_GCS:
        print(f"  Storage: Google Cloud Storage (gs://{BUCKET_NAME})")
    else:
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
