from config.feast_config import healthcare_feast_store
import pandas as pd
from datetime import datetime, timedelta
import asyncio

async def main():
    # Initialize and setup
    store = healthcare_feast_store

    # 1. Setup feature store (run once)
    store.create_feature_repo()
    store.apply_feature_definitions()

    # 2. Get features for real-time inference
    patient_ids = [1, 2, 3]
    features_df = await store.get_online_features_for_inference(
        patient_ids=patient_ids,
        feature_service_name="cardiovascular_risk_prediction"
    )
    print("Online features for inference:")
    print(features_df.head())

    # 3. Get historical features for training
    entity_df = pd.DataFrame({
        'patient_id': [1, 2, 3, 4, 5],
        'event_timestamp': [datetime.now() - timedelta(days=i) for i in range(5)]
    })

    training_df = await store.get_historical_features_for_training(
        entity_df=entity_df,
        feature_service_name="diabetes_risk_prediction"
    )
    print("Historical features for training:")
    print(training_df.head())

    # 4. Materialize features for serving
    store.materialize_features(
        start_date="2024-01-01T00:00:00",
        end_date="2024-12-31T23:59:59"
    )

if __name__ == "__main__":
    asyncio.run(main())