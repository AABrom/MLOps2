import pandas as pd
import logging
import xgboost as xgb
from pathlib import Path

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = xgb.XGBClassifier()
model_path = Path('./models/xgb.bin')
model.load_model(model_path)

logger.info('Pretrained model imported successfully...')

# Make prediction
def make_pred(dt, source_info="kafka"):

    # Make submission dataframe
    submission = pd.DataFrame({
        'prediction': model.predict(dt)
    })
    logger.info(f'Prediction complete for for data from {source_info}')

    # Return proba for positive class
    return submission
