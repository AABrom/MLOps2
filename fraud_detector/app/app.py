import os
import sys
import pandas as pd
import logging
import json
from pathlib import Path
from confluent_kafka import Consumer, Producer, KafkaError

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred

log_dir = Path('./logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'service.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set kafka configuration file
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scoring")


class ProcessingService:
    def __init__(self):
            logger.info('Initializing ProcessingService...')
            self.consumer_config = {
                'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
                'group.id': 'ml-scorer',
                'auto.offset.reset': 'earliest'
            }
            self.producer_config = {
                'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS
                }
            self.consumer = Consumer(self.consumer_config)
            self.consumer.subscribe([TRANSACTIONS_TOPIC])
            self.producer = Producer(self.producer_config)
            
            # Загрузка данных для препроцессинга
            self.train = load_train_data()
        
            logger.info('Service initialized')

    def process_messages(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            try:
                # Десериализация JSON
                data = json.loads(msg.value().decode('utf-8'))

                # Извлекаем ID и данные
                transaction_id = data['transaction_id']
                input_df = pd.DataFrame([data['data']])

                # Препроцессинг и предсказание
                processed_df = run_preproc(input_df)
                submission = make_pred(processed_df)

                # Добавляем ID в результат
                submission['transaction_id'] = transaction_id

                # Отправка результата в топик scoring
                self.producer.produce(
                    'scoring',
                    value=submission.to_json(orient='records')
                )
                self.producer.flush()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
    

if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')