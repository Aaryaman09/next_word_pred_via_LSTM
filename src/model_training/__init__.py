from src.logger import get_logger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict
import numpy as np
import os

logger = get_logger()

class ModelTraining:
    def __init__(self, total_words:int, config:Dict):
        logger.info("#"*30)
        logger.info("Model Training started.")
        logger.info("#"*30)
        self.total_words=total_words
        self.continue_execution = True
        self.config = config

    def build_model(self)->Sequential:
        try:
            logger.info(f"Building the model")

            model = Sequential()
            model.add(Embedding(self.total_words, 100))
            model.add(LSTM(150, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dense(self.total_words, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            logger.info(f'Model Build successfully. Model summary : {model.summary()}')
            return model
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)
            return None

    def fetch_processed_inputs(self, path):
        try:
            logger.info(f"Fetching the processed file at path : {path}.")
            numpy_array = np.load(path)

            logger.info('File fetched successfully.')
            return numpy_array
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)
            return []

    def training_the_model(self, model:Sequential)->None:
        try:
            logger.info(f"Fitting the model on inputs.")

            early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True, verbose=1, patience=5)

            history = model.fit(
                self.fetch_processed_inputs(path=self.config.get('processed_dataset_paths').get('root_dir_name').get('train_data').get('x_train')), 
                self.fetch_processed_inputs(path=self.config.get('processed_dataset_paths').get('root_dir_name').get('train_data').get('y_train')),
                epochs=75, 
                validation_data=(self.fetch_processed_inputs(path=self.config.get('processed_dataset_paths').get('root_dir_name').get('test_data').get('x_test')), 
                                 self.fetch_processed_inputs(path=self.config.get('processed_dataset_paths').get('root_dir_name').get('test_data').get('y_test'))), 
                verbose=1, 
                # callbacks=[early_stopping]
            )

            logger.info('training completed successfully.')
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)

    def save_model(self, model):
        try:
            model_path = os.path.join(
                self.config.get('artifacts_path').get('root_dir_name'),
                self.config.get('artifacts_path').get('model_file_name'))
            
            logger.info(f"Saving the model at directory : {model_path}.")
            model.save(model_path)
            logger.info('Model saved successfully.')
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)

    def main(self):
        model = self.build_model()

        if self.continue_execution:
            self.training_the_model(model)

        if self.continue_execution:
            self.save_model(model=model)

        logger.info("#"*30)
        logger.info("Model Training completed.")
        logger.info("#"*30)
        