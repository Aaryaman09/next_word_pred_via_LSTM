from src.logger import get_logger
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import os, pickle

logger = get_logger()

def load_model_file(path):
    try:
        logger.info(f"Loading model from : {path}.")
        model = load_model(path)

        logger.info('File fetched successfully.')
        return model
    except Exception as e:
        logger.error("An error occurred: {}", e)
        return None

def load_tokenizer(path):
    try:
        logger.info(f'Loading tokenizer from : {path}.')
        with open(path, 'rb') as handle:
            pickle.load(handle)

        logger.info(f'File fetched successfully..')
    except Exception as e:
        logger.error("An error occurred: {}", e)

def predict_new_word(config)->str:
    try:
        logger.info(f"Predicting.")

        text = config.get('test_input_string')

        # loading model and tokenizer
        model = load_model_file(
            path=os.path.join(config.get('artifacts_path').get('root_dir_name'),
                            config.get('artifacts_path').get('model_file_name')))
        tokenizer = load_tokenizer(
            path=os.path.join(config.get('artifacts_path').get('root_dir_name'),
                            config.get('artifacts_path').get('tokenizer_file_name')))

        max_sequence_length = model.input_shape[1]+1

        # random initialize if word is not found.
        word = None
        
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_length:
            token_list= token_list[-(max_sequence_length-1):] 
        
        token_list= pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = model.predict(token_list, verbose = 0)
        predicted_word_index = np.argmax(predicted, axis = 1)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word

        logger.info(f'Next predicted word for text :: {text} : {word}.')
        return word
    except Exception as e:
        logger.error("An error occurred: {}", e)
        return None
    
