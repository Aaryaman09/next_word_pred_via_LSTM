import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pathlib import Path
import tensorflow as tf
import os, pickle
from typing import List, Dict, Union, Tuple

from src.logger import get_logger

logger = get_logger()

class PreprocessingSourceData:
    def __init__(self, config:Dict):
        logger.info("#"*30)
        logger.info("Preprocessing started.")
        logger.info("#"*30)
        self.tokenizer = Tokenizer()
        self.config = config
        self.continue_execution = True

    def pull_raw_data_file(self, path:Path = Path.cwd()/Path('data')/'hamlet.txt')->str:
        try:
            logger.info(f"Pulling hamlet data file from : {path}")
            with open(path,'r') as file:
                text=file.read().lower()

            logger.info('File read successfully.')
            return text
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)
            return None
        
    def fitting_tokenizer_on_text(self, text:str)->None:
        try:
            logger.info(f"Fitting tokenizer on text.")
            self.tokenizer.fit_on_texts([text])

            logger.info('Fitting tokenizer on text successfully.')
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)
        
    def creating_sequences(self, text:str)->Union[List[List[int]],None]:
        """
        spliting sentences into sequeunces, later spliting each sentence into small piece to train 

        example 

        Let's take the sentence:

        "to be or not to be"
        Tokenized as: [1, 2, 3, 4, 1, 2]

        From this, the n-gram sequences generated will be:

        [1, 2] → Learn that "to" is followed by "be"
        [1, 2, 3] → Learn that "to be" is followed by "or"
        [1, 2, 3, 4] → Learn that "to be or" is followed by "not"
        [1, 2, 3, 4, 1] → Learn that "to be or not" is followed by "to"
        [1, 2, 3, 4, 1, 2] → Learn that "to be or not to" is followed by "be"
        Each sequence trains the model to predict the last word.

        """
        try:
            logger.info(f"Creating sequences on text.")

            input_sequences = []

            for line in text.split('\n'):
                token_list = self.tokenizer.texts_to_sequences([line])[0]
                for i in range(1,len(token_list)):
                    n_gram = token_list[:i+1]
                    input_sequences.append(n_gram)

            logger.info('Sequence created successfully.')
            return input_sequences
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)
            return None
        
    def adding_padding(self, input_sequences:List[List[int]])->Union[List[List[int]],None]:
        """
        Why Padding?
        LSTMs expect all inputs to be the same length. Since our sequences have different lengths, we pad them with zeros at the beginning (padding='pre').
        Example Before Padding:

        [ [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5] ]
        After Padding (maxlen=5):

        [ [0, 0, 0, 1, 2],  
        [0, 0, 1, 2, 3],  
        [0, 1, 2, 3, 4],  
        [1, 2, 3, 4, 5] ]  
        Now, all sequences have the same length (5).
        """
        try:
            logger.info(f"Adding padding to sequences.")
            max_sequence_len = max([len(x) for x in input_sequences])
            input_sequences= np.array(pad_sequences(input_sequences, maxlen=max_sequence_len,padding='pre'))

            logger.info('Padding added to sequences successfully.')
            return input_sequences
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)
            return None
        
    def do_train_test_split(self, input_sequences:List[List[int]])->Tuple:
        '''
            X (Input) → All words except the last one (used to predict the next word).
            y (Target) → The last word in each sequence (the word the model should predict).

            Example:
            For the padded sequence [0, 0, 1, 2, 3]:

            x = [0, 0, 1, 2] (features)
            y = 3 (target word)

            Since y contains numerical labels (word indices), we convert it into one-hot encoding for categorical classification.

            Example:
            If total_words = 6 and y = 3, one-hot encoding would be:

            y = [0, 0, 0, 1, 0, 0]  (index 3 is set to 1)
            This helps the model treat it as a classification problem rather than a regression.
            '''
        try:
            logger.info(f"Splitting input_sequences data into train test split.")
            total_words = len(self.tokenizer.word_index)+1

            x,y = input_sequences[:,:-1], input_sequences[:,-1]
            y = tf.keras.utils.to_categorical(y, num_classes=total_words)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            logger.info('input_sequences data splited into train test successfully.')
            return x_train, x_test, y_train, y_test
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)
            return None, None, None, None
        
    def save_file(self, numpy_array:np.array, output_path:Dict):
        try:
            logger.info(f'Saving file.')
            np.save(Path.cwd()/Path('data')/Path(output_path[0])/f'{output_path[1]}.npy', numpy_array)
            logger.info(f"file saved at : {Path.cwd()/Path('data')/Path(output_path[0])/f'{output_path[1]}.npy'}.")
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)

    def save_tokenizer(self):
        try:
            tokenizer_path = os.path.join(
                self.config.get('artifacts_path').get('root_dir_name'),
                self.config.get('artifacts_path').get('tokenizer_file_name'))
            
            logger.info(f'Saving file.')
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f'Tokenizer saved at : {tokenizer_path}.')
        except Exception as e:
            self.continue_execution = False
            logger.error("An error occurred: {}", e)

    def main(self):
        text = self.pull_raw_data_file()
        if self.continue_execution:
            self.fitting_tokenizer_on_text(text=text)

        if self.continue_execution:
            input_sequences = self.creating_sequences(text=text)

        if self.continue_execution:
            input_sequences = self.adding_padding(input_sequences=input_sequences)

        if self.continue_execution:
            x_train, x_test, y_train, y_test = self.do_train_test_split(input_sequences=input_sequences)

        # Save training and testing data
        os.makedirs(Path.cwd()/Path('data')/Path('train_data'),exist_ok=True)
        os.makedirs(Path.cwd()/Path('data')/Path('test_data'),exist_ok=True)

        if self.continue_execution:
            self.save_file(numpy_array=x_train,
                            output_path=['train_data','x_train'])
            
            self.save_file(numpy_array=y_train,
                            output_path=['train_data','y_train'])
            
            self.save_file(numpy_array=x_test,
                            output_path=['test_data','x_test'])
            
            self.save_file(numpy_array=y_test,
                            output_path=['test_data','y_test'])
        
        if self.continue_execution:
            self.save_tokenizer()

        logger.info("#"*30)
        logger.info("Preprocessing completed.")
        logger.info("#"*30)

        return len(self.tokenizer.word_index)+1
