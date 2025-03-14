from src.predictor_method import predict_new_word
from src.utils import read_json
import os

if __name__=='__main__':

    config = read_json(os.path.join('config','config.json'))

    print(f'Next predicted word for text :: {config.get("test_input_string")} : {predict_new_word(config=config)}.')