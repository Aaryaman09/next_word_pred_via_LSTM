from src.preprocessing.data_preprocessing import PreprocessingSourceData
from src.model_training import ModelTraining
from src.utils import read_json
import os

if __name__ == '__main__':
    config = read_json(os.path.join('config','config.json'))
    obj = PreprocessingSourceData(config=config)
    total_words = obj.main()

    model_obj = ModelTraining(config=config,
                              total_words=total_words)
    model_obj.main()