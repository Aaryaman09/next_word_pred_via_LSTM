import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pathlib import Path
import tensorflow as tf

with open(Path.cwd()/Path('data')/'hamlet.txt','r') as file:
    text=file.read().lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index)+1

# print(tokenizer.word_index)

input_sequences = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences= np.array(pad_sequences(input_sequences, maxlen=max_sequence_len,padding='pre'))

x,y = input_sequences[:,:-1], input_sequences[:,-1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



