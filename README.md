# Next Word Prediction using LSTM & GRU

## Overview

This project implements a next-word prediction model using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) networks. The model is trained on a text corpus and deployed as a web application using Streamlit. Users can input a sequence of words, and the app predicts the most probable next word based on the trained deep learning models.

## Features

* Supports both LSTM and GRU architectures for comparison.

* Trained on a large corpus to provide accurate next-word predictions.

* Interactive web application built with Streamlit.

* Real-time text input and word prediction.

* User-friendly UI with model selection options.

## Technologies Used

* Python

* TensorFlow/Keras

* Streamlit

* Numpy & Pandas

* NLTK & spaCy (for text preprocessing)

* Matplotlib & Seaborn (for visualization)

## Installation

1. ### Clone the repository:
  ```python
  git clone https://github.com/yourusername/next-word-prediction.git
  cd next-word-prediction
  ```
2. ### Create a virtual environment and activate it:
  ```python
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. ### Install dependencies:
  ```python
pip install -r requirements.txt
```
4. ### Run the Streamlit app:
  ```python
streamlit run app.py
```
## Usage

* Open the Streamlit app in your browser.

* Enter a sequence of words in the input box.

* Choose the prediction model (LSTM or GRU).

* Click the predict button to get the next-word suggestion.

## Dataset

* The model is trained on a publicly available text corpus such as Wikipedia dumps, news articles, or literature datasets.

* Preprocessing includes tokenization, padding, and word embeddings (Word2Vec, GloVe, or embeddings learned during training).

## Model Training

* The LSTM and GRU models are trained using Keras with TensorFlow backend.

* Tokenization and sequence generation are handled using NLTK and spaCy.

* The models are optimized using Adam optimizer with categorical cross-entropy loss.

## Results & Evaluation

* Performance is measured using perplexity and accuracy.

* Comparison between LSTM and GRU architectures is visualized using accuracy and loss curves.

## Future Improvements

* Add transformer-based models like GPT or BERT for better performance.

* Enable fine-tuning with user-provided datasets.

* Deploy on cloud platforms like AWS, GCP, or Hugging Face Spaces.

## Contributing

* Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License

* This project is licensed under the MIT License.

## Contact

* For any queries or collaboration, reach out via GitHub issues or email at aaryaman09@gmail.com.
