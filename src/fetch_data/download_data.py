import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
from pathlib import Path

data = gutenberg.raw('shakespeare-hamlet.txt')

with open(Path.cwd()/Path('data')/'hamlet.txt','w') as file:
    file.write(data)