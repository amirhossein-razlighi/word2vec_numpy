# Word2Vec — Pure NumPy Implementation
This repo is for JetBrains internship, and will be completed before the mentioned deadline (7 days from today, which will be 2026-03-11).

## Overview
This project implements the Word2Vec skip-gram model with negative sampling in pure NumPy.
Implemented by: AmirHossein (Amir) NaghiRazlighi

## Usage
We are using the text8 corpus for training, which is a cleaned version of the first 100 MB of Wikipedia text. The script will download it automatically if not found in the current directory (under `data/` folder).

### Environment
Make sure you have Python 3.8+ and NumPy installed. You can install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
Also, matplotlib is optional but recommended for visualizing training curves (and LR schedule).

### Running the Code
To train the model on a subset of the data (for quick testing), you can run:
```bash
python main.py --n_epochs 1
```
The default version uses only a subsample of tokens (~ 2M) for faster iteration. For full training on all 17 M tokens, run:
```bash
python main.py --max_tokens 0 --n_epochs 5
```

You can also specify a custom corpus by providing the path to your text file:
```bash
python main.py --corpus path/to/mytext.txt
```

The training will save the vectors in a __vectors.txt__ file (txt because it can be loaded also by gensim), which can be loaded later for evaluation without retraining:
```bash
python main.py --no_train
```

## Evaluation
After training, the script will evaluate the learned embeddings on a set of analogy tasks (e.g. "king is to man as queen is to ?"). The results will be printed in the console, showing the predicted word and its cosine similarity score.
The evaluation methods is these two:
1. K nearest neighbours (KNN) for a given set of query words.
2. Word analogies, where we test the model's ability to capture relationships between words (e.g. "king is to man as queen is to ?").

