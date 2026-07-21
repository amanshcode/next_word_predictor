# 📝 Next Word Predictor (Sentence Completion)

A deep learning-based **Next Word Predictor (Sentence Completion)** built using **LSTM (Long Short-Term Memory)** networks and **Natural Language Processing (NLP)**. The application predicts the most probable next word based on the text entered by the user, demonstrating how sequence models can understand context in natural language.

> ⚠️ This is a learning project focused on understanding RNNs, LSTMs, and NLP. Since the model is trained on a relatively small dataset, its predictions may not always be as accurate as production-grade language models.

---

## 🚀 Live Demo

🔗 https://nextwordpredictor-2kpukymhbqjvwmazzd9iz9.streamlit.app/


## ✨ Features

- Predicts the next word from user input
- Sentence completion functionality
- Built using LSTM for sequence learning
- User-friendly web interface
- Real-time predictions
- Trained on a text dataset using TensorFlow/Keras

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- LSTM
- RNN
- Natural Language Processing (NLP)
- NumPy
- Pandas
- Streamlit

---

## 🧠 How It Works

1. The text dataset is cleaned and preprocessed.
2. Text is tokenized into numerical sequences.
3. Input sequences are padded to a fixed length.
4. An Embedding layer converts words into dense vectors.
5. An LSTM network learns contextual relationships between words.
6. The final Dense layer predicts the probability of the next word.
7. The predicted word is displayed to the user.

---

## 📁 Project Structure

```text
Next-Word-Predictor/
│
├── app.py                  # Streamlit application
├── model.keras             # Trained LSTM model
├── tokenizer.pkl           # Saved tokenizer
├── requirements.txt
├── runtime.txt
├── README.md
├── notebook.ipynb          # Model training notebook
└── dataset/
    └── dataset.csv
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/ammanshcode/next-word-predictor.git
```

```bash
cd next-word-predictor
```

### Create a virtual environment

**Windows**

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 📊 Model Overview

| Component | Description |
|-----------|-------------|
| Embedding Layer | Converts words into dense vectors |
| LSTM Layer | Learns sequential context |
| Dense Layer | Predicts next word probabilities |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |

---

## 📌 Limitations

- Trained on a relatively small dataset.
- Accuracy is lower than large language models.
- Predictions depend heavily on training data quality.
- Cannot understand long-range context like transformer-based models (e.g., GPT).

---

## 📚 What I Learned

Through this project, I gained practical experience with:

- Text preprocessing
- Tokenization
- Word embeddings
- Sequence generation
- Padding
- LSTM architecture
- Training deep learning models
- Saving and loading TensorFlow models
- Building NLP applications with Streamlit

---

## 🔮 Future Improvements

- Train on a much larger dataset
- Add beam search prediction
- Predict multiple next-word suggestions
- Improve UI/UX
- Experiment with GRU models
- Build a Transformer-based version
- Deploy using Docker

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome!

If you find this project useful, feel free to fork the repository and submit a pull request.

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.

---
