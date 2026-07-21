from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


st.set_page_config(
	page_title="Sentence Completion",
	page_icon="✦",
	layout="wide",
	initial_sidebar_state="expanded",
)


st.markdown(
	"""
	<style>
	:root {
		--bg: #f5efe6;
		--panel: rgba(255, 255, 255, 0.72);
		--panel-border: rgba(46, 52, 64, 0.08);
		--text: #1f2328;
		--muted: #5c6570;
		--accent: #b85c38;
		--accent-soft: #f6d7c3;
	}

	.stApp {
		background:
			radial-gradient(circle at top left, rgba(184, 92, 56, 0.18), transparent 28%),
			radial-gradient(circle at right 15%, rgba(72, 105, 147, 0.16), transparent 22%),
			linear-gradient(180deg, #fbf7f2 0%, #f3ede3 100%);
		color: var(--text);
	}

	.block-container {
		padding-top: 2rem;
		padding-bottom: 2rem;
		max-width: 1180px;
	}

	.hero {
		padding: 1.4rem 1.5rem;
		border-radius: 24px;
		background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(255, 248, 240, 0.74));
		border: 1px solid var(--panel-border);
		box-shadow: 0 20px 60px rgba(51, 41, 28, 0.08);
		margin-bottom: 1.25rem;
	}

	.hero h1 {
		margin: 0;
		font-size: 3rem;
		letter-spacing: -0.04em;
		color: var(--text);
	}

	.hero p {
		margin: 0.45rem 0 0;
		color: var(--muted);
		font-size: 1.03rem;
		line-height: 1.55;
	}

	.panel {
		background: var(--panel);
		backdrop-filter: blur(18px);
		border: 1px solid var(--panel-border);
		border-radius: 22px;
		padding: 1.15rem 1.15rem 1.05rem;
		box-shadow: 0 16px 42px rgba(51, 41, 28, 0.07);
		height: 100%;
	}

	.metric-card {
		border-radius: 18px;
		padding: 0.95rem 1rem;
		background: rgba(255, 255, 255, 0.8);
		border: 1px solid rgba(184, 92, 56, 0.12);
	}

	.metric-label {
		font-size: 0.8rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--muted);
		margin-bottom: 0.25rem;
	}

	.metric-value {
		font-size: 1.55rem;
		font-weight: 700;
		color: var(--text);
	}

	.suggestion-pill {
		display: inline-block;
		margin: 0.25rem 0.4rem 0.25rem 0;
		padding: 0.45rem 0.8rem;
		border-radius: 999px;
		background: var(--accent-soft);
		color: #5c2f1f;
		border: 1px solid rgba(184, 92, 56, 0.16);
		font-size: 0.95rem;
		font-weight: 600;
	}

	.stTextArea textarea {
		border-radius: 18px !important;
		border: 1px solid rgba(92, 101, 112, 0.18) !important;
		background: rgba(255, 255, 255, 0.9) !important;
		min-height: 160px;
		font-size: 1rem !important;
		line-height: 1.6 !important;
	}

	.stButton > button {
		border-radius: 999px;
		padding: 0.7rem 1.1rem;
		border: 0;
		background: linear-gradient(135deg, #b85c38, #7c4c3e);
		color: white;
		font-weight: 700;
		box-shadow: 0 12px 24px rgba(184, 92, 56, 0.25);
	}

	.stButton > button:hover {
		filter: brightness(1.04);
		transform: translateY(-1px);
	}

	.footer-note {
		color: var(--muted);
		font-size: 0.9rem;
		line-height: 1.55;
	}

	</style>
	""",
	unsafe_allow_html=True,
)


def clean_text(text: str) -> str:
	text = text.lower().strip()
	text = re.sub(r"[^a-z0-9\s']+", " ", text)
	text = re.sub(r"\s+", " ", text)
	return text.strip()


@st.cache_resource(show_spinner=False)
def load_artifacts():
	try:
		import tensorflow as tf
		from tensorflow.keras.preprocessing.sequence import pad_sequences
	except Exception as exc:  # pragma: no cover - runtime dependency guard
		return {"error": f"TensorFlow is required to load the model: {exc}"}

	try:
		with open(TOKENIZER_PATH, "rb") as file_handle:
			tokenizer = pickle.load(file_handle)
		with open(MAX_LEN_PATH, "rb") as file_handle:
			max_len = pickle.load(file_handle)
		model = tf.keras.models.load_model(MODEL_PATH)
	except Exception as exc:  # pragma: no cover - runtime dependency guard
		return {"error": f"Failed to load model artifacts: {exc}"}

	return {
		"model": model,
		"tokenizer": tokenizer,
		"max_len": int(max_len),
		"pad_sequences": pad_sequences,
	}


def predict_next_words(model, tokenizer, pad_sequences, max_len: int, seed_text: str, top_k: int, temperature: float):
	cleaned = clean_text(seed_text)
	sequence = tokenizer.texts_to_sequences([cleaned])[0]

	if not sequence:
		return []

	padded = pad_sequences([sequence], maxlen=max_len, padding="pre")
	probabilities = np.asarray(model.predict(padded, verbose=0)[0], dtype=np.float64)
	probabilities = np.maximum(probabilities, 1e-12)

	if temperature != 1.0:
		logits = np.log(probabilities) / max(temperature, 1e-6)
		logits -= logits.max()
		probabilities = np.exp(logits)
		probabilities /= probabilities.sum()

	top_indices = np.argsort(probabilities)[::-1]
	predictions = []

	for index in top_indices:
		if index == 0:
			continue
		word = tokenizer.index_word.get(int(index))
		if not word:
			continue
		predictions.append((word, float(probabilities[index])))
		if len(predictions) >= top_k:
			break

	return predictions


def extend_text(model, tokenizer, pad_sequences, max_len: int, seed_text: str, new_words: int):
	generated = seed_text.strip()
	current = generated

	for _ in range(new_words):
		predictions = predict_next_words(
			model=model,
			tokenizer=tokenizer,
			pad_sequences=pad_sequences,
			max_len=max_len,
			seed_text=current,
			top_k=1,
			temperature=1.0,
		)
		if not predictions:
			break
		next_word = predictions[0][0]
		generated = f"{generated} {next_word}".strip()
		current = generated

	return generated


artifacts = load_artifacts()

st.markdown(
	"""
	<div class="hero">
		<h1>Sentence Completion</h1>
		<p>
			Type a prompt and let the LSTM model suggest the next word or extend the sentence.
			The app uses the saved tokenizer and max sequence length from the trained pipeline.
		</p>
	</div>
	""",
	unsafe_allow_html=True,
)

if "error" in artifacts:
	st.error(artifacts["error"])
	st.info(
		"Install the model runtime dependencies in a Python environment that supports TensorFlow, then rerun the app."
	)
	st.stop()

model = artifacts["model"]
tokenizer = artifacts["tokenizer"]
max_len = artifacts["max_len"]
pad_sequences = artifacts["pad_sequences"]

with st.sidebar:
	st.markdown("### Model")
	st.markdown(
		f"""
		<div class="metric-card">
			<div class="metric-label">Vocabulary size</div>
			<div class="metric-value">{len(tokenizer.word_index):,}</div>
		</div>
		<div style="height: 0.75rem;"></div>
		<div class="metric-card">
			<div class="metric-label">Max sequence length</div>
			<div class="metric-value">{max_len:,}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	top_k = st.slider("Suggestions", min_value=1, max_value=10, value=5)
	temperature = st.slider("Creativity", min_value=0.4, max_value=1.8, value=1.0, step=0.1)
	new_words = st.slider("Auto-extend words", min_value=1, max_value=12, value=5)


left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
	st.markdown("### Prompt")
	default_prompt = "the quick brown fox"
	seed_text = st.text_area(
		"Enter a sentence fragment",
		value=default_prompt,
		height=160,
		placeholder="Start typing a phrase...",
		label_visibility="collapsed",
	)

	action_col_1, action_col_2 = st.columns(2)
	predict_clicked = action_col_1.button("Predict next word", use_container_width=True)
	extend_clicked = action_col_2.button("Generate continuation", use_container_width=True)

	if predict_clicked or extend_clicked:
		if not seed_text.strip():
			st.warning("Enter a prompt first.")
		else:
			if extend_clicked:
				completed = extend_text(model, tokenizer, pad_sequences, max_len, seed_text, new_words)
				st.markdown("### Generated text")
				st.success(completed)

			predictions = predict_next_words(
				model=model,
				tokenizer=tokenizer,
				pad_sequences=pad_sequences,
				max_len=max_len,
				seed_text=seed_text,
				top_k=top_k,
				temperature=temperature,
			)

			st.markdown("### Top suggestions")
			if predictions:
				pill_html = "".join(
					f'<span class="suggestion-pill">{word} <span style="opacity:0.7">{prob:.2%}</span></span>'
					for word, prob in predictions
				)
				st.markdown(pill_html, unsafe_allow_html=True)

				suggestion_table = {
					"Rank": list(range(1, len(predictions) + 1)),
					"Word": [word for word, _ in predictions],
					"Probability": [f"{prob:.2%}" for _, prob in predictions],
				}
				st.table(suggestion_table)
			else:
				st.warning("No valid token could be produced from the current prompt.")

with right_col:
	st.markdown("### How it works")
	st.markdown(
		"""
		<div class="panel">
			<p class="footer-note">
				The model tokenizes your prompt, pads it to the sequence length used during training,
				and returns the most likely next token from the learned vocabulary.
			</p>
			<p class="footer-note">
				Use the slider to control how many candidate words appear and whether the model should
				stay conservative or sample a slightly broader distribution.
			</p>
			<p class="footer-note">
				If you want a stronger demo, replace the default prompt with a sentence fragment from your
				training corpus so the model can leverage familiar context.
			</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

