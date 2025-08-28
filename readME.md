# Deep Learning for Comment-Toxicity Detection  🚫💬

Automated moderation is critical to keep on-line communities safe.  
This repository delivers a full pipeline – from exploratory data analysis to a production-ready Streamlit app – that flags toxic comments using state-of-the-art NLP models (Bi-LSTM, CNN and DistilBERT).

## 📂 Repository structure
.
├── CommentToxicityDL.ipynb # End-to-end notebook (EDA → modelling → explainability)
├── APP.py # Streamlit front-end
├── train.csv # Kaggle Jigsaw toxic-comment training split
├── test.csv # Kaggle Jigsaw test split
├── best_tuned_toxicity_model.keras # Bi-LSTM weights
├── best_tuned_cnn_toxicity_model.keras # CNN weights
├── distilbert_toxicity_model/ # Fine-tuned DistilBERT (TF SavedModel)
│ ├── config.json …
├── requirements.txt
└── README.md # ⇦ you are here
## 🚀 Quick start
## Dataset Download Link
https://drive.google.com/drive/folders/1WXLTp57_TYa61rcPfQIzRUcE1Rz76Emk
(Copy and paste the above link in browser if it isn't redirecting upon clicking..)
## Download the Distilbert model folder(if you do not wish to run the ipynb file then use this from here..)
https://drive.google.com/drive/folders/1TpwOC_6SbPEAXpZViusrnTXAQd1aYC1C?usp=sharing
1. **Clone & install**
git clone 🔧YOUR-GITHUB-URL.git
cd comment-toxicity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
2. **Run the notebook (optional)**
jupyter lab CommentToxicityDL.ipynb
Executing every cell end-to-end will:
- perform EDA with 15+ U/B/M-level charts
- clean & tokenize text (contractions → lemmatisation)
- balance classes (SMOTE) and split train/val
- train three models (Bi-LSTM, CNN, DistilBERT) with Keras-Tuner random search
- benchmark metrics (Accuracy, ROC-AUC, Precision/Recall/F1)
- save best checkpoints to the project root

3. **Launch the Streamlit app**
• Choose a model from the sidebar,  
• paste a single comment **or** upload a CSV with a `comment_text` column.  
The app returns a probability score and a toxic / non-toxic tag for each row.

## 🏗️ Training from scratch
If you deleted the pre-trained weights:


Each script mirrors the hyper-parameters reported in the notebook (batch-size, max_seq_len, learning-rate schedule, etc.).

## 📊 Model performance (validation set)
| Model            | Accuracy | ROC-AUC | F1-Score |
|------------------|----------|---------|----------|
| Bi-LSTM (tuned)  | 0.89     | 0.93    | 0.87     |
| CNN (tuned)      | 0.88     | 0.91    | 0.86     |
| **DistilBERT (tuned)** | **0.91** | **0.94** | **0.89** |

DistilBERT is the production choice: it pairs the best scores with fast inference (<70 ms on CPU for a 120-token comment).

## 🧐 Explainability
SHAP explanations (see notebook section 7.3) highlight tokens like *“idiot”, “hate”* as positive contributors to toxicity probability, ensuring the model’s decisions align with human intuition.

## 🛠️ Requirements
Python 3.9+  
Key packages: `tensorflow>=2.15`, `torch>=2.2`, `transformers>=4.40`, `keras-tuner`, `streamlit`, `imblearn`, `shap`, `wordcloud`.  
Exact versions are pinned in `requirements.txt`.

## 📜 License
MIT – see LICENSE file.

## 🙌 Acknowledgements
Dataset: Jigsaw/Conversation-AI “Toxic Comment Classification Challenge”.  
Project author: Dhruv Tamirisa.  
Guidelines template courtesy of the AlmaBetter ML capstone rubric.



