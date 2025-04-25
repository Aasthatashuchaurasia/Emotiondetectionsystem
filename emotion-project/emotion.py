import pandas as pd
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# 1-For loading the dataset.
try:
    df = pd.read_csv("go_emotions_dataset.csv")
    print(" Dataset loaded successfully")
except Exception as e:
    print(f" Error loading dataset: {e}")
    exit()

# 2-For cleaning the text.
df['clean_text'] = df['text'].apply(lambda x: nfx.remove_stopwords(str(x)))
df['clean_text'] = df['clean_text'].apply(nfx.remove_special_characters)

# 3- Preparing the  emotion labels - Specific to GoEmotions dataset.
emotion_columns = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]
 
# To Convert emotion columns to integers (1/0) if found boolean.
for col in emotion_columns:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Create emotion label (the emotion with highest score for each text)
df['emotion'] = df[emotion_columns].idxmax(axis=1)

# 4-To analyze class distribution.
print("\n Emotion Distribution:")
print(df['emotion'].value_counts())

# 5- Train-Test Split.
X = df['clean_text']
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  #It will maintain the class distribution.
)

# 6- To build model Pipeline.
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=5,
        max_df=0.7
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='saga'
    ))
])

# 7. for training model
print("\n Training model...")
model.fit(X_train, y_train)

# 8- to Save model
joblib.dump(model, "go_emotions_model.pkl")
print(" Model saved as 'go_emotions_model.pkl'")

# 9- Evaluate
y_pred = model.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 10- The Prediction Function
def predict_emotion(text):
    try:
        cleaned = nfx.remove_stopwords(text)
        cleaned = nfx.remove_special_characters(cleaned)
        return model.predict([cleaned])[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown"

# Demo
print("\n Emotion Detection Demo (type 'quit' to exit)")
while True:
    user_input = input("\nHow are you feeling? ")
    if user_input.lower() == 'quit':
        break
    emotion = predict_emotion(user_input)
    print(f" Detected emotion: {emotion}")