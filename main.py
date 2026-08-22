from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import re
import pickle


# ============================================================
# 1. CONSTANTS
# ============================================================

# Model path
model_path = r"C:\Users\deepa\OneDrive\Documents\Desktop\Sentiment Analysis\Artifacts\Bigru_model.keras"

# Tokenizer path
tokenizer_path = r"Artifacts\tokenizer_pkl"

# Maximum sequence length
max_length = 50

# Emotion labels
emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

# Emotion emojis
emotion_emoji = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================

def preprocessing_txt(text: str) -> str:

    # Convert to lowercase
    text = text.lower()

    # Remove apostrophes
    # Example: can't -> cant
    text = re.sub(r"'", "", text)

    # Remove special characters and punctuation
    text = re.sub(r"[^a-z0-9\s,]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 3. REQUEST / RESPONSE SCHEMAS
# ============================================================

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyse",
        json_schema_extra={
            "example": "I feel so excited"
        }
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================
# 4. MODEL LOADING / LIFESPAN
# ============================================================

dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading the model and tokenizer...")

    # Load BiGRU model
    dl_model["BiGRU"] = load_model(model_path)

    # Load tokenizer
    with open(tokenizer_path, "rb") as file:
        dl_model["tokenizer"] = pickle.load(file)

    print("Model loaded successfully")

    yield

    # Clear models when server stops
    dl_model.clear()

    print("Model unloaded")


# ============================================================
# 5. CREATE FASTAPI APP
# ============================================================

app = FastAPI(lifespan=lifespan)


# ============================================================
# 6. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# 7. STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# 8. API ENDPOINTS
# ============================================================

# -------------------------
# Homepage
# -------------------------

@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index_1.html")


# -------------------------
# Health check
# -------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():

    return HealthResponse(
        status="server is running",
        model_loaded=bool(dl_model)
    )


# -------------------------
# Prediction
# -------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):

    # Get model
    bigru_model = dl_model.get("BiGRU")

    # Get tokenizer
    tokenizer_model = dl_model.get("tokenizer")

    # Check model availability
    if bigru_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    # --------------------------------------------------------
    # 1. Preprocess text
    # --------------------------------------------------------

    cleaned_text = preprocessing_txt(text_input.text)

    # --------------------------------------------------------
    # 2. Tokenization
    # --------------------------------------------------------

    tokenized_text = tokenizer_model.texts_to_sequences(
        [cleaned_text]
    )

    # --------------------------------------------------------
    # 3. Padding
    # --------------------------------------------------------

    padded_text = pad_sequences(
        tokenized_text,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    # --------------------------------------------------------
    # 4. Prediction
    # --------------------------------------------------------

    probabilities = bigru_model.predict(
        padded_text,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # 5. Find highest probability emotion
    # --------------------------------------------------------

    top_emotion_index = int(
        np.argmax(probabilities)
    )

    # --------------------------------------------------------
    # 6. All probabilities
    # --------------------------------------------------------

    all_probabilities = {
        label: float(prob)
        for label, prob in zip(
            emotion_labels,
            probabilities
        )
    }

    # --------------------------------------------------------
    # 7. Return response
    # --------------------------------------------------------

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(
            probabilities[top_emotion_index]
        ),
        all_probabilities=all_probabilities
    )