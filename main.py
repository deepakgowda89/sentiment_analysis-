from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import onnxruntime as ort

import numpy as np
import re
import pickle


# ============================================================
# 1. CONSTANTS
# ============================================================

# Model path
model_path = "Artifacts/Bigru_model.onnx"

# Tokenizer path
tokenizer_path = "Artifacts/tokenizer_word_index.pkl"

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

    # Load ONNX inference session
    session = ort.InferenceSession(model_path)
    dl_model["session"] = session
    dl_model["input_name"] = session.get_inputs()[0].name
    dl_model["output_name"] = session.get_outputs()[0].name

    # Load tokenizer word index (plain dict, no Keras dependency)
    with open(tokenizer_path, "rb") as file:
        dl_model["word_index"] = pickle.load(file)

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

    # Get ONNX session
    session = dl_model.get("session")
    input_name = dl_model.get("input_name")
    output_name = dl_model.get("output_name")

    # Get tokenizer word index
    word_index = dl_model.get("word_index")

    # Check model availability
    if session is None or word_index is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later."
        )

    # --------------------------------------------------------
    # 1. Preprocess text
    # --------------------------------------------------------

    cleaned_text = preprocessing_txt(text_input.text)

    # --------------------------------------------------------
    # 2. Tokenization (pure Python, no Keras)
    # --------------------------------------------------------

    tokens = [word_index.get(w, 0) for w in cleaned_text.split()]

    # --------------------------------------------------------
    # 3. Padding
    # --------------------------------------------------------

    seq = tokens[:max_length]
    padded = np.zeros((1, max_length), dtype=np.float32)
    padded[0, :len(seq)] = seq

    # --------------------------------------------------------
    # 4. Prediction
    # --------------------------------------------------------

    input_data = padded.astype(np.float32)
    probabilities = session.run([output_name], {input_name: input_data})[0][0]

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