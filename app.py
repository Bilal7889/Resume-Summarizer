import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------------
#  Load Model
# -----------------------------
MODEL_PATH = "final_resume_summarizer_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

# -----------------------------
#  FastAPI App
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeText(BaseModel):
    text: str

# -----------------------------
#  Generate Summary Function
# -----------------------------
def generate_summary(raw_text):

    prompt = (
        "Summarize this resume in 3 professional sentences: "
        + raw_text
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    output = model.generate(
        inputs["input_ids"],
        max_length=200,
        min_length=60,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# -----------------------------
#  API Route
# -----------------------------
@app.post("/summarize")
async def summarize(data: ResumeText):
    summary = generate_summary(data.text)
    return {"summary": summary}
