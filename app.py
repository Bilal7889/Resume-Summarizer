import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

print("✅ Starting application...")

# -----------------------------
#  Load Model
# -----------------------------
try:
    print("🔄 Loading model...")
    MODEL_PATH = "final_resume_summarizer_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    print("✅ Tokenizer loaded")
    
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

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

def clean_resume_text(text):
    if pd.isna(text) or not text:
        return ""

    text = str(text)
    print(f"📝 Original text: {text[:200]}...")

    # 1. Remove URLs, Emails, and Phone Numbers
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{10})', '', text)

    # 2. Remove special characters BUT KEEP BASIC PUNCTUATION
    # Keep: letters, numbers, spaces, basic punctuation (. , - +)
    text = re.sub(r'[^\w\s\.\,\-\+]', ' ', text)
    
    # 3. KEEP NUMBERS - they're important for experience, dates, etc.
    # Don't remove numbers: text = re.sub(r'[0-9]', '', text)  # REMOVE THIS LINE

    # 4. Normalize spacing but KEEP ORIGINAL CASE
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. OPTIONAL: Remove stopwords (but test without first)
    # tokens = text.split()
    # tokens = [word for word in tokens if word not in stop_words]
    # text = " ".join(tokens)

    print(f"🔧 Cleaned text: {text[:200]}...")
    return text
def generate_summary(raw_text):
    try:
        print(f"📝 Generating summary for text length: {len(raw_text)}")
        cleaned_text = clean_resume_text(raw_text)
        print(f"🔧 Cleaned text length: {len(cleaned_text)}")
        prompt = (
            "Generate a concise, 2-3 sentence professional summary highlighting key skills. "
            "Act as a Senior Recruiter writing a candidate profile based on the following resume text: "+ cleaned_text
            
            
            # "Act as a Senior Recruiter writing a candidate profile. "
            # "Generate a highly impactful, 3-sentence summary. "
            # "TEXT: " + raw_text + " --- SUMMARY:"
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

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Summary generated: {result[:100]}...")
        return result
        
    except Exception as e:
        print(f"❌ Error in generate_summary: {e}")
        return f"Error: {str(e)}"

# -----------------------------
#  API Route
# -----------------------------
@app.post("/summarize")
async def summarize(data: ResumeText):
    summary = generate_summary(data.text)
    return {"summary": summary}

@app.get("/")
async def root():
    return {"message": "Resume Summarizer API is running!"}

print("🎯 Starting server...")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Server starting on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)