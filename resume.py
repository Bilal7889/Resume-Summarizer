import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------
# LOAD MODEL LOCALLY (NO INTERNET)
# -------------------------------
MODEL_PATH = "final_resume_summarizer_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)


# -------------------------------
# INFERENCE FUNCTION
# -------------------------------
def generate_summary(raw_text):
    prompt = (
        "Act as a Senior Recruiter writing a candidate profile. "
        "Generate a highly impactful, 3-sentence summary. "
        "TEXT: " + raw_text + " --- SUMMARY:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    output_ids = model.generate(
        inputs["input_ids"],
        do_sample=True,
        top_p=0.92,
        temperature=0.7,
        max_length=200,
        min_length=60,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# -------------------------------
# RUN INTERACTIVE LOOP
# -------------------------------
if __name__ == "__main__":
    print("\n=== Resume Summarizer (Local FLAN-T5) ===\n")
    while True:
        text = input("\nPaste resume text (or type 'exit'): \n> ")

        if text.lower().strip() == "exit":
            break

        print("\n--- SUMMARY ---")
        print(generate_summary(text))
        print("\n----------------\n")
