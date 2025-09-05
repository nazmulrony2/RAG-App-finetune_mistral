# finetune_mistral.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# 1. Load the base model with QLoRA configuration (4-bit quantization)
model_name = "mistral:latest"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 2. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Define LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Common modules for Mistral
)

# 4. Create a sample dataset for RAG
# FORMAT: "Answer the question based on the context. Context: {context} Question: {question} Answer: {answer}"
sample_data = [
    {
        "text": "Answer the question based on the context. Context: The Apollo program was launched by NASA in the 1960s. Its most famous mission, Apollo 11, landed the first humans on the Moon in 1969. Question: When did Apollo 11 land on the Moon? Answer: Apollo 11 landed on the Moon in 1969."
    },
    {
        "text": "Answer the question based on the context. Context: Python is a high-level programming language known for its readability. It was created by Guido van Rossum. Question: Who created Python? Answer: Python was created by Guido van Rossum."
    },
    # ADD YOUR OWN DATA HERE. This is the crucial step.
    # You need 100s to 1000s of examples for good fine-tuning.
    # Use your own documents to create Q&A pairs.
]

dataset = Dataset.from_list(sample_data)

# 5. Set Training Arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=30,
    num_train_epochs=1,
    max_steps=100, # Small for testing, increase for real training
    fp16=True,
    push_to_hub=False, # Set to True if you want to push to Hugging Face Hub
)

# 6. Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512, # Adjust based on your context length
)

# 7. Train the model
trainer.train()

# 8. Save the fine-tuned model
trainer.model.save_pretrained("fine_tuned_mistral_rag")
# You can also merge the adapter and save the full model if you have the RAM
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("full_fine_tuned_model")
print("Training complete and model saved!")