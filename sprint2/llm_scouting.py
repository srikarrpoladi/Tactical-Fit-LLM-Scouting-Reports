import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
import torch
import os

BASE_MODEL = "sshleifer/tiny-gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"


def fine_tune_model(df, output_dir="fine_tuned_tiny_gpt2", epochs=1):
    """
    Fine-tune tiny-GPT2 on player stats.
    df: DataFrame with columns ['name','position','attack','passing','dribbling','defense','physical','goalkeeping','play_style']
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare training text
    texts = df.apply(
        lambda row: f"Player: {row.get('name','N/A')}, Position: {row.get('position','N/A')}, "
                    f"Stats: Attack {row.get('attack','N/A')}, Passing {row.get('passing','N/A')}, "
                    f"Dribbling {row.get('dribbling','N/A')}, Defense {row.get('defense','N/A')}, "
                    f"Physical {row.get('physical','N/A')}, Goalkeeping {row.get('goalkeeping','N/A')}, "
                    f"Playstyles: {row.get('play_style','N/A')}. Report:",
        axis=1
    ).tolist()

    dataset = Dataset.from_dict({"text": texts})
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
        learning_rate=5e-5
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Fine-tuning tiny-GPT2 on scouting data (may take a while)...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def generate_batch_reports(df, fine_tuned_dir=None, limit=5):
    """
    Generate scouting reports using the fine-tuned tiny-GPT2.
    """
    if fine_tuned_dir:
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_dir)
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_dir).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=150
    )

    reports = []
    for i, row in df.iterrows():
        if limit and i >= limit:
            break
        prompt = f"Player: {row.get('name','N/A')}, Position: {row.get('position','N/A')}, " \
                 f"Stats: Attack {row.get('attack','N/A')}, Passing {row.get('passing','N/A')}, " \
                 f"Dribbling {row.get('dribbling','N/A')}, Defense {row.get('defense','N/A')}, " \
                 f"Physical {row.get('physical','N/A')}, Goalkeeping {row.get('goalkeeping','N/A')}, " \
                 f"Playstyles: {row.get('play_style','N/A')}. Report:"

        output = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
        reports.append({
            "name": row.get("name", "N/A"),
            "cluster": row.get("cluster", "N/A"),
            "scouting_report": output[0]['generated_text'].strip()
        })

    return pd.DataFrame(reports)
