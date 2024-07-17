import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

class ModelTrainer:
    def __init__(self, data_file, model_name, output_dir, max_seq_length=2048, batch_size=8, accumulation_steps=4, warmup_steps=10, num_epochs=20, learning_rate=2e-4):
        self.data_file = data_file
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def load_data(self):
        self.dataset = load_dataset("csv", data_files=self.data_file, split='train')
    
    def load_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True
        )
    
    def configure_model(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
            max_seq_length=self.max_seq_length
        )
    
    def train_model(self):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            dataset_text_field="Text",
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            args=TrainingArguments(
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=self.accumulation_steps,
                warmup_steps=self.warmup_steps,
                num_train_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                output_dir=self.output_dir,
                optim="adamw_8bit",
                seed=3407,
            )
        )
        trainer.train()
    
    def run(self):
        self.load_data()
        self.load_model()
        self.configure_model()
        self.train_model()

if __name__ == "__main__":
    trainer = ModelTrainer(
        data_file="data.csv",
        model_name="unsloth/mistral-7b-bnb-4bit",
        output_dir="unsloth-test"
    )
    trainer.run()
