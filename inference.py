import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextGenerator:
    def __init__(self, checkpoint_path, torch_dtype=torch.float16, device_map="auto"):
        self.checkpoint_path = checkpoint_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map
        )

    def generate_text(self, input_text, max_new_tokens=100, temperature=0.2, do_sample=True, eos_token_id=1):
        # Tokenize the input
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=eos_token_id
            )

        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

if __name__ == "__main__":
    # Path to your checkpoint
    checkpoint_path = "/content/drive/MyDrive/unsloth-test/checkpoint-10"

    # Initialize the text generator
    text_generator = TextGenerator(checkpoint_path)

    # Input text
    input_text = "step of US in Chinisse Aggression"

    # Generate and print the output
    generated_text = text_generator.generate_text(input_text)
    print(generated_text)
