import pandas as pd
from datasets import load_dataset

class DataPreprocessor:
    def __init__(self, dataset_name, split, output_file):
        self.dataset_name = dataset_name
        self.split = split
        self.output_file = output_file

    def load_data(self):
        self.dataset = load_dataset(self.dataset_name, split=self.split)
    
    def format_row(self, row):
        question = row['question']
        answer = row["answer"]
        formatted_string = f"[INST] {question} [/INST] {answer}"
        return formatted_string
    
    def process_data(self):
        df = pd.DataFrame(self.dataset)
        df['Formatted'] = df.apply(self.format_row, axis=1)
        new_df = df.rename(columns={"Formatted": "Text"})
        new_df = new_df[["Text"]]
        new_df.to_csv(self.output_file, index=False)
    
    def run(self):
        self.load_data()
        self.process_data()

if __name__ == "__main__":
        preprocessor = DataPreprocessor("LangChainDatasets/question-answering-state-of-the-union", "train", "data.csv")
        preprocessor.run()
