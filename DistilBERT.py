import pandas as pd
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# conda create -n 470assignment4

torch.cuda.empty_cache()


# Load the tokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    # Read CSV
    find_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(find_directory, 'filtered_lyrics.csv')
    df = pd.read_csv(data_path)
    data_size = df["Genre"].value_counts().min()

    samples_per_group = data_size  # Number of samples per group
    print("Number of samples per group: ", samples_per_group)
    column_to_group_by = 'Genre'

    # Sample n rows from each group
    df = df.groupby(column_to_group_by).apply(lambda x: x.sample(n=data_size),include_groups=True).reset_index(drop=True)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Genre'])


    def tokenized_data(df):
        # Tokenize the lyrics
        max_length = 128  # Max length for BERT input
        tokenized_data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing Lyrics"):
            lyric = str(row['Lyrics'])
            label = row['Label']

            encoding = tokenizer.encode_plus(
                lyric,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            tokenized_data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            })
        return tokenized_data

    tokenized_data = tokenized_data(df)
    # Split into train and validation
    train_size = 0.8
    train_data, val_data = train_test_split(tokenized_data, test_size=1 - train_size, random_state=42)

    # Define model
    num_labels = len(df['Label'].unique())
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    # hyperperimeters
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=15,
        per_device_train_batch_size=128,  # Increased batch size
        per_device_eval_batch_size=128,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy='epoch',
        gradient_accumulation_steps=4,  
        save_steps=50,
        save_total_limit=5
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train(resume_from_checkpoint='./results/checkpoint-50')

    find_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(find_directory, 'test.csv')
    test_df = pd.read_csv(data_path)

    label_encoder = LabelEncoder()
    test_df['Label'] = label_encoder.fit_transform(test_df['Genre'])

    test_tokenized_data = tokenized_data(test_df)

    # Evaluate on validation data
    eval_results = trainer.evaluate(eval_dataset=test_tokenized_data)
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    # Save the model
    trainer.save_model("./my_fine_tuned_distilbert_model")
if __name__ == "__main__":
    main()
