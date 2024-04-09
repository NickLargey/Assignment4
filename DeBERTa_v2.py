################ SETUP  ################
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, DebertaForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from accelerate import Accelerator
import evaluate
import sys
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

accelerator = Accelerator()

accuracy = evaluate.load("accuracy")

################ LOAD DATA ################
test_df = pd.read_csv("./test.csv")

df = pd.read_csv('./filtered_lyrics.csv')
num_genres = df['Genre'].max()
data_size = df["Genre"].value_counts().min()

samples_per_group = data_size  # Number of samples per group
print("Number of samples per group: ", samples_per_group)
column_to_group_by = 'Genre'

# Sample n rows from each group
df = df.groupby(column_to_group_by).apply(lambda x: x.sample(n=data_size),include_groups=True).reset_index(drop=True)

genre_ints = {
  'Blues': 0,
  'Country': 1,
  'Metal': 2,
  'Pop': 3,
  'Rap': 4,
  'Rock': 5
}

df['Genre'] = df['Genre'].replace(genre_ints)
test_df['Genre'] = test_df['Genre'].replace(genre_ints)

################ MODEL INFERENCE ################ 
m = "microsoft/deberta-base"

# After initial training on deberta-base fine-tune model output checkpoints
model = DebertaForSequenceClassification.from_pretrained(m, num_labels=6) 
tokenizer = AutoTokenizer.from_pretrained(m, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

################ FUNCTIONS ################ 
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, num_steps=10):
        self.num_steps = num_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_steps:
            return {"should_training_stop": True}
        else:
            return {}
        
def label(data):
    return {'title': data['Song Title'],'text': data['Lyrics'], 'labels': data['Genre']}

def tokenize_format(data):
    tokenized = tokenizer(data['text'], truncation=True, max_length=256)
    return tokenized

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

################ PREPROCESS DATA ################ 
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True)

train_dataset = Dataset.from_pandas(train_df).map(label, batched=True)
train_dataset = train_dataset.map(tokenize_format, batched=True)


val_dataset = Dataset.from_pandas(val_df).map(label, batched=True)
val_dataset = val_dataset.map(tokenize_format, batched=True)

test_dataset = Dataset.from_pandas(test_df).map(label, batched=True)
test_dataset = test_dataset.map(tokenize_format, batched=True)

train_dataset = train_dataset.remove_columns(["Song Title","Lyrics", "Genre","__index_level_0__","title", "text"])
train_dataset.set_format("torch")

val_dataset = val_dataset.remove_columns(["Song Title","Lyrics", "Genre","__index_level_0__","title", "text"])
val_dataset.set_format("torch")

test_dataset = test_dataset.remove_columns(["Song Title","Lyrics", "Genre","title", "text"])
test_dataset.set_format("torch")

###### !!!!!!!!!TRYING TO FIGURE OUT WHY THIS GOD DAMN MODEL TAKES UP ALL OF THE GPU MEM SPACE!!!!!!!! ###### 
# print("Train Size: ",sys.getsizeof(train_dataset),"Eval Size: ",sys.getsizeof(val_dataset),"Text Size: ",sys.getsizeof(test_dataset))


dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

################ TRAINING SETUP ################ 
training_args = TrainingArguments(
    output_dir="model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    neftune_noise_alpha=0.1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset= dataset_dict['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callback=EarlyStoppingCallback(),
)

################ TRAINING ################ 
# tqdm(trainer.train(resume_from_checkpoint="model/checkpoint-14020"))

trainer.save_model("./my_fine_tuned_deberta_model")
################ TESTING #################
# eval = trainer.evaluate(dataset_dict['test'])
# print(eval)


