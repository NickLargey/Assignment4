# Sequence Classification

1. **Definition:** Sequence classification involves assigning a single label or category to an entire sequence of text. The sequence can be a sentence, a paragraph, or a document.
2. **Applications:** Common applications of sequence classification include sentiment analysis (where the sequence is classified as expressing a positive, negative, or neutral sentiment), document classification (e.g., classifying news articles into categories like sports, politics, or entertainment), and spam detection (classifying emails as spam or not spam).
3. **Characteristics:** The main characteristic of sequence classification is that the output is a single label that applies to the entire input sequence. This means that the model needs to understand the overall meaning or theme of the text to make a classification decision.

# HYPERPARAMETERS:

The BERT (Bidirectional Encoder Representations from Transformers) model, introduced by Google, revolutionized the field of Natural Language Processing (NLP) with its deep bidirectional representations. When tuning a BERT model, several hyperparameters can significantly impact its performance. Here's an overview of key hyperparameters associated with BERT:

### Model Architecture Hyperparameters

1. **Number of Layers (L)**: The depth of the model, i.e., the number of transformer blocks. BERT-base uses 12 layers, while BERT-large uses 24.
2. **Hidden Size (H)**: The size of the hidden layers, or the dimensionality of the embeddings. For BERT-base, it's 768, and for BERT-large, it's 1024.
3. **Number of Attention Heads (A)**: The number of attention heads in each transformer block. BERT-base uses 12, and BERT-large uses 16. This parameter influences the model's ability to focus on different parts of the input sequence.

### Training Hyperparameters

1. **Batch Size**: The number of training examples utilized in one iteration. A larger batch size requires more memory but can result in faster training.
2. **Learning Rate**: The step size at each iteration while moving toward a minimum of the loss function. BERT uses a small learning rate, often in the range of 2e-5 to 5e-5 for fine-tuning.
3. **Number of Epochs**: The number of times the learning algorithm will work through the entire training dataset. For fine-tuning, a small number of epochs (2-4) is often sufficient.
4. **Learning Rate Warmup Proportion**: The proportion of training to perform linear learning rate warmup. This means starting with a low learning rate and gradually increasing it to the specified maximum. This is crucial for stabilizing the model's training in the initial phases.
5. **Weight Decay**: Regularization parameter to prevent overfitting by penalizing large weights.

### Tokenization Hyperparameters

1. **Maximum Sequence Length**: The maximum length of the input sequences. Longer sequences require more memory and computational power. BERT has a maximum sequence length of 512 tokens.
2. **Vocabulary Size**: The size of the vocabulary used for tokenization. The original BERT models use a vocabulary size of 30,522 tokens.

### Optimization Hyperparameters

1. **Optimizer**: BERT originally uses the Adam optimizer with specific parameter tuning (like learning rate warmup and weight decay fix).
2. **Gradient Clipping**: The maximum value for gradients to prevent them from becoming too large, which can lead to training instability.

Fine-tuning these hyperparameters can significantly affect the model's performance, depending on the specific task and dataset. It's common to use a validation set to find the optimal set of hyperparameters before evaluating the model on a test set.

# CODE NOTES

## Q: _What does this line of code do:_

```python
with torch.no_grad():
logits = model(**inputs).logits
```

## A:

This line of Python code, often seen in the context of Natural Language Processing (NLP) tasks using PyTorch and models like BERT or other transformers, is doing a few important things in a compact way. Let's break it down:

1. **`with torch.no_grad():`**:

   - This is a context manager provided by PyTorch that temporarily sets all the `requires_grad` flags to `false`. Essentially, it tells PyTorch not to compute or store gradients for any computations that happen within this block. This is useful during inference (i.e., when making predictions with the model on new data), as it reduces memory consumption and speeds up computations since it eliminates the need for gradient computations.

2. **`logits = model(**inputs).logits`\*\*:
   - `model(**inputs)`: This part of the line is where the model makes predictions. The `model` is your pre-trained NLP model loaded into memory, and `inputs` is a dictionary containing input data that the model expects (e.g., input IDs, attention masks). The `**` operator unpacks the `inputs` dictionary into keyword arguments that the model function can accept. This is how you feed input data into the model for it to process.
   - `.logits`: After the model processes the inputs, it returns an output object containing several fields, one of which is `logits`. The `logits` are the raw, unnormalized scores that the model has assigned to each class or output token before any activation function like softmax is applied. In classification tasks, for instance, you would apply a softmax to these logits to get probabilities for each class. For token classification tasks, these logits correspond to scores for each token being in each possible class.

In summary, this line of code is used for running a pre-trained NLP model in inference mode (without computing gradients) on some provided inputs, and it extracts the raw scores (logits) for each class or token produced by the model. This is a common pattern in evaluating or deploying NLP models when you're interested in making predictions without the overhead of gradient computation.

## Q: _How would I create a tokenized set of inputs to run through my model from a pandas df?_

## A:
To create a tokenized set of inputs from a pandas DataFrame to run through an NLP model, such as one from the Transformers library by Hugging Face, follow these steps. I'll assume you're working with a text classification task and using a tokenizer compatible with your model (e.g., BERT tokenizer for a BERT model). The process involves installing necessary libraries, loading your model's tokenizer, preparing your DataFrame, and tokenizing the text data.

### Step 1: Install Necessary Libraries

First, ensure you have the necessary libraries installed. You'll need `pandas` for handling the DataFrame and `transformers` for the tokenizer:

```bash
pip install pandas transformers
```

### Step 2: Load Your Model's Tokenizer

Load the tokenizer associated with your model. This example uses BERT, but adjust according to your specific model:

```python
from transformers import BertTokenizer

# Initialize the tokenizer
model_name = 'bert-base-uncased' # Example model name, change as needed
tokenizer = BertTokenizer.from_pretrained(model_name)
```

### Step 3: Prepare Your DataFrame

Ensure your DataFrame is ready, with at least one column containing the text you want to tokenize. For demonstration, let's assume your DataFrame is named `df` and the text column is named `"text"`.

### Step 4: Tokenize the Text Data

You'll tokenize each row of your DataFrame's text column. The tokenizer can encode your text in a format your model understands, including attention masks and input IDs. For batch processing efficiency, consider tokenizing in batches if your dataset is large.

```python
import pandas as pd

# Assuming df is your DataFrame and it has a column named 'text' containing the text to tokenize

# Tokenize the text
tokenized_inputs = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

# tokenized_inputs is now a dictionary with keys like 'input_ids' and 'attention_mask',
# and values are PyTorch tensors ready to be fed to your model.
```

### Using the Tokenized Inputs

With `tokenized_inputs`, you can feed these directly into your model for inference or training. For example, to run inference:

```python
with torch.no_grad():
    outputs = model(**tokenized_inputs)
    logits = outputs.logits
# Process logits as needed, e.g., applying softmax for probabilities
```

This example demonstrates a straightforward workflow for preparing text data from a pandas DataFrame for NLP modeling with a pre-trained transformer. Adjustments may be necessary depending on your specific model and task (e.g., sequence classification, token classification).


# Paper
This model is an extention of BERT with two extra features, Disentangled Attention and an enhanced mask decoder

DeBERTa improves BERT with two novel components: DA (Disentangled Attention) and an enhanced
mask decoder. Unlike existing approaches that use a single vector to represent both the content and
the position of each input word, the DA mechanism uses two separate vectors: one for the content
and the other for the position. Meanwhile, the DA mechanism’s attention weights among words
are computed via disentangled matrices on both their contents and relative positions. Like BERT,
DeBERTa is pre-trained using masked language modeling. The DA mechanism already considers the
contents and relative positions of the context words, but not the absolute positions of these words,
which in many cases are crucial for the prediction. DeBERTa uses an enhanced mask decoder to
improve MLM by adding absolute position information of the context words at the MLM decoding
layer.




# HUGGINGFACE NLP COURSE NOTES

- Look into Zero-Shot-Classification fine tuning
- 