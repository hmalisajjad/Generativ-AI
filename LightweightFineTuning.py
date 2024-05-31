#!/usr/bin/env python
# coding: utf-8

# # Lightweight Fine-Tuning Project

# TODO: In this cell, describe your choices for each of the following
# 
# * PEFT technique: PEFT token classification
# * Model: GPT 2
# * Evaluation approach: "evaluate" and also try to use hugging face "Trainer"
# * Fine-tuning dataset: Hugging Face library "datasets"

# ## Loading and Evaluating a Foundation Model
# 
# TODO: In the cells below, load your chosen pre-trained Hugging Face model and evaluate its performance prior to fine-tuning. This step includes loading an appropriate tokenizer and dataset.

# In[1]:


get_ipython().system('pip install transformers datasets peft')


# In[5]:


from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.resize_token_embeddings(len(tokenizer)) 

model.config.pad_token_id = tokenizer.pad_token_id


# In[6]:


dataset = load_dataset('imdb')  # Using IMDB dataset for binary text classification

def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

encoded_dataset = dataset.map(preprocess_data, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

eval_results = trainer.evaluate()
print(f"Initial Evaluation Results: {eval_results}")


# In[7]:


from peft import get_peft_model, LoraConfig, TaskType


# In[8]:


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=["attn.c_attn"]
)

lora_model = get_peft_model(model, peft_config)

# Fine-tune the Model
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

trainer.train()


# In[9]:


fine_tuned_results = trainer.evaluate()
print(f"Fine-tuned Evaluation Results: {fine_tuned_results}")

print(f"Original Model Accuracy: {eval_results['eval_accuracy']}")
print(f"Fine-tuned Model Accuracy: {fine_tuned_results['eval_accuracy']}")


# ## Performing Parameter-Efficient Fine-Tuning
# 
# TODO: In the cells below, create a PEFT model from your loaded model, run a training loop, and save the PEFT model weights.

# In[10]:


from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=["attn.c_attn"]
)


# In[11]:


from peft import get_peft_model

peft_model = get_peft_model(model, peft_config)


# In[12]:


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

trainer.train()

peft_model.save_pretrained('./trained_peft_model')


# ## Performing Inference with a PEFT Model
# 
# TODO: In the cells below, load the saved PEFT model weights and evaluate the performance of the trained PEFT model. Be sure to compare the results to the results from prior to fine-tuning.

# In[13]:


peft_model = GPT2ForSequenceClassification.from_pretrained('./trained_peft_model', num_labels=2)
peft_model.config.pad_token_id = tokenizer.pad_token_id 

peft_model.eval()


# In[14]:


trainer = Trainer(
    model=peft_model,
    args=training_args,
    eval_dataset=encoded_dataset['test'],
    compute_metrics=compute_metrics,
)

# Evaluate the model
peft_eval_results = trainer.evaluate()
print(f"Evaluation Results with PEFT Model: {peft_eval_results}")


# In[ ]:





# In[ ]:




