from datasets import load_dataset

from transformers import ElectraForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import label_ranking_average_precision_score
import torch
import numpy as np

dataset = load_dataset('smilegate-ai/kor_unsmile')

unsmile_labels = ["여성/가족","남성","성소수자","인종/국적","연령","지역","종교","기타 혐오","악플/욕설","clean"]

model_name = 'beomi/KcELECTRA-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    tokenized_examples = tokenizer(str(examples["문장"]))
    tokenized_examples['labels'] = torch.tensor(examples["labels"], dtype=torch.float)
    # multi label classification 학습을 위해선 label이 float 형태로 변형되어야 합니다.
    # huggingface datasets 최신 버전에는 'map' 함수에 버그가 있어서 변형이 올바르게 되지 않습니다.

    return tokenized_examples

tokenized_dataset = dataset.map(preprocess_function)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask', 'token_type_ids'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

num_labels=len(unsmile_labels) # Label 갯수

model = ElectraForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)
model.config.id2label = {i: label for i, label in zip(range(num_labels), unsmile_labels)}
model.config.label2id = {label: i for i, label in zip(range(num_labels), unsmile_labels)}

def compute_metrics(x):
    return {
        'lrap': label_ranking_average_precision_score(x.label_ids, x.predictions),
    }

batch_size = 32

args = TrainingArguments(
    output_dir="checkpoints",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='lrap',
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.save_model()