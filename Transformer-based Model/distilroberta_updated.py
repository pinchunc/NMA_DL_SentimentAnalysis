import pandas as pd
import re # library called regular expresions that helps to search, match, and manipulate strings using patterns.
import nltk # library called natural language toolkit that allows us to tokenize, classify and clean language data.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import RobertaTokenizerFast, AutoModel, AutoConfig, PreTrainedModel, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader # allows to create a class object dataset
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

# Set seed
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)

# Set device
def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device

DEVICE = set_device()

# Load the dataset
url = 'https://raw.githubusercontent.com/pinchunc/NMA_DL_SentimentAnalysis/main/data/hippoCorpusV2.csv'
df = pd.read_csv(url)
df = df[['story', 'memType', 'stressful']].dropna() # we exclude n/a spaces
df = df[df['memType'] == 'recalled'] # Filter to include only 'recall' entries in 'memType'
df.head()

# @title Exploratory Data Analysis

# Count number of words in each story
df['word_count'] = df['story'].astype(str).apply(lambda x: len(x.split()))
word_count_average = df['word_count'].mean()

# Plot distribution of word count
df['word_count'].hist(bins=50)
plt.xlabel("Number of words")
plt.ylabel("Number of stories")
plt.title("Story length distribution")
plt.show()

# Plot distribution of stress levels
plt.figure(figsize=(6, 4))
sns.histplot(df['stressful'], discrete=True, kde=False)
plt.title('Histogram of Stress Levels')
plt.xlabel('Stress Level')
plt.ylabel('Count')
plt.show()

# Plot distribution of memory types
plt.figure(figsize=(6, 4))
sns.histplot(df['memType'], discrete=True, kde=False)
plt.title('Histogram of Memory Types')
plt.xlabel('Memory Type')
plt.ylabel('Count')
plt.show()

print(f"Word Count Average = {word_count_average}")
print(df['stressful'].value_counts().sort_index())
print(df['memType'].value_counts())

# @title Relabel Stress Levels

# Exclude rows with stress level 2
df = df[df['stressful'] != 2]

def relabel_stress(level):
  if level == 1:
    return 0
  else:
    return 1

df['stress_label'] = df['stressful'].apply(relabel_stress)

# Plot new grouped stress levels
plt.figure(figsize=(6, 4))
sns.histplot(df['stress_label'], discrete=True, kde=False)
plt.title('Histogram of Grouped Stress Levels')
plt.xlabel('Stress Category')
plt.ylabel('Count')
plt.xticks([0, 1], ['Low', 'High'])
plt.show()

print("Unique values in original 'stressful' column:", df['stressful'].unique())
print("Unique values in new 'stress_label' column:", df['stress_label'].unique())

# @title 1) Preprocessing
# Minimal cleaning because we are using DistilRoberta that is trained already in raw unnormalised data
def minimal_clean(text):
    text = str(text)                              # 1) Ensure it's a string
    text = re.sub(r'\s+', ' ', text)              # 2) Replace multiple spaces/newlines with a single space
    text = text.strip()                           # 3) Trim leading/trailing whitespace
    return text

df["clean_story"] = df["story"].apply(minimal_clean)

# Minimal preprocessing for using pandas (faster)
# df['story'] = df['story'].astype(str).fillna('').str.strip()

# Encode labels as intergers for classification
df['stress_label'] = df['stress_label'].astype(int)

# @title 2) Tokenization with DistilRoberta tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')

# Tokenize only first "max_length" words of each story (truncate if they exceed the word limit and pad to mantain length of stories consistent)
tokens = tokenizer(list(df['clean_story']), truncation=True, padding='max_length', max_length=512)

# @title 3) Create Class Object
# Create an object of class StressDataset to use Hugging Face's 'Trainer'
class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
      """
        texts: list or array of input texts (cleaned stories)
        labels: list or array of corresponding stress level labels
        tokenizer: Hugging Face DestilRoberta Tokenizer
        max_len: max length of tokenized input (default = 400 tokens)
      """

      self.texts = texts
      self.labels = labels
      self.tokenizer = tokenizer
      self.max_len = max_len

    def __getitem__(self, idx):
        encoding = self.tokenizer(    # tokenize the text
            self.texts[idx],          # retrieves the story text at index 'idx'
            truncation=True,          # cuts the input off it it exceeds max_len
            padding='max_length',     # pads the input up to max_len tokens
            max_length=self.max_len,  # sets the maximum length of tokens
            return_tensors='pt'       # Returns tensors in PyTorch format
        )

        # This diccionary with token IDs, masks for padding and labels of stress levels
        return {
            'input_ids': encoding['input_ids'].squeeze(),               # token IDs of the story text
            'attention_mask': encoding['attention_mask'].squeeze(),     # binary mask (1s and 0s) telling the model which tokens are content and which are padding.
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)  # retrieves the corresponding label of the story at index 'idx' and converts it into a PyTorch tensor with interger type (long) for classification
        }

    def __len__(self):
        return len(self.texts) # this returns the number of stories in the dataset

# @title 4) Split the dataset (training, validation and testing)

texts = df["clean_story"]
labels = df["stress_label"]

# Step 1: Split into 80% train and 20% temp (for val + test)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2025)
train_idx, temp_idx = next(splitter.split(texts, labels))

train_texts  = texts.iloc[train_idx].tolist()
train_labels = labels.iloc[train_idx].tolist()
temp_texts   = texts.iloc[temp_idx].tolist()
temp_labels  = labels.iloc[temp_idx].tolist()

# Step 2: Split temp into 50% val and 50% test → 10% each of the full dataset
splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(splitter_val.split(temp_texts, temp_labels))

val_texts = [temp_texts[i] for i in val_idx]
val_labels = [temp_labels[i] for i in val_idx]
test_texts = [temp_texts[i] for i in test_idx]
test_labels = [temp_labels[i] for i in test_idx]

# Build datasets
train_dataset = StressDataset(train_texts, train_labels, tokenizer)
val_dataset = StressDataset(val_texts, val_labels, tokenizer)
test_dataset = StressDataset(test_texts, test_labels, tokenizer)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=[0, 1],
    y=train_labels
)

weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print("Class weights:", weights)

# Focal Loss to include in Model
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.alpha is not None and not isinstance(self.alpha, torch.Tensor):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float, device=DEVICE)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # predicted prob of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Attention pooling to include in Model
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(weights * lstm_output, dim=1)
        return pooled

# @title 5) Define Custom DistilRoberta Model with Hidden Layer
class DistilRobertaClassifier(PreTrainedModel):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.class_weights = class_weights
        self.distilroberta = AutoModel.from_pretrained("distilroberta-base", config=config) # DistilRoberta model

        self.hidden_size = config.hidden_size
        self.lstm_hidden_size = 128
        self.attn_pool = AttentionPooling(input_dim=self.lstm_hidden_size * 2)

        # Bidirectional LSTM
        self.bi_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected layers
        self.fused_dim = self.lstm_hidden_size * 2 + self.hidden_size
        self.pre_classifier = nn.Linear(self.fused_dim, 256)
        self.relu = nn.ReLU() # ReLU
        self.dropout = nn.Dropout(0.5) # regularisation
        self.classifier = nn.Linear(256, 2)  # 2 output classes

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.distilroberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0] # Extract CLS embedding
        hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Save CLS embedding as an instance variable if needed externally
        self._last_cls = cls_embedding

        # LSTM layer
        lstm_output, _ = self.bi_lstm(hidden_state)

        # Attention pooling to capture the meaning of the story
        pooled_output = self.attn_pool(lstm_output)

        # Fuse CLS and pooled embeddings
        fused_output = torch.cat([pooled_output, cls_embedding], dim=1)

        # Feedforward classifier
        x = self.pre_classifier(fused_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
            loss = loss_fn(logits, labels)

        if loss is not None:
            loss = loss.unsqueeze(0)  # Adds batch dimension

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

# @title 6) Define Model: DistilRoberta
# DistilRoberta Model with 3 labels for classification
config = AutoConfig.from_pretrained(
    'distilroberta-base',
    num_labels=2
)

model = DistilRobertaClassifier(config)
print(model)


# @title 7) Define Evaluation Metrics
# Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids               # true labels
    preds = pred.predictions.argmax(-1)   # from the predictions (logits), it picks the index of the class with the highest predicted score.

    # Print predicted vs true label distribution
    print("Predicted label distribution:", np.bincount(preds))
    # print("True label distribution:", np.bincount(labels))

    # Returns a diccionary so the 'Trainer' can log them
    return {
        'accuracy': accuracy_score(labels, preds),                    # Fraction of predictions that were exactly correct
        'f1_macro': f1_score(labels, preds, average='macro'),         # F1-score (combination of Precision and Recall) avg equally across all classes (not weighted by class frequency - good for imbalanced data)
        'precision': precision_score(labels, preds, average='macro', zero_division=0), # Precision avg across all classes
        'recall': recall_score(labels, preds, average='macro')       # Recall avg across all classes
    }

# Per-class metrics report
"""    print("\n Classification report:\n", classification_report(labels, preds, digits=3))

    # Plot confusion matrix here
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'High'],
                yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    """

# @title 8) Define Training Arguments
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',         # Directory where model checkpoints and other outputs will be saved during training
    learning_rate=2e-5,             # Common starting LR for transformers
    per_device_train_batch_size=16, # Reasonable for most GPUs
    per_device_eval_batch_size=16,  # Batch for evaluation
    num_train_epochs=5,
    weight_decay=0.01,              # L2 regularization: prevents overfitting by discouraging large weights
    #momentum=0.99,

    lr_scheduler_type="linear",     # Stabilise early training, and the learning rate drops over time, preventing overshooting
    warmup_steps=50,                # gradually increases the learning rate from zero to target learning rate over a certain number of steps at the start of training
    max_grad_norm=1.0,              # to avoid LSTM gradients to explode

    # Training loss
    logging_strategy="steps",
    logging_steps=50,              # Log training loss and metrics every 100 steps

    # Evaluation metrics
    eval_strategy="steps",
    eval_steps=50,                # Evaluate every 100 steps

    # Save model
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,            # Keeps only 1 checkpoint in the output_dir, deleting older ones to control disk usage

    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,

    logging_dir='./logs',           # Directory where log files will be written
    report_to='none',               # disable wanb
)

# @title 9) Define Trainer
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# @title 10) Train and Validate Model
# Train model
trainer.train()

# @title Plot Training Loss and Validation Loss and Accuracy
# Convert logs to DataFrame
log_history = pd.DataFrame(trainer.state.log_history)
eval_logs = log_history.dropna(subset=["eval_loss"])
train_logs = log_history.dropna(subset=["loss"])

# Plot Accuracy
plt.figure(figsize=(10, 5))
if "eval_accuracy" in eval_logs.columns:
    plt.plot(eval_logs["step"], eval_logs["eval_accuracy"], label="Validation Accuracy")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot validation loss and accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_logs["step"], train_logs["loss"], label="Training Loss")
plt.plot(eval_logs["step"], eval_logs["eval_loss"], label="Validation Loss")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

# @title 11) Evaluate Model
test_results = trainer.evaluate(test_dataset)
print("Final Test Set Results:", test_results)

# Plot CLS embeddings
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model.eval()
cls_embeddings = []
labels = []

for batch in val_loader:
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    with torch.no_grad():
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        cls_embed = model._last_cls.cpu().numpy()

    cls_embeddings.append(cls_embed)
    labels.extend(batch["labels"].cpu().numpy())

# Stack all embeddings
cls_embeddings = np.vstack(cls_embeddings)
labels = np.array(labels)

# Standardize
scaler = StandardScaler()
cls_std = scaler.fit_transform(cls_embeddings)

# Reduce
cls_umap = UMAP(n_components=2, random_state=2025).fit_transform(cls_std)
cls_tsne = TSNE(n_components=2, random_state=2025).fit_transform(cls_std)
label_names = np.array(["Low", "High"])[labels]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(x=cls_umap[:, 0], y=cls_umap[:, 1], hue=label_names,
                palette={"Low": "#1f77b4", "High": "#ff7f0e"}, ax=axes[0])
axes[0].set_title("UMAP: CLS Embeddings")
axes[0].set_xlabel("Dim 1")
axes[0].set_ylabel("Dim 2")

sns.scatterplot(x=cls_tsne[:, 0], y=cls_tsne[:, 1], hue=label_names,
                palette={"Low": "#1f77b4", "High": "#ff7f0e"}, ax=axes[1])
axes[1].set_title("t-SNE: CLS Embeddings")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")

plt.tight_layout()
plt.legend(title="Stress Level", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.show()

# Plot Fused Embeddings
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model.eval()
embeddings = []
labels = []

for batch in val_loader:
    # Move batch to GPU
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    # Inside eval loop
    with torch.no_grad():
        outputs = model.distilroberta(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        hidden_state = outputs.last_hidden_state
        cls_embedding = hidden_state[:, 0]

        lstm_output, _ = model.bi_lstm(hidden_state)
        pooled_output = model.attn_pool(lstm_output)
        fused_output = torch.cat([pooled_output, cls_embedding], dim=1)

        embeddings.append(fused_output.cpu())
        labels.extend(batch['labels'].cpu())

embeddings = torch.cat(embeddings).numpy()  # Combine and convert to NumPy

# Standardize embeddings for better visualization
scaler = StandardScaler()
embeddings_std = scaler.fit_transform(embeddings)

# UMAP reduction
umap_reducer = UMAP(n_components=2, random_state=2025)
umap_reduced = umap_reducer.fit_transform(embeddings_std)

# t-SNE reduction
tsne_reducer = TSNE(n_components=2, random_state=2025)
tsne_reduced = tsne_reducer.fit_transform(embeddings_std)

label_names = np.array(["Low", "High"])[np.array(labels)]

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(
    x=umap_reduced[:, 0],
    y=umap_reduced[:, 1],
    hue=label_names,
    palette={"Low": "#1f77b4", "High": "#ff7f0e"},
    ax=axes[0]
)
axes[0].set_title("UMAP: Fused Embeddings")
axes[0].set_xlabel("Dim 1")
axes[0].set_ylabel("Dim 2")

sns.scatterplot(
    x=tsne_reduced[:, 0],
    y=tsne_reduced[:, 1],
    hue=label_names,
    palette={"Low": "#1f77b4", "High": "#ff7f0e"},
    ax=axes[1]
)
axes[1].set_title("t-SNE: Fused Embeddings")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")

plt.tight_layout()
plt.legend(title="Stress Level", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.show()