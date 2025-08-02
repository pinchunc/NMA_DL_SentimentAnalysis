# Transformer-based Model: Stress Level Classification in Stories

An end-to-end pipeline using **PyTorch** and **Hugging Face** (DistilRoBERTa) to classify stress levels in stories.

---

## Key Features

1. **Data loading and preprocessing**
2. **Exploratory analysis**
3. **Stratified train/val/test splits**
4. **Custom dataset** with **focal loss**
5. **Custom model architecture**:
   - Bidirectional LSTM
   - Attention pooling
   - CLS token fusion
6. **Training** using Hugging Face's `Trainer`
7. **Evaluation**
8. **Visualizations**:
   - Validation accuracy
   - Training and validation loss
   - Embedding projections
