# QML - Quantum Recurrent Neural Network Research

This repository archives research conducted on Quantum Recurrent Neural Networks (QRNNs) and related architectures for quantum machine learning applications in sequence modeling tasks

## Experimental Results Overview

We conducted 18 experiments comparing various RNN architectures for sequence modeling, evaluating model performance across different classical configurations

### Best Performing Models

|   Model Name   |     Architecture     | Test Accuracy  | Train Accuracy |     Dataset      |
|----------------|----------------------|----------------|----------------|------------------|
| model_bidir    | Bidirectional RNN    | 63.59%         | 62.42%         | fb_hi_cg_train2  |
| model_bidir    | Bidirectional RNN    | 61.14%         | 59.83%         | fb_hi_cg_train2  |
| model_2layer   | Two-Layered RNN      | 58.23%         | 60.41%         | fb_hi_cg_train2  |
| model_apiLSTM  | LSTM                 | 57.90%         | 55.79%         | fb_hi_cg_train2  |

## Key Findings

### Top Performing Model: Bidirectional RNN
- Achieved highest test accuracy (**63.59%** - test, **61.14%** - train)
- Outperformed all unidirectional and shallow architectures
- Shows strong generalization (test ≈ train accuracy)

### Architecture Performance (Test Accuracy)

|      Model         |   Accuracy   |
|--------------------|--------------|
| Bidirectional RNN  | 61–64%       |
| Two-Layered RNN    | ~58%         |
| Vanilla RNN        | 27–40%       |

### Dataset Impact
- `fb_hi_cg_train2` enabled higher accuracy (20–64%)
- `hi_dataset/CR` yielded more stable but lower results (23–47%)

### Training Configuration Insights
- **FastText embeddings** outperformed random embeddings 
- **64 hidden dimensions** worked best
- **Extended training (500–800 epochs)** critical for performance
- **Class weighting** helped with data imbalance

---

## Performance Summary

|              Setup                      |   Accuracy   |
|-----------------------------------------|--------------|
| Random Embeddings + Vanilla RNN         | ~24–40%      |
| FastText + LSTM                         | ~40–50%      |
| FastText + BiRNN + Extended Training    | **~60–64%**  |

> ✅ **Conclusion**: Combining Bidirectional RNNs, FastText embeddings, and long training yields the best results in this setup
