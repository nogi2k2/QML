# QML - Quantum Sequential Modelling Research

This repository documents research conducted on Quantum Recurrent Neural Networks (QRNNs) and related architectures for quantum machine learning applications in sequence modeling tasks, specifically sentiment analysis and POS tagging. The core objective is to explore the feasibility and performance of quantum models relative to classical recurrent architectures, using standardized datasets, controlled training setups, and a trial at building a pure Quantum RNN

---

## Experimental Overview

Our experimentation is currently in progress, and the results provided/displayed are only temporary. That said, we have conducted a two-phase experimentation as of now:

1. **Classical DL models for establishing a Baseline**  
   18 model configurations using various RNN and LSTM based architectures across different dataset splits, embedding strategies, and training setups to establish a reliable performance benchmark
   
2. **Quantum Model Experiments**  
   Implementation and evaluation of **Quantum RNN** and **Quantum LSTM** cells using Pennylane, with custom quantum circuits designed to handle temporal encoding and prediction in a sequence classification task

---

## Quantum Model Results (Updated)

|     Model      |             Architecture Description              | Test Accuracy | Train Accuracy |      Convergence Behavior       |
|----------------|---------------------------------------------------|---------------|----------------|---------------------------------|
| Quantum RNN v1 | Simple Angle Encoding (1 QNode)                   |     46.5%     |     50.0%      | Stagnated early                 |
| Quantum RNN v2 | Simple Angle Encoding + 2 Repetitions             |     46.5%     |     50.0%      | No improvement from repetition  |
| Quantum RNN v3a| Dense Angle Encoding                              |     53.5%     |     51.6%      | Moderate gain                   |
| Quantum RNN v3b| Dense Angle Encoding + 3 Repetitions              |     53.5%     |     53.5%      | Highest accuracy among QRNNs    |
| Quantum RNN v4 | Amplitude Encoding                                |     53.5%     |     52.8%      | Stable, expressive performance  |
| Quantum RNN v5 | Hybrid (Amp + Dense Angle Encoding)               |     52.0%     |     52.0%      | Slight regression               |
| Quantum LSTM   | Gate-wise QNodes w/ Amplitude + Angle per gate    |     53.5%     |     52.8%      | Most consistent generalization  |

### Key Observations:

- Unlike earlier experiments using 100 training samples, this phase used the **entire dataset**, reducing overfitting
- All quantum models achieved **generalization above 50%**, with **dense angle encodings** (v3a/v3b) and **Quantum LSTM** performing best
- **QNode repetition**, when paired with dense encoding, helped improve training stability
- Quantum LSTM continues to be the **most stable and generalizable QNN**, benefiting from gate-wise separation and hybrid encodings

---

## Classical Baseline Results

To establish a meaningful comparison, the following top-performing classical models were evaluated:

|     Architecture    | Test Accuracy  | Train Accuracy |     Dataset      |
|---------------------|----------------|----------------|------------------|
| Bidirectional RNN   | **63.59%**     | 62.42%         | fb_hi_cg_train2  |
| Manual LSTM         | **61.13%**     | 59.83%         | fb_hi_cg_train2  |
| Two-Layered RNN     | 58.23%         | 60.41%         | fb_hi_cg_train2  |

### Classical Insights:

- **Bidirectional RNNs** achieved the best results, benefiting from forward and backward context awareness
- **Manual LSTMs** with class weights and extended training (up to 800 epochs) performed nearly as well
- **FastText embeddings** significantly outperformed randomly initialized vectors
- Simpler RNNs (e.g., Vanilla RNN) stagnated early and showed poor generalization

---

## Quantum vs Classical: Performance Summary

|         Configuration                     |   Accuracy Range     |
|-------------------------------------------|----------------------|
| Quantum RNN / LSTM                        | **46.5% – 53.5%**    |
| Classical Bidirectional RNN               | **63.6%** (best)     |
| Classical Manual LSTM + Class Weights     | ~61.1%               |

---

## Detailed Reports 

For deeper insights into the model architectures and training statistics refer:
```
results/Classical_Report.pdf
results/Quantum_Report.pdf
```