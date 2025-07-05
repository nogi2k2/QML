# QML - Quantum Sequential Modelling Research

This repository documents research conducted on Quantum Recurrent Neural Networks (QRNNs) and related architectures for quantum machine learning applications in sequence modeling tasks, specifically sentiment analysis and POS tagging. The core objective is to explore the feasibility and performance of quantum models relative to classical recurrent architectures, using standardized datasets, controlled training setups, and a trial at building a pure Quantum RNN.

---

## Experimental Overview

Our experimentation is currently in progress, and the results provided/displayed are only temporary. That said, we have conducted a two-phase experimentation as of now:

1. **Classical DL models for establishing a Baseline**  
   18 model configurations using various RNN and LSTM based architectures across different dataset splits, embedding strategies, and training setups to establish a reliable performance benchmark.
   
2. **Quantum Model Experiments**  
   Implementation and evaluation of **Quantum RNN** and **Quantum LSTM** cells using Pennylane, with custom quantum circuits designed to handle temporal encoding and prediction in a sequence classification task.

---

Key results:  
## Quantum Model Results

|     Model     |                 Architecture                         | Test Accuracy  | Train Accuracy |    Convergence Behavior      |
|---------------|------------------------------------------------------|----------------|----------------|------------------------------|
| Quantum RNN   | Amplitude + Angle Encoding, 1 QNode per step         |   **46.5%**    |     ~53%       | Stagnated after 10–15 epochs |
| Quantum LSTM  | Gate-wise QNodes (input, forget, candidate, output)  |   **46.5%**    |     ~53%       | Stagnated after 10–15 epochs |

### Key Observations:

- Both quantum models converged early around **random guess-level performance** (~50%), with little gain through extended training.
- **Quantum LSTM**, despite being more complex (4 QNodes for each gate per time step), **did not outperform** the simpler Quantum RNN.
- Low prediction confidence and probability saturation (only 3–5 unique outputs) suggest model convergence failure or vanishing gradient dynamics.
- **Bias towards the negative class** was observed across confusion matrices.

---

## Classical Baseline Results

To establish a meaningful comparison, the following top-performing classical models were evaluated:

|     Architecture    | Test Accuracy  | Train Accuracy |     Dataset      |
|---------------------|----------------|----------------|------------------|
| Bidirectional RNN   | **63.59%**     | 62.42%         | fb_hi_cg_train2  |
| Manual LSTM         | **61.13%**     | 59.83%         | fb_hi_cg_train2  |
| Two-Layered RNN     | 58.23%         | 60.41%         | fb_hi_cg_train2  |

### Classical Insights:

- **Bidirectional RNNs** achieved the best results, benefiting from forward and backward context awareness.
- **Manual LSTMs** with class weights and extended training (up to 800 epochs) performed nearly as well.
- **FastText embeddings** significantly outperformed randomly initialized vectors.
- Simpler RNNs (e.g., Vanilla RNN) stagnated early and showed poor generalization.

---

## Quantum vs Classical: Performance Summary

|         Configuration                     |   Accuracy Range     |
|-------------------------------------------|----------------------|
| Quantum RNN / LSTM                        | **46.5%** (stagnant) |
| Classical Bidirectional RNN               | **63.6%** (best)     |
| Classical Manual LSTM + Class Weights     | ~61.1%               |

---

## Detailed Reports 

For deeper insights into the model architectures, training statistics:
```
results/Classical_Report.pdf
results/Quantum_Report.pdf
```