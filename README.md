# AI6127 - Deep Neural Networks for Natural Language Processing

## Assignment 2: Neural Machine Translation

### Overview

This repository contains the codebase for Assignment 2 of the AI6127 Deep Neural Networks for Natural Language Processing course. The assignment focuses on implementing and evaluating sequence-to-sequence models with attention mechanisms for neural machine translation (NMT) tasks, specifically translating between French and English languages.

### Installation

To set up the environment for this project, run:

```bash
# Clone the repository
git clone <repository-url>
cd codebase

# Create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
codebase/
├── data/                   # Dataset files
├── Assignment2_V2.ipynb    # Jupiter note file for the assignment           
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Dataset

The project uses the French-English parallel corpus from the Tatoeba Project, available through Manythings.org. This dataset contains a collection of sentence pairs in French and English.

- **Source**: Downloaded from http://www.manythings.org/anki/fra-eng.zip
- **Format**: Tab-separated text file with French-English sentence pairs
- **Size**: Initially contains 232,736 sentence pairs, filtered down to approximately 22,907 pairs after preprocessing
- **Preprocessing**:
  - Normalization (lowercase, removing accents, trimming whitespace)
  - Filtering for pairs where both sentences are shorter than MAX_LENGTH (50 tokens)
  - Construction of vocabulary dictionaries for both languages
  - Train-test split (90% training, 10% testing) using scikit-learn's train_test_split function
  
The dataset is organized using custom Lang classes that manage word indexing and vocabulary construction. Special tokens include SOS (Start of Sentence, index 0) and EOS (End of Sentence, index 1).

### Model Architecture

The assignment implements and compares five different sequence-to-sequence architectures for the neural machine translation task:

1. **Baseline GRU Model (Task 1)**
   - Encoder: Single-layer GRU with embedding layer
   - Decoder: Single-layer GRU with embedding layer and output softmax
   - Training: Teacher forcing with 0.5 probability

2. **LSTM Model (Task 2)**
   - Encoder: Modified with LSTM cells instead of GRU
   - Decoder: Modified with LSTM cells instead of GRU
   - Maintains same overall architecture as baseline

3. **Bidirectional LSTM Encoder with GRU Decoder (Task 3)**
   - Encoder: Bidirectional LSTM that processes sequences in both directions
   - Decoder: Standard GRU decoder
   - Handles bidirectional hidden states by combining outputs from both directions

4. **Attention-based GRU Model (Task 4)**
   - Encoder: Standard GRU encoder
   - Decoder: GRU with attention mechanism
   - Attention: Calculates attention weights over encoder outputs
   - Combines context vector with decoder input for improved translation

5. **Transformer Encoder with GRU Decoder (Task 5)**
   - Encoder: Transformer encoder with multi-head self-attention
   - Positional encoding to capture sequence order information
   - Decoder: Standard GRU decoder
   - Integrates modern transformer architecture with traditional RNN decoding

All models use embedding layers to convert word indices to dense vectors, and employ negative log-likelihood loss for training. Optimization is performed using SGD with a learning rate of 0.01.

### Results

The performance of each model was evaluated using ROUGE-1 and ROUGE-2 scores, which measure the overlap of unigrams and bigrams between the generated translations and reference translations.

| Model                        | ROUGE-1 F-measure | ROUGE-2 F-measure | Training Time |
|------------------------------|-------------------|-------------------|---------------|
| Baseline GRU                 | 0.613             | 0.437             | ~3 hours      |
| LSTM                         | 0.617             | 0.438             | ~3 hours      |
| Bidirectional LSTM + GRU     | 0.618             | 0.435             | ~3.5 hours    |
| GRU with Attention           | 0.597             | 0.408             | ~4 hours      |
| Transformer Encoder + GRU    | 0.207             | 0.121             | ~5 hours      |

Key findings:
- The bidirectional LSTM encoder with GRU decoder achieved the highest ROUGE-1 score, suggesting better capture of word-level translation accuracy
- The LSTM model performed slightly better than the baseline GRU in both metrics
- The attention mechanism, while theoretically more sophisticated, showed slightly lower performance on this specific dataset
- The transformer encoder with GRU decoder showed significantly lower performance, potentially due to optimization challenges or architectural mismatch

Qualitative evaluation through manual inspection of translations showed that all models were capable of producing coherent translations for simple sentences, but the bidirectional model handled complex grammatical structures more effectively.

### Task Implementation

The codebase implements each task as follows:

1. **Task 1 (Baseline GRU)**: 
   - Implemented EncoderRNN and Decoder classes
   - Train using trainIters function with teacher forcing
   - Evaluated with ROUGE metrics and random sample translations

2. **Task 2 (LSTM)**: 
   - Created EncoderLSTM and DecoderLSTM classes, replacing GRU cells with LSTM cells
   - Used the same training and evaluation framework as the baseline

3. **Task 3 (Bidirectional LSTM)**:
   - Implemented EncoderBiLSTM with bidirectional=True parameter
   - Modified the decoder to handle the concatenated forward and backward hidden states
   - Training and evaluation follow the same pattern as previous models

4. **Task 4 (Attention)**:
   - Created AttentionDecoder class with an additional attention layer
   - Implemented attention scoring mechanism to focus on relevant parts of encoder outputs
   - Modified training process with trainIters_attention to accommodate the attention mechanism

5. **Task 5 (Transformer)**:
   - Implemented PositionalEncoding class for sequence position information
   - Created EncoderTransformer using nn.TransformerEncoderLayer and nn.TransformerEncoder
   - Integrated with the existing GRU decoder and evaluation framework

Each model implementation was followed by training, saving model weights, evaluation on test data, and reporting ROUGE scores for comparative analysis.

