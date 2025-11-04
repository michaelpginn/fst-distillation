# Motivation
FSTs have traditionally been popular for tasks such as morphological inflection, grapheme-to-phoneme, transliteration, and other noisy channel problems. They provide many benefits, including:
- Fast inference on CPUs (no GPUs required)
- Minimal storage requirements
- Interpretability/editability
- Predictable behavior on OOD inputs

With careful design, FSTs can often achieve near-perfect accuracy and compete with SOTA neural systems, as in [Beemer et al., 2020](https://aclanthology.org/2020.sigmorphon-1.18.pdf). However, creating FSTs is a laborious and painstaking process, requiring expert knowledge of both the target domain and finite-state theory. While algorithms for automatic induction of FSTs exist, they are generally very sample inefficient, requiring huge amounts of labeled pairs to converge to an accurate solution.

# Method
This package provides an algorithm for FST induction using a neural surrogate model (specifically, an Elman RNN) and the state-clustering algorithm of Giles et al. (1991).

1. Align the input and output strings using **CRPAlign**.
2. Train an **alignment prediction model** (seq2seq transformer) to predict the aligned input string from the unaligned string. Use this model to predict aligned inputs for the full domain (or a large sample if infinite).
3. Train an **Elman RNN** on language modeling or binary classification.
4. Extract activations from the RNN for both the training data and full domain as predicted in (2).
5. Cluster activations into *macrostates* and aggregate *microtransitions* for each macrostate.
6. Run the *state splitting algorithm* to split any macrostates with non-deterministic outgoing transitions.

# Usage
```shell
# Python >=3.11
pip install -r requirements.txt

gcc -O3 -Wall -Wextra -fPIC -shared src/crpalign/align.c -o src/crpalign/libalign.so

# Run a full extraction on some dataset
python -m src.sweep <data/inflection> <ceb> --objective <lm>
```

- The first parameter `<data/inflection>` is the path to a folder containing the raw data files.
- The second parameter `<ceb>` is the name of the dataset. Files named `ceb.trn`, `ceb.dev`, and `ceb.tst` must exist. The format is the 2020 SIGMORPHON shared task format (see files for an example).
- The RNN training objective may either be `lm` or `classification`


# Experiments

Analysis
- Impact of hidden state size
- Impact of standardization
- Impact of clustering process/n clusters

Ablations
- State splitting
- Full coverage
- Spec norm

Datasets:
- inflection
- g2p
- transliteration
- historical normalization
