# Motivation
FSTs have traditionally been popular for tasks such as morphological inflection, grapheme-to-phoneme, transliteration, and other noisy channel problems. They provide many benefits, including:
- Fast inference on CPUs (no GPUs required)
- Minimal storage requirements
- Interpretability/editability
- Predictable behavior on OOD inputs

With careful design, FSTs can often achieve near-perfect accuracy and compete with SOTA neural systems, as in [Beemer et al., 2020](https://aclanthology.org/2020.sigmorphon-1.18.pdf). However, creating FSTs is a laborious and painstaking process, requiring expert knowledge of both the target domain and finite-state theory. While algorithms for automatic induction of FSTs exist, they are generally very sample inefficient, requiring huge amounts of labeled pairs to converge to an accurate solution.

Meanwhile, neural models have achieved impressive performance on these sorts of tasks, even with limited data. We seek to leverage this behavior for FST construction through **knowledge distillation**, where a trained neural model becomes a teacher in order to train an FST, which enjoys the benefits mentioned earlier.

# Goals

Our initial work consists of two main goals:

1. We will compare the **sample efficiency** of **neural and finite-state learning algorithms**, by studying how performance of these systems varies with the size of the provided training set and the amount of data needed to converge.
2. We will train high-quality neural models and perform **knowledge distillation to FSTs** by sampling novel forms from the neural model in order to vastly increase the amount of labeled data. To this end, we will study various finite-state induction algorithms, also including those which utilize negative samples and uncertainty estimates.

# Usage
```shell
pip install -r requirements.txt
python -m exp1-student-teacher.train_transformer
```
