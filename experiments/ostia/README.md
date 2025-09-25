The core idea for this experiment is entirely black-box distillation. 

1. We train a model (transformer, RNN) on seq2seq inflection.
2. Then, use the trained model to generate predictions for all possible lemma/feature combinations.
3. Finally, use some existing transducer learning algorithm over the full paradigms.

