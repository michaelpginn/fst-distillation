python -m src.run_alignment data/inflection kon --use-med

python -m src.train_alignment_predictor data/inflection kon \
        --batch-size 32 --d-model 64 --epochs 800 --learning-rate 0.002 \
        --weight-decay 0.3 --run-pred --label ablation1

python -m src.train_rnn data/inflection kon --batch-size 8 --hidden-dim 128 \
        --epochs 1000 --learning-rate 0.0002 --dropout 0  \
        --objective transduction --label ablation1

last_run_name=$(< tmp/last-run.txt)

python -m src.extract_fst data/inflection kon --model-id $last_run_name --num-clusters 3779 \
        --min-transition-count 5 --state-split-classifier svm \
        --full-domain-mode sample
