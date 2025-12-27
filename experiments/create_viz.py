import argparse
from pathlib import Path

import pandas as pd

import wandb
from src.data.unaligned.example import load_examples_from_file
from src.paths import create_paths

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=["inflection", "g2p", "histnorm"])
args = parser.parse_args()

viz_path = Path(__file__).parent / "viz"
if args.task == "inflection":
    beemer_results = pd.read_csv(viz_path / "beemer.csv")
    beemer_results.index = ["Human Expert"]
else:
    beemer_results = None

api = wandb.Api()

# OSTIA baselines
ostia_results = {}
ostia_dd_results = {}
for run in api.runs(path="lecs-general/fst-distillation.ostia"):
    task, lang = run.config["paths"]["identifier"].split(".")
    if task != args.task or run.state != "finished":
        continue
    score = run.summary_metrics["test"]["f1"]
    if run.config["order"] == "lex":
        ostia_results[lang] = score
    elif run.config["order"] == "dd":
        ostia_dd_results[lang] = score
    else:
        raise ValueError()


# Ours
sweeps = api.project(
    name="fst-distillation.extraction.v2", entity="lecs-general"
).sweeps()
our_results = {}
for sweep in sweeps:
    best_run = sweep.best_run()
    assert best_run
    task, lang = sweep.name.split(".")
    if task != args.task:
        continue
    our_results[lang] = best_run.summary_metrics["test"]["f1"]

results = pd.DataFrame(
    [ostia_results, ostia_dd_results, our_results], index=["OSTIA", "OSTIA-DD", "Ours"]
)
if beemer_results is not None:
    results = pd.concat([beemer_results, results])


if args.task == "histnorm":
    nochange_results = {}
    for lang in our_results:
        paths = create_paths(
            data_folder="data/histnorm",
            dataset=lang,
            has_features=False,
            output_split=False,
            merge_outputs="right",
            models_folder=None,
        )
        test_examples = load_examples_from_file(
            paths["test"], paths["has_features"], paths["output_split_into_chars"]
        )
        nochange_results[lang] = len(
            [ex for ex in test_examples if ex.input_string == ex.output_string]
        ) / len(test_examples)
    results = pd.concat(
        [pd.DataFrame([nochange_results], index=["No Change"]), results]
    )


# Create LaTeX
def fmt(x):
    if pd.isna(x):
        return ""
    return f"{x:.3f}"


with open(viz_path / f"{args.task}.tex", "w") as f:
    fields = ["\\rot{OSTIA}", "\\rot{OSTIA-DD}", "\\rot{Ours}"]
    if args.task == "inflection":
        fields = ["\\rot{Human Expert}"] + fields
    if args.task == "histnorm":
        fields = ["\\rot{No Change}"] + fields
    f.write(
        "\\begin{table}[htb]\n"
        "\\begin{tabularx}{\\columnwidth}{l | X | X X X }\n"
        "\\toprule"
        f"& {' & '.join(f for f in fields)} \\\\ \n"
        "\\midrule \n"
    )
    for row in results.T.iterrows():
        lang = row[0]
        scores = []
        if "Human Expert" in row[1]:
            scores.append(fmt(row[1]["Human Expert"]))
        if "No Change" in row[1]:
            scores.append(fmt(row[1]["No Change"]))
        scores.append(fmt(row[1]["OSTIA"]))
        scores.append(fmt(row[1]["OSTIA-DD"]))
        scores.append("\\textbf{" + fmt(row[1]["Ours"]) + "}")
        scores_str = " & ".join(scores)
        f.write(f"{lang} & {scores_str} \\\\\n")
    if args.task == "inflection":
        task_name = "morphological inflection"
    elif args.task == "g2p":
        task_name = "grapheme-to-phoneme"
    else:
        task_name = "historical normalization"
    f.write(
        "\\bottomrule"
        "\\end{tabularx}\n"
        f"\\caption{{Accuracy of learned transducers for {task_name} datasets on held-out test set.}}\n"
        f"\\label{{tab:{args.task}}}\n"
        "\\end{table}"
    )
