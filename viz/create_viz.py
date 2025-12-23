import argparse
from pathlib import Path

import pandas as pd

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("task", choices=["inflection", "g2p", "histnorm"])
args = parser.parse_args()

viz_path = Path(__file__).parent
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
    lang = sweep.name.split(".")[-1]
    our_results[lang] = best_run.summary_metrics["test"]["f1"]

results = pd.DataFrame(
    [ostia_results, ostia_dd_results, our_results], index=["OSTIA", "OSTIA-DD", "Ours"]
)
if beemer_results is not None:
    results = pd.concat([beemer_results, results])


# Create LaTeX
def fmt(x):
    if pd.isna(x):
        return ""
    return f"{x:.3f}"


with open(viz_path / "main.tex", "w") as f:
    f.write(
        "\\begin{table}[htb]\n"
        "\\begin{tabularx}{\\columnwidth}{l | X | X X X }\n"
        "& \\rot{Human Expert} & \\rot{OSTIA} & \\rot{OSTIA-DD} & \\rot{Ours} \\\\ \n"
        "\\hline \n"
    )
    for row in results.T.iterrows():
        lang = row[0]
        beemer = fmt(row[1]["Human Expert"])
        ostia = fmt(row[1]["OSTIA"])
        ostia_dd = fmt(row[1]["OSTIA-DD"])
        ours = "\\textbf{" + fmt(row[1]["Ours"]) + "}"
        f.write(f"{lang} & {beemer} & {ostia} & {ostia_dd} & {ours} \\\\\n")
    f.write(
        "\\end{tabularx}\n"
        "\\caption{Accuracy of learned transducers for morphological inflection datasets on held-out test set. Our system produces far more accurate transducers than the OSTIA-based algorithms on every language, and often nearly as accurate as expert-crafted FSTs from \\citet{beemer-etal-2020-linguist}.}\n"
        "\\label{tab:inflection}\n"
        "\\end{table}"
    )
