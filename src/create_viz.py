from pathlib import Path

import pandas as pd

import wandb

# TODO: Adapt for non inflection task

viz_path = Path(__file__).parent.parent / "viz"
results = pd.read_csv(viz_path / "beemer.csv")
results.index = ["beemer"]

api = wandb.Api()

# OSTIA baselines
ostia_results = {}
ostia_dd_results = {}
for run in api.runs(path="lecs-general/fst-distillation.ostia"):
    lang = run.config["paths"]["identifier"].split(".")[-1]
    score = run.summary_metrics["test"]["f1"]
    if run.config["order"] == "lex":
        ostia_results[lang] = score
    elif run.config["order"] == "dd":
        ostia_dd_results[lang] = score
    else:
        raise ValueError()
results.loc["OSTIA"] = ostia_results
results.loc["OSTIA-DD"] = ostia_dd_results

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
results.loc["Ours"] = our_results


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
        beemer = fmt(row[1]["beemer"])
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
