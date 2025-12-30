import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")

# Ablation 1
f, ax = plt.subplots(figsize=(5, 2))
df = pd.read_csv("experiments/ablations.csv")
df["dataset"] = df["dataset"].str.replace("#", "\n")
out = df.melt(
    id_vars="dataset",
    var_name="Condition",
    value_name="Accuracy",
)
ax.tick_params(
    axis="x",
    which="both",
    bottom=True,
    top=False,
    length=4,
    width=1,
    labelsize=9,
)
g = sns.barplot(
    data=out[out["Condition"].isin(["Base", "– CRPAlign"])],
    x="Accuracy",
    y="dataset",
    hue="Condition",
    palette="deep",
    edgecolor="black",
    linewidth=1,
    alpha=1,
    ax=ax,
)
sns.despine(
    bottom=False,
)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set(ylabel=None)
ax.set_xlim(0, 1)
leg = ax.legend(loc="lower right")
for text in leg.get_texts():
    text.set_fontweight("bold")
plt.savefig("experiments/viz/ablation1.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close(f)

# Ablation 2
f, ax = plt.subplots(figsize=(5, 2))
ax.tick_params(
    axis="x",
    which="both",
    bottom=True,
    top=False,
    length=4,
    width=1,
    labelsize=9,
)
g = sns.barplot(
    data=out[out["Condition"].isin(["Base", "– synth. data"])],
    x="Accuracy",
    y="dataset",
    hue="Condition",
    palette="deep",
    edgecolor="black",
    linewidth=1,
    alpha=1,
    ax=ax,
)
sns.despine(bottom=False)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set(ylabel=None)
ax.set_xlim(0, 1)
leg = ax.legend(loc="lower right")
for text in leg.get_texts():
    text.set_fontweight("bold")
plt.savefig("experiments/viz/ablation2.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Ablation 3
f, ax = plt.subplots(figsize=(5, 2))
ax.tick_params(
    axis="x",
    which="both",
    bottom=True,
    top=False,
    length=4,
    width=1,
    labelsize=9,
)
g = sns.barplot(
    data=out[out["Condition"].isin(["Base", "LM", "Class."])],
    x="Accuracy",
    y="dataset",
    hue="Condition",
    palette="deep",
    edgecolor="black",
    linewidth=1,
    alpha=1,
    ax=ax,
)
sns.despine(bottom=False)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set(ylabel=None)
ax.set_xlim(0, 1)
leg = ax.legend(loc="lower right")
for text in leg.get_texts():
    text.set_fontweight("bold")
plt.savefig("experiments/viz/ablation3.pdf", format="pdf", bbox_inches="tight")
plt.show()
