import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(5, 3))

df = pd.read_csv("experiments/ablations.csv")
out = df.melt(
    id_vars="dataset",
    var_name="Condition",
    value_name="Accuracy",
)
g = sns.barplot(
    data=out[out["Condition"] != "— synth. data"],
    x="Accuracy",
    y="dataset",
    hue="Condition",
    palette="dark",
    alpha=0.6,
    ax=ax,
)
sns.despine(bottom=True)
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

f, ax = plt.subplots(figsize=(5, 3))

g = sns.barplot(
    data=out[out["Condition"] != "— CRPAlign"],
    x="Accuracy",
    y="dataset",
    hue="Condition",
    palette="dark",
    alpha=0.6,
    ax=ax,
)
sns.despine(bottom=True)
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
