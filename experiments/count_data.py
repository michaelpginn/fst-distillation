"""Count examples for each dataset"""

from src.data.unaligned.example import load_examples_from_file
from src.paths import create_paths

datasets = {
    "inflection": [
        "aka",
        "ceb",
        "crh",
        "czn",
        "dje",
        "gaa",
        "izh",
        "kon",
        "lin",
        "mao",
        "mlg",
        "nya",
        "ood",
        "orm",
        "ote",
        "san",
        "sot",
        "swa",
        "syc",
        "tgk",
        "tgl",
        "xty",
        "zpv",
        "zul",
    ],
    "g2p": [
        "ady",
        "arm",
        "bul",
        "dut",
        "fre",
        "geo",
        "gre",
        "hin",
        "hun",
        "ice",
        "jpn",
        "kor",
        "lit",
        "rum",
        "vie",
    ],
    "histnorm": ["deu", "hun", "isl", "por", "slv", "spa", "swe"],
}

for key in datasets:
    rows = []
    for lang in datasets[key]:
        paths = create_paths(
            data_folder=f"data/{key}",
            dataset=lang,
            has_features=key == "inflection",
            output_split=key == "g2p",
            merge_outputs="bpe" if key == "inflection" else "right",
            models_folder=None,
        )
        rows.append(
            [lang, ""]
            + [
                format(
                    len(
                        load_examples_from_file(
                            paths[k],
                            paths["has_features"],
                            paths["output_split_into_chars"],
                        )
                    ),
                    ",",
                )
                for k in ["train", "eval", "test"]
            ]
        )
    for row in rows:
        print(" & ".join(row) + " \\\\")
    print("\n\n")
