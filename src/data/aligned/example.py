import re
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from typing import Literal

ALIGNMENT_SYMBOL = "~"
Bigram = tuple[tuple[str, str], tuple[str, str]]


@dataclass
class AlignedStringExample:
    aligned_chars: list[tuple[str, str]]
    features: list[str] | None
    label: bool

    @property
    def aligned_chars_as_strs(self):
        return [f"({p[0]},{p[1]})" for p in self.aligned_chars]

    @property
    def input_string(self):
        return "".join(
            in_char for in_char, _ in self.aligned_chars if in_char != ALIGNMENT_SYMBOL
        )

    @property
    def output_string(self):
        return "".join(
            out_char
            for _, out_char in self.aligned_chars
            if out_char != ALIGNMENT_SYMBOL
        )


def load_examples_from_file(
    path: str | PathLike,
    merge_outputs: Literal["none", "right", "bpe"],
    pretrained_merges: list[Bigram] | None = None,
):
    """Loads `AlignedStringExample` instances from a TSV file.
    If the file includes two columns, treat the second column as a
    feature column (e.g. for inflection)."""
    examples: list[AlignedStringExample] = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split("\t")
            if len(row) == 1:
                [chars] = row
                features = None
            elif len(row) == 2:
                [chars, features] = row
                features = [f"[{f}]" for f in features.split(";")]
            else:
                raise ValueError("File must be TSV with 1-2 columns")
            char_pairs: list[tuple[str, str]] = re.findall(r"\((.*?),(.*?)\)", chars)
            char_pairs = [("<sep>", "<sep>")] + char_pairs + [("<sink>", "<sink>")]
            examples.append(AlignedStringExample(char_pairs, features, label=True))

    if merge_outputs == "right":
        return merge_outputs_right(examples), None
    elif merge_outputs == "bpe":
        return merge_outputs_bpe(examples, pretrained_merges)
    else:
        return examples, None


def merge_outputs_right(examples: list[AlignedStringExample]):
    """Removes epsilon input pairs by delaying outputs.

    For example, (~,r)(~,e)(d,d)(~,x)(o,o) would become (d:red)(o,xo)
    """

    def process_example(example: AlignedStringExample):
        new_aligned_chars: list[tuple[str, str]] = []

        out_char_buffer = []
        for in_char, out_char in example.aligned_chars:
            if in_char == ALIGNMENT_SYMBOL:
                out_char_buffer.append(out_char)
            elif len(out_char_buffer) > 0:
                new_aligned_chars.append((in_char, "".join(out_char_buffer) + out_char))
                out_char_buffer = []
            else:
                new_aligned_chars.append((in_char, out_char))

        if len(out_char_buffer) > 0:
            last_in, last_out = new_aligned_chars[-1]
            new_aligned_chars[-1] = (last_in, last_out + "".join(out_char_buffer))

        example.aligned_chars = new_aligned_chars
        return example

    return [process_example(ex) for ex in examples]


def merge_outputs_bpe(
    examples: list[AlignedStringExample], pretrained_merges: list[Bigram] | None
):
    """Removes epsilon input pairs by merging outputs either left or right.
    Follows BPE, identifying globally common edges and merging in order.

    Returns a list of merges
    """

    def count_bigrams(exs: list[AlignedStringExample]):
        counts: Counter[Bigram] = Counter()
        for ex in exs:
            for i in range(len(ex.aligned_chars) - 1):
                first_pair = ex.aligned_chars[i]
                second_pair = ex.aligned_chars[i + 1]
                if (
                    first_pair[0] == ALIGNMENT_SYMBOL
                    and second_pair[0] != ALIGNMENT_SYMBOL
                ) or (
                    first_pair[0] != ALIGNMENT_SYMBOL
                    and second_pair[0] == ALIGNMENT_SYMBOL
                ):
                    counts[(first_pair, second_pair)] += 1
        return counts

    def replace_bigram(exs: list[AlignedStringExample], bigram: Bigram):
        new_examples: list[AlignedStringExample] = []
        for ex in exs:
            new_aligned_chars = []
            last_merged = False
            for i in range(len(ex.aligned_chars) - 1):
                if last_merged:
                    last_merged = False
                    continue
                first_pair = ex.aligned_chars[i]
                second_pair = ex.aligned_chars[i + 1]
                if (first_pair, second_pair) == bigram:
                    if first_pair[0] == ALIGNMENT_SYMBOL:
                        new_in = second_pair[0]
                    elif second_pair[0] == ALIGNMENT_SYMBOL:
                        new_in = first_pair[0]
                    else:
                        raise ValueError()
                    new_aligned_chars.append((new_in, first_pair[1] + second_pair[1]))
                    last_merged = True
                else:
                    new_aligned_chars.append(first_pair)
                    last_merged = False
            if not last_merged:
                new_aligned_chars.append(ex.aligned_chars[-1])
            new_examples.append(
                AlignedStringExample(
                    aligned_chars=new_aligned_chars,
                    features=ex.features,
                    label=ex.label,
                )
            )
        return new_examples

    if not pretrained_merges:
        merges: list[Bigram] = []
        counts = count_bigrams(examples)
        while counts.total() > 0:
            next_merge = counts.most_common(1)[0][0]
            merges.append(next_merge)
            examples = replace_bigram(examples, next_merge)
            counts = count_bigrams(examples)
    else:
        merges = pretrained_merges
        for merge in merges:
            examples = replace_bigram(examples, merge)
    return examples, merges
