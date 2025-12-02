def tokenize(
    s: str | list[str],
    post_diacritics: set[str] = {"⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"},
):
    """Splits a string into chars.
    Any symbols in `post_diacritics` will be attached to the previous character as a single symbol."""
    if isinstance(s, list):
        return s

    chars = list(s)
    new_chars: list[str] = []
    for c in chars:
        if c in post_diacritics:
            new_chars[-1] += c
        else:
            new_chars.append(c)
    return new_chars
