"""Refresh Lu/Ll/Lt flags without changing the runtime's Unicode coverage.

Requires the Python regex package for Unicode general-category properties.
"""

import argparse
from pathlib import Path
import re

import regex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "cpp_runtime/components/tokenizer/src/unicode-data.cpp"
    text = path.read_text(encoding="utf-8")
    start = text.index("const std::initializer_list<std::pair<uint32_t, uint16_t>> unicode_ranges_flags = {")
    end = text.index("\n};", start)
    section = text[start:end]
    ranges = [(int(cp, 16), int(flags, 16)) for cp, flags in
              re.findall(r"\{0x([0-9A-F]+), 0x([0-9A-F]+)\}", section)]
    if not ranges or ranges[0][0] != 0 or ranges[-1][0] != 0x110000:
        raise ValueError("Incomplete Unicode category table")
    properties = [(regex.compile(r"\p{" + category + "}"), flag)
                  for category, flag in (("Lu", 0x1000), ("Ll", 0x2000), ("Lt", 0x4000))]
    updated = []

    def append(cp, flags):
        if not updated or updated[-1][1] != flags:
            updated.append((cp, flags))

    for (first, flags), (last, _) in zip(ranges, ranges[1:]):
        flags &= 0x0FFF
        if flags & 0x0004:
            for cp in range(first, last):
                subclass = next((flag for prop, flag in properties if prop.fullmatch(chr(cp))), 0)
                append(cp, flags | subclass)
        else:
            append(first, flags)
    updated.append(ranges[-1])
    table = section.splitlines()[0] + "\n" + "\n".join(
        f"{{0x{cp:06X}, 0x{flags:04X}}}," for cp, flags in updated)
    result = text[:start] + table + text[end:]
    if args.check:
        if result != text:
            raise SystemExit("Unicode letter-category flags need regeneration")
    else:
        path.write_text(result, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
