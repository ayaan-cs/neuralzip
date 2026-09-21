"""Paste results.md (from bench.py) into README.md between the RESULTS marker and the next blank line."""
import re
readme = open("README.md", encoding="utf-8").read()
table = open("results.md", encoding="utf-8").read().strip()
readme = re.sub(r"<!-- RESULTS -->\n(?:\|.*\n)*", "<!-- RESULTS -->\n" + table + "\n", readme)
open("README.md", "w", encoding="utf-8", newline="\n").write(readme)
print(table)
