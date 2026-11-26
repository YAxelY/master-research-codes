"""Read KEY=value lines from a .env file (comments, blank lines and `export` allowed)."""
import re
from pathlib import Path

LINE = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$')


def read_env_file(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        match = LINE.match(line)
        if match and not line.lstrip().startswith('#'):
            values[match.group(1)] = match.group(2).strip('"\'')
    return values
