#!/usr/bin/env python3
"""Report names a function reads as globals that its module never defines.

    python tools/check_names.py [src]

Uses Python's own symbol tables, scope by scope: a name that is a parameter of
one function does not count as defined in another.
"""
import builtins
import symtable
import sys
from pathlib import Path


def module_names(table):
    return {s.get_name() for s in table.get_symbols()
            if s.is_assigned() or s.is_imported() or s.is_namespace()}


def global_reads(table):
    for child in table.get_children():
        for symbol in child.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and not symbol.is_declared_global():
                yield child.get_name(), symbol.get_name()
        yield from global_reads(child)


def check(path):
    table = symtable.symtable(path.read_text(), str(path), 'exec')
    defined = module_names(table) | set(dir(builtins))
    return sorted({(scope, name) for scope, name in global_reads(table) if name not in defined})


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'src')
    problems = {p: check(p) for p in sorted(root.rglob('*.py'))}
    problems = {p: v for p, v in problems.items() if v}
    for path, items in problems.items():
        for scope, name in items:
            print(f"{path}: {scope} reads undefined name {name}")
    print(f"{sum(len(v) for v in problems.values())} undefined name(s)")
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
