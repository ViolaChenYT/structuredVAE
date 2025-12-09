import json
import gzip
from itertools import product


def load_paths_dict(paths_dict_file):
    """Load the paths dictionary from the compressed JSON file."""
    with gzip.open(paths_dict_file, 'rt') as f:
        return json.load(f)


def expand_choices(items, sep="/", strip=True, dedup=False):
    """
    items: list[str] possibly containing 'a/b' meaning choose one
    returns: list[list[str]] of all expanded lists
    """
    # turn each item into a list of options
    opts = []
    for s in items:
        parts = s.split(sep)
        if strip:
            parts = [p.strip() for p in parts]
        # if no sep, keep as single option; if sep, use all parts
        opts.append(parts if len(parts) > 1 else [parts[0]])
    # Cartesian product over positions
    combos = [list(choice) for choice in product(*opts)]
    if dedup:  # remove duplicate lists if options had duplicates
        seen = set()
        uniq = []
        for c in combos:
            t = tuple(c)
            if t not in seen:
                seen.add(t); uniq.append(c)
        return uniq
    return combos


def is_prefix_match(prefix: str, s: str, wildcard: str = "x") -> bool:
    """Return True iff prefix (with 'x' as single-char wildcard) matches the start of s."""
    if len(prefix) > len(s):
        return False
    for a, b in zip(prefix, s):
        if a != wildcard and b!=wildcard and a != b:
            return False
    return True


def is_prefix_chain_wild(L, wildcard: str = "x") -> bool:
    """Check L[i] is a wildcard-prefix of L[i+1] for all i."""
    return all(is_prefix_match(L[i], L[i+1], wildcard) for i in range(len(L) - 1))


def label_order_index(label_list, sep="/", wildcard="x"):
    # Find the first expanded path whose length-lex order forms a valid wildcard-prefix chain
    path = next(
        (sorted(p, key=lambda s: (len(s), s))
         for p in expand_choices(label_list, sep=sep)
         if is_prefix_chain_wild(sorted(p, key=lambda s: (len(s), s)), wildcard)),
        None
    )
    if path is None:
        return {}, 0
    label_idx = {lab: len(lab) for lab in path}
    #print(label_idx)
    #print(path)
    missing_node = int(len(label_idx) < len(path[-1])-len(path[0])+1)  # keep original rule
    # Map each original token like "a/b" to the first option that appears in label_idx
    origin = {}
    for lab in label_list:
        val = next((label_idx[o] for o in lab.split(sep) if o in label_idx), None)
        if val is not None:
            origin[lab] = val
    return origin, missing_node
