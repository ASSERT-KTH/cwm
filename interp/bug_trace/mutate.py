"""Controlled bug injection for CRUXEval programs.

Applies AST-level mutations and validates that the mutation changes the output
when executed on the provided input.

Usage:
    python -m interp.bug_trace.mutate --dry_run 5 --seed 42
    python -m interp.bug_trace.mutate --output_path interp/bug_trace/data/pairs.json --seed 42
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
import logging
import random
import signal
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AST mutation transformers
# ---------------------------------------------------------------------------


class _FirstMutationTransformer(ast.NodeTransformer):
    """Base: mutates the first eligible node, then stops (visited flag)."""

    def __init__(self) -> None:
        self.mutated = False


class OffByOneMutator(_FirstMutationTransformer):
    """Adjusts integer/float constants by +1 or -1 in BinOp / comparison / Subscript contexts."""

    def __init__(self, delta: int = 1) -> None:
        super().__init__()
        self.delta = delta

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not self.mutated and isinstance(node.value, int) and node.value != 0:
            self.mutated = True
            return ast.Constant(value=node.value + self.delta)
        return node


class WrongOperatorMutator(_FirstMutationTransformer):
    """Swaps binary operators in arithmetic expressions."""

    _SWAPS: dict[type, type] = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Add,
        ast.FloorDiv: ast.Mult,
        ast.Mod: ast.Add,
    }

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if not self.mutated:
            replacement = self._SWAPS.get(type(node.op))
            if replacement is not None:
                self.mutated = True
                new_node = copy.copy(node)
                new_node.op = replacement()
                return new_node
        return self.generic_visit(node)


class ConditionFlipMutator(_FirstMutationTransformer):
    """Wraps the test of an if/while with `not`."""

    def _flip(self, test: ast.expr) -> ast.expr:
        return ast.UnaryOp(op=ast.Not(), operand=test)

    def visit_If(self, node: ast.If) -> ast.AST:
        if not self.mutated:
            self.mutated = True
            new_node = copy.copy(node)
            new_node.test = self._flip(node.test)
            return new_node
        return self.generic_visit(node)

    def visit_While(self, node: ast.While) -> ast.AST:
        if not self.mutated:
            self.mutated = True
            new_node = copy.copy(node)
            new_node.test = self._flip(node.test)
            return new_node
        return self.generic_visit(node)


class WrongComparatorMutator(_FirstMutationTransformer):
    """Swaps comparison operators."""

    _SWAPS: dict[type, type] = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
    }

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if not self.mutated and node.ops:
            replacement = self._SWAPS.get(type(node.ops[0]))
            if replacement is not None:
                self.mutated = True
                new_node = copy.copy(node)
                new_node.ops = [replacement()] + node.ops[1:]
                return new_node
        return self.generic_visit(node)


class WrongVariableMutator(ast.NodeTransformer):
    """Swap the returned local variable for a different local variable in the same function.

    Requires execution reasoning: to know which variable is the intended result, the
    reader must mentally trace the computation — surface-level syntactic reading is not
    enough to distinguish e.g. ``return result`` vs ``return total``.
    """

    def __init__(self) -> None:
        self.mutated = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_func(node)

    def _visit_func(self, node: ast.AST) -> ast.AST:
        if self.mutated:
            return node

        # Collect all local variable names assigned anywhere in this function
        assigned: list[str] = []
        seen: set[str] = set()
        for n in ast.walk(node):
            targets: list[ast.expr] = []
            if isinstance(n, ast.Assign):
                targets = list(n.targets)
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                targets = [n.target]
            elif isinstance(n, ast.For):
                targets = [n.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id not in seen:
                    seen.add(t.id)
                    assigned.append(t.id)

        # Find the first return statement that returns a bare Name
        for n in ast.walk(node):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Name):
                ret_name = n.value.id
                candidates = [v for v in assigned if v != ret_name]
                if candidates:
                    self.mutated = True
                    # Modify in-place (safe: we stop after this)
                    n.value = ast.Name(id=candidates[0], ctx=ast.Load())
                    break

        return node


class DeletedAccumulatorMutator(_FirstMutationTransformer):
    """Remove the first augmented-assignment accumulation step inside a for/while loop.

    Example: removes ``result += x`` from a loop body, leaving the initialiser
    (``result = 0``) intact.  The model must simulate the loop to detect that the
    accumulation is missing — a pure read of variable names does not reveal the bug.
    """

    _AUG_OPS = (ast.Add, ast.Sub, ast.Mult, ast.BitOr, ast.BitAnd)

    def _remove_first_augassign(self, body: list[ast.stmt]) -> list[ast.stmt] | None:
        """Return new body with first matching AugAssign removed, or None."""
        new_body: list[ast.stmt] = []
        removed = False
        for stmt in body:
            if (
                not removed
                and isinstance(stmt, ast.AugAssign)
                and isinstance(stmt.op, self._AUG_OPS)
            ):
                removed = True
                continue
            new_body.append(stmt)
        if removed:
            return new_body or [ast.Pass()]
        return None

    def visit_For(self, node: ast.For) -> ast.AST:
        if self.mutated:
            return self.generic_visit(node)
        new_body = self._remove_first_augassign(node.body)
        if new_body is not None:
            self.mutated = True
            new_node = copy.copy(node)
            new_node.body = new_body
            return new_node
        return self.generic_visit(node)

    def visit_While(self, node: ast.While) -> ast.AST:
        if self.mutated:
            return self.generic_visit(node)
        new_body = self._remove_first_augassign(node.body)
        if new_body is not None:
            self.mutated = True
            new_node = copy.copy(node)
            new_node.body = new_body
            return new_node
        return self.generic_visit(node)


class SwappedArgumentsMutator(_FirstMutationTransformer):
    """Swap the first two positional arguments in the first eligible function call.

    Targets calls with ≥2 positional args that are not in the function signature
    (i.e. internal calls), e.g. ``sorted(lst, key=fn)`` → the two positional args
    swap.  Detecting this requires knowing what the callee expects — execution
    knowledge, not just token matching.
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not self.mutated and len(node.args) >= 2:
            self.mutated = True
            new_node = copy.copy(node)
            new_args = list(node.args)
            new_args[0], new_args[1] = new_args[1], new_args[0]
            new_node.args = new_args
            return new_node
        return self.generic_visit(node)


_MUTATOR_CLASSES: dict[str, list] = {
    "off_by_one_plus": [lambda: OffByOneMutator(delta=1)],
    "off_by_one_minus": [lambda: OffByOneMutator(delta=-1)],
    "wrong_operator": [WrongOperatorMutator],
    "condition_flip": [ConditionFlipMutator],
    "wrong_comparator": [WrongComparatorMutator],
    # Hard mutations (require execution reasoning)
    "wrong_variable": [WrongVariableMutator],
    "deleted_accumulator": [DeletedAccumulatorMutator],
    "swapped_arguments": [SwappedArgumentsMutator],
}

HARD_MUTATION_TYPES = ["wrong_variable", "deleted_accumulator", "swapped_arguments"]


# ---------------------------------------------------------------------------
# Core mutation function
# ---------------------------------------------------------------------------


def apply_mutation(code: str, mutation_type: str) -> str | None:
    """Apply one mutation of `mutation_type` to `code`.

    Returns the mutated source string, or None if no eligible node was found.
    """
    constructors = _MUTATOR_CLASSES.get(mutation_type)
    if constructors is None:
        raise ValueError(f"Unknown mutation_type: {mutation_type!r}")

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for ctor in constructors:
        transformer = ctor()
        new_tree = transformer.visit(copy.deepcopy(tree))
        if transformer.mutated:
            ast.fix_missing_locations(new_tree)
            try:
                return ast.unparse(new_tree)
            except Exception:
                return None

    return None


# ---------------------------------------------------------------------------
# Execution-based validation
# ---------------------------------------------------------------------------


def _execute_function(code: str, input_str: str, timeout: int = 5) -> tuple[bool, str]:
    """Execute `code` with `f({input_str})` and return (ok, output_repr).

    Returns (True, repr(result)) on success, (False, error_msg) on exception.
    Times out after `timeout` seconds to avoid infinite loops in mutated code.
    """
    def _alarm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
        raise TimeoutError("execution timed out")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        globs: dict = {}
        exec(compile(code, "<string>", "exec"), globs)  # noqa: S102
        f = globs.get("f")
        if f is None:
            return False, "no function 'f'"
        # Parse input_str to Python values
        result = eval(f"f({input_str})", globs)  # noqa: S307
        return True, repr(result)
    except TimeoutError:
        return False, "execution timed out"
    except Exception as exc:
        return False, str(exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def validate_mutation(
    original_code: str,
    mutated_code: str,
    input_str: str,
    original_output: str,
) -> bool:
    """Return True iff mutated_code produces a different output than original."""
    ok_mut, out_mut = _execute_function(mutated_code, input_str)
    if not ok_mut:
        return False
    # Outputs differ (compare repr-level)
    return out_mut.strip() != original_output.strip()


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


@dataclass
class MutationPair:
    pair_id: str
    original_id: str
    original_code: str
    input_str: str
    correct_output: str
    buggy_code: str
    wrong_output: str
    mutation_type: str


def generate_pairs(
    samples: list[dict],
    rng: random.Random,
    mutation_types: list[str] | None = None,
    max_mutations_per_sample: int = 3,
) -> Iterator[MutationPair]:
    """Yield MutationPair for each successful (original, buggy) pair."""
    if mutation_types is None:
        mutation_types = list(_MUTATOR_CLASSES.keys())

    for sample in samples:
        sample_id = sample.get("id", sample.get("sample_id", "?"))
        code = sample["code"]
        input_str = sample["input"]
        correct_output = sample["output"]

        # Shuffle mutation types so we get variety
        shuffled = list(mutation_types)
        rng.shuffle(shuffled)
        n_found = 0

        for mtype in shuffled:
            if n_found >= max_mutations_per_sample:
                break
            mutated = apply_mutation(code, mtype)
            if mutated is None:
                continue
            ok, wrong_out = _execute_function(mutated, input_str)
            if not ok:
                continue
            if wrong_out.strip() == correct_output.strip():
                continue  # mutation didn't change output
            pair_id = f"{sample_id}__{mtype}"
            yield MutationPair(
                pair_id=pair_id,
                original_id=str(sample_id),
                original_code=code,
                input_str=input_str,
                correct_output=correct_output,
                buggy_code=mutated,
                wrong_output=wrong_out,
                mutation_type=mtype,
            )
            n_found += 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build bug-injection dataset from CRUXEval")
    parser.add_argument("--output_path", type=str, default="interp/bug_trace/data/pairs.json")
    parser.add_argument("--n_samples", type=int, default=800)
    parser.add_argument("--max_mutations_per_sample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", type=int, default=0, help="Print N pairs and exit")
    parser.add_argument(
        "--hard", action="store_true",
        help="Use hard mutation types only (wrong_variable, deleted_accumulator, swapped_arguments)"
    )
    parser.add_argument(
        "--mutation_types", type=str, default="",
        help="Comma-separated list of mutation types to use (overrides --hard)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Determine which mutation types to use
    if args.mutation_types:
        mutation_types = [m.strip() for m in args.mutation_types.split(",") if m.strip()]
    elif args.hard:
        mutation_types = HARD_MUTATION_TYPES
    else:
        mutation_types = None  # use all

    from datasets import load_dataset

    dataset = list(load_dataset("cruxeval-org/cruxeval", split="test"))
    if args.n_samples > 0:
        dataset = dataset[: args.n_samples]

    rng = random.Random(args.seed)
    pairs = list(generate_pairs(
        dataset, rng,
        mutation_types=mutation_types,
        max_mutations_per_sample=args.max_mutations_per_sample,
    ))

    # Statistics
    from collections import Counter
    type_counts = Counter(p.mutation_type for p in pairs)
    logger.info(f"Generated {len(pairs)} mutation pairs from {len(dataset)} samples")
    for mtype, cnt in sorted(type_counts.items()):
        logger.info(f"  {mtype}: {cnt}")

    if args.dry_run > 0:
        print(f"\n=== Dry run: {min(args.dry_run, len(pairs))} pairs ===")
        for p in pairs[: args.dry_run]:
            print(f"\n--- {p.pair_id} ({p.mutation_type}) ---")
            print("ORIGINAL:")
            print(textwrap.indent(p.original_code, "  "))
            print(f"BUGGY ({p.mutation_type}):")
            print(textwrap.indent(p.buggy_code, "  "))
            print(f"Input: f({p.input_str})")
            print(f"Correct output: {p.correct_output}")
            print(f"Wrong output:   {p.wrong_output}")
        return

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(
            [
                {
                    "pair_id": p.pair_id,
                    "original_id": p.original_id,
                    "original_code": p.original_code,
                    "input_str": p.input_str,
                    "correct_output": p.correct_output,
                    "buggy_code": p.buggy_code,
                    "wrong_output": p.wrong_output,
                    "mutation_type": p.mutation_type,
                }
                for p in pairs
            ],
            f,
            indent=2,
        )
    logger.info(f"Saved {len(pairs)} pairs to {out}")


if __name__ == "__main__":
    main()
