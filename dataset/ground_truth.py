# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Ground-truth execution-trace generation in CWM's trace format.

We execute the CRUXEval function under ``sys.settrace`` and reconstruct the
same frame sequence that CWM is trained to predict: a CALL frame on entering a
scope, a LINE frame before each executed line, and a RETURN/EXCEPTION frame on
leaving a scope. Locals use the diff-based representation (unchanged variables
render as ``".."``), and every value is rendered with Python ``repr``.

This mirrors the entry-point convention used by the trace prompt
(``evals.cruxeval.prompts.make_trace_full_prompt_tokens``): a synthetic
``def main(): return f(<input>)`` wraps the function under test, and the trace
starts when ``main`` is called. The prompt seeds the first ``call main()``
frame, so ``ground_truth_trace`` drops it by default (see ``drop_entry_call``).

Caveats (documented in README.md): the exact diff-reset rules used to build
CWM's original training traces are not published in this repo. Values are
rendered with ``repr`` (confirmed against real generations: single-quoted
strings, parenthesized tuples, bare ints), but exotic objects may differ from
CWM's internal renderer. Treat the resulting numbers as a faithful
re-implementation, not a bit-exact replica of Meta's internal tracer.
"""

from __future__ import annotations

import builtins
import linecache
import multiprocessing
import os
import re
import signal
import sys
import warnings
from types import FrameType
from typing import Any

# Default object reprs embed a non-reproducible heap address, e.g.
# ``<list_iterator object at 0x7f57...>``. Strip the ``at 0x...`` so traces are
# deterministic across runs and alignable with model generations.
_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")
# Module reprs embed a machine-specific absolute path, e.g.
# ``<module 'math' from '/proj/.../math...so'>``. Drop the ``from '...'`` so the
# rendered value is portable across machines.
_MODULE_RE = re.compile(r"(<module '[^']+') from '[^']*'>")

from .trace_format import (
    DIFF_PLACEHOLDER,
    TraceEvent,
    TraceFrame,
    normalize_source,
    render_frames_to_generation,
)

_FILENAME = "<cwm_trace>"
_ENTRY = "main"


def make_trace_context(code: str, input_str: str) -> str:
    """Source context for trace prediction (matches cruxeval.prompts)."""
    return f"\n{code}\ndef main():  # << START_OF_TRACE\n    return f({input_str})\n"


def render_value(value: Any) -> str:
    """Render a Python value as CWM does: the Python source ``repr``.

    Confirmed against real CWM generations: tuples render as ``(4, 1)`` and
    strings as ``'x'`` (single-quoted), i.e. ``repr`` semantics, *not*
    ``json.dumps`` (which would emit ``[4, 1]`` / ``"x"``). The value string is
    then stored as a JSON string inside the frame's locals object.
    """
    try:
        return _MODULE_RE.sub(r"\1>", _ADDR_RE.sub("", repr(value)))
    except Exception:  # noqa: BLE001 - a broken __repr__ shouldn't crash eval
        return "<unrepr>"


class _GroundTruthTracer:
    def __init__(
        self, code: str, input_str: str, max_frames: int | None = None
    ) -> None:
        self.context = make_trace_context(code, input_str)
        # Register the context source so frame line numbers resolve to lines.
        src_lines = self.context.splitlines(keepends=True)
        linecache.cache[_FILENAME] = (
            len(self.context),
            None,
            src_lines,
            _FILENAME,
        )
        self._code_obj = compile(self.context, _FILENAME, "exec")
        self.frames: list[TraceFrame] = []
        # Per-scope (keyed by id(frame)) snapshot of last-rendered locals, to
        # compute the diff-based representation.
        self._scope_prev: dict[int, dict[str, str]] = {}
        self._entry_frame_id: int | None = None
        self.error: str | None = None
        # Safety valve: stop accumulating frames past this count (runaway loops
        # / recursion). ``None`` disables it. ``truncated`` flags when it fired.
        self._max_frames = max_frames
        self.truncated = False

    # -- tracer callbacks ---------------------------------------------------

    def _source_line(self, frame: FrameType) -> str:
        line = linecache.getline(_FILENAME, frame.f_lineno)
        return normalize_source(line)

    def _diff_locals(self, frame: FrameType) -> dict[str, str]:
        scope = id(frame)
        prev = self._scope_prev.setdefault(scope, {})
        current: dict[str, str] = {}
        rendered: dict[str, str] = {}
        for name, val in frame.f_locals.items():
            r = render_value(val)
            rendered[name] = r
            if name in prev and prev[name] == r:
                current[name] = DIFF_PLACEHOLDER
            else:
                current[name] = r
        self._scope_prev[scope] = rendered
        return current

    def _trace(self, frame: FrameType, event: str, arg: Any):  # noqa: ANN001
        # Only follow execution at or below the entry point's scope.
        if self._entry_frame_id is None:
            if event == "call" and frame.f_code.co_name == _ENTRY:
                self._entry_frame_id = id(frame)
            else:
                return None

        # Don't descend into library / non-user code (re, codecs, math, ...);
        # only frames from our synthetic source belong in the trace.
        if frame.f_code.co_filename != _FILENAME:
            return None

        # Runaway loop/recursion guard: stop accumulating (and stop tracing) so
        # we don't exhaust memory. Any wall-clock timeout still bounds runtime.
        if self._max_frames is not None and len(self.frames) >= self._max_frames:
            if not self.truncated:
                self.truncated = True
                sys.settrace(None)
            return None

        if event == "call":
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.CALL,
                    source=self._source_line(frame),
                    locals=self._diff_locals(frame),
                )
            )
            return self._trace
        if event == "line":
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.LINE,
                    source=self._source_line(frame),
                    locals=self._diff_locals(frame),
                )
            )
            return self._trace
        if event == "return":
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.RETURN,
                    source=self._source_line(frame),
                    arg=render_value(arg),
                )
            )
            return self._trace
        if event == "exception":
            exc_type = arg[0]
            self.frames.append(
                TraceFrame(
                    event=TraceEvent.EXCEPTION,
                    source=self._source_line(frame),
                    arg=render_value(getattr(exc_type, "__name__", str(exc_type))),
                )
            )
            return self._trace
        return self._trace

    # -- driver -------------------------------------------------------------

    def run(self, timeout_unused: float = 0.0) -> None:
        ns: dict[str, Any] = {}
        # Define f and main without tracing module-level execution. Module-level
        # failures (e.g. definition-after-use NameError) are recorded on
        # ``error`` rather than raised, so callers can filter such unexecutable
        # samples uniformly instead of crashing.
        try:
            exec(self._code_obj, ns)
        except Exception as e:  # noqa: BLE001 - unexecutable sample
            self.error = f"{type(e).__name__}: {e}"
            return
        main = ns[_ENTRY]
        old = sys.gettrace()
        sys.settrace(self._trace)
        try:
            main()
        except Exception as e:  # noqa: BLE001 - record but don't crash eval
            self.error = f"{type(e).__name__}: {e}"
        finally:
            sys.settrace(old)


def drop_entry_call(frames: list[TraceFrame]) -> list[TraceFrame]:
    """Drop the leading ``call main()`` frame.

    The full-trace prompt seeds ``<|call_sep|>{}<|action_sep|>def main():`` so
    the model only generates from the *next* frame onward. To align a generated
    trace with the ground truth we must drop this seeded entry frame.
    """
    if (
        frames
        and frames[0].event == TraceEvent.CALL
        and frames[0].source.startswith("def main()")
    ):
        return frames[1:]
    return frames


def ground_truth_trace(
    code: str,
    input_str: str,
    align_to_prompt: bool = True,
    max_frames: int | None = None,
) -> tuple[list[TraceFrame], str | None]:
    """Return (ground-truth frames, error) for executing ``f(input_str)``.

    When ``align_to_prompt`` is True (the default), the leading seeded
    ``call main()`` frame is dropped so the frames line up positionally with a
    model generation produced from ``make_trace_full_prompt_tokens``.

    ``error`` is non-None if the traced program raised; the frames captured up
    to that point are still returned.
    """
    tracer = _GroundTruthTracer(code, input_str, max_frames=max_frames)
    tracer.run()
    frames = drop_entry_call(tracer.frames) if align_to_prompt else tracer.frames
    # A runaway trace stopped at ``max_frames`` is incomplete (no proper end);
    # surface it via ``error`` so consumers can drop it (signature unchanged).
    error = "Truncated: max_frames exceeded" if tracer.truncated else tracer.error
    return frames, error


# --- pure-function detection (training-data filter) -------------------------
# A sample is *impure* if running ``f(input)`` writes stdout/stderr, opens a
# file, reads stdin, or does a network/process/fs-mutating syscall. We filter
# training data to the pure-function distribution of the CRUXEval test set.
# Detection is a settrace-free run bounded by the caller's SIGALRM
# (``alarm_handler`` raises ``Timeout``, a BaseException, so ``check_purity``
# never swallows it). A raised program is NOT impure -- only I/O counts. The
# audit hook + ``open`` stub both flag and block (host safety for PyX's
# arbitrary code) and are inert outside a check (``_io_active`` gate).
_IO_EVENTS = frozenset({
    "socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
    "os.remove", "os.rename", "os.mkdir", "os.rmdir",
    "shutil.copytree", "shutil.rmtree", "shutil.move",
    "urllib.Request", "builtins.input",
})
_io_active = False
_io_seen: set[str] = set()
_hook_installed = False


class Timeout(BaseException):
    pass


class _BlockedIO(Exception):
    pass


def alarm_handler(signum, frame):  # noqa: ANN001
    raise Timeout


def _audit(event, args):  # noqa: ANN001
    if _io_active and event in _IO_EVENTS:
        _io_seen.add(event)
        raise _BlockedIO


class _FlagWriter:
    """Stream that only records whether anything was written."""

    def __init__(self) -> None:
        self.wrote = False

    def write(self, s):  # noqa: ANN001
        if s:
            self.wrote = True
        return len(s) if s else 0

    def flush(self):
        pass

    def isatty(self):
        return False


def check_purity(code: str, input_str: str) -> set[str]:
    """Run ``f(input)`` (no settrace); return I/O signals seen, empty = pure.

    Signals are a subset of ``{stdout, file, syscall}``; ``{error}`` if the
    sample won't compile. Caller must arm SIGALRM (see ``alarm_handler``).
    """
    global _io_active, _hook_installed
    if not _hook_installed:
        sys.addaudithook(_audit)
        _hook_installed = True
    try:
        code_obj = compile(make_trace_context(code, input_str), _FILENAME, "exec")
    except Exception:  # noqa: BLE001 - uncompilable
        return {"error"}

    writer = _FlagWriter()
    _io_seen.clear()
    opened: list[int] = []
    real_open = builtins.open

    def _open_stub(*a, **k):  # noqa: ANN002, ANN003
        opened.append(1)
        raise FileNotFoundError("blocked by purity check")

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = writer
    builtins.open = _open_stub
    _io_active = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # re-fire import warnings each call
            ns: dict = {}
            try:
                exec(code_obj, ns)
                if (main := ns.get(_ENTRY)) is not None:
                    main()
            except (_BlockedIO, Exception):  # noqa: BLE001 - raised trace is fine
                pass
    finally:
        _io_active = False
        builtins.open = real_open
        sys.stdout, sys.stderr = old_out, old_err

    signals = set()
    if writer.wrote:
        signals.add("stdout")
    if opened:
        signals.add("file")
    if _io_seen:
        signals.add("syscall")
    return signals


# --- training-data filter (single entry; train & eval share these thresholds) ---
MAX_FRAMES = 128   # traces longer than this are dropped (the cap marks them Truncated)
TIMEOUT_S = 5      # per-sample wall-clock bound (runaway-loop guard)


def trace_prompt(code: str, input_str: str) -> str:
    """Trace-prediction prompt seeding frame 0 (``call main()``)."""
    ctx = make_trace_context(code, input_str)
    return f"<|trace_context_start|>{ctx}<|frame_sep|><|call_sep|>{{}}<|action_sep|>def main():\n<|frame_sep|>"


def clean_trace(
    code: str, input_str: str, tokenizer, *, max_seq_len: int, max_frames: int = MAX_FRAMES
) -> tuple[list[int], list[int]] | None:
    """Tokenized ``(prompt_ids, trace_ids)`` for one sample, or None to drop it.

    Every training-data filter lives here: impure I/O (off the CRUXEval
    pure-function distribution), runaway/timeout (``max_frames`` + SIGALRM),
    NameError, empty, and over-token-budget (prompt+trace > ``max_seq_len``).
    A *raised* program is kept (its EXCEPTION frame is valid trace data).
    Caller must arm SIGALRM -> ``alarm_handler``.
    """
    signal.alarm(TIMEOUT_S)
    try:
        if check_purity(code, input_str):  # any I/O signal (or compile error)
            return None
        frames, error = ground_truth_trace(code, input_str, align_to_prompt=True, max_frames=max_frames)
    except Timeout:
        return None
    finally:
        signal.alarm(0)
    if not frames or (error is not None and error.startswith(("Truncated", "NameError"))):
        return None
    prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(
        trace_prompt(code, input_str), add_special_tokens=False
    )
    trace_ids = tokenizer.encode(render_frames_to_generation(frames), add_special_tokens=False)
    if len(prompt_ids) + len(trace_ids) > max_seq_len:
        return None
    return prompt_ids, trace_ids


# Parallel build: workers *fork* the (already tokenizer-loaded) parent and
# inherit it copy-on-write, so they never re-import transformers -- that
# re-import is what spawned per-worker BLAS thread pools and exhausted
# RLIMIT_NPROC. The fork carries these globals into each worker.
_W_TOK = None
_W_SEQ = 0
_W_FRAMES = MAX_FRAMES


def _init_worker() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # no BLAS oversubscription
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    signal.signal(signal.SIGALRM, alarm_handler)


def _worker_clean(item: tuple[str, str]) -> tuple[list[int], list[int]] | None:
    return clean_trace(item[0], item[1], _W_TOK, max_seq_len=_W_SEQ, max_frames=_W_FRAMES)


def clean_traces(
    rows, tokenizer, *, max_seq_len: int, workers: int = 0, max_frames: int = MAX_FRAMES
) -> list[tuple[list[int], list[int]] | None]:
    """``clean_trace`` over ``rows`` ({code, input}); returns a list aligned to
    ``rows`` (None = dropped), order preserved. ``workers<=1`` runs serially in
    this process; otherwise a forked Pool parallelizes the one-off filter+tokenize.
    """
    items = [(r["code"], r["input"]) for r in rows]
    if workers <= 1:
        signal.signal(signal.SIGALRM, alarm_handler)
        return [
            clean_trace(c, i, tokenizer, max_seq_len=max_seq_len, max_frames=max_frames)
            for c, i in items
        ]
    global _W_TOK, _W_SEQ, _W_FRAMES
    _W_TOK, _W_SEQ, _W_FRAMES = tokenizer, max_seq_len, max_frames
    chunk = max(1, len(items) // (workers * 8))
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        return list(pool.imap(_worker_clean, items, chunksize=chunk))
