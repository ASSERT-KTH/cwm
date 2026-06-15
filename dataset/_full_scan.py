"""Full execution scan of all three datasets (multiprocess).

Reports, per dataset and overall:
  1. non-executable samples (NameError filtered / raised in main / timed out)
  2. special-format values that may need filtering (default-repr objects,
     residual ``at 0x`` addresses, non-finite floats, oversized values)
  3. overly long traces (frame-count distribution + worst offenders)
  4. pure-function vs I/O split (``dataset.purity``), the training-filter signal

Each sample is traced in a worker process with its own SIGALRM timeout, and the
worker returns only compact stats (never the raw frames) so runaway traces never
get pickled back to the parent. A per-trace ``MAX_FRAMES`` valve stops runaway
loops from exhausting worker memory; the wall-clock timeout still bounds runtime.
"""
from __future__ import annotations

import os
import re
import signal
from collections import Counter
from multiprocessing import Pool

from dataset.sources import load_rows
from dataset.ground_truth import Timeout, alarm_handler, check_purity, ground_truth_trace

DATASETS = os.environ.get("SCAN_DATASETS", "cruxeval_o,mbpp,humaneval").split(",")
TIMEOUT_S = 5
LONG_TRACE = 200          # frame count above which we flag a trace as "long"
BIG_VALUE = 200           # rendered-value length above which we flag it
MAX_FRAMES = 50_000       # per-trace accumulation cap (runaway-loop guard)
NPROC = int(os.environ.get("NPROC", min(os.cpu_count() or 1, 32)))

_OBJ_REPR = re.compile(r"^<.*>$")               # default-repr objects: <...>
_ADDR_LEFT = re.compile(r" at 0x[0-9a-fA-F]+")  # residual heap address


def _init_worker():
    signal.signal(signal.SIGALRM, alarm_handler)  # raises ground_truth.Timeout


def _obj_key(v: str) -> str:
    """Collapse a default-repr to a stable key (drop trailing instance detail)."""
    return v.split(" object")[0] + (" object>" if " object" in v else "")


def iter_values(frames):
    """Yield every rendered local value and return/exception arg in a trace."""
    for fr in frames:
        if fr.locals:
            yield from fr.locals.values()
        if fr.arg is not None:
            yield fr.arg


def _scan_one(task):
    """Trace one sample and return compact stats (runs in a worker process)."""
    rid, code, input_str = task
    signal.alarm(TIMEOUT_S)
    try:
        frames, err = ground_truth_trace(code, input_str, max_frames=MAX_FRAMES)
        signal.alarm(0)
    except Timeout:
        signal.alarm(0)
        return {"id": rid, "status": "timeout"}
    except Exception as e:  # noqa: BLE001 - unexpected def/compile failure
        signal.alarm(0)
        return {"id": rid, "status": "filtered", "err": f"{type(e).__name__}: {e}"}

    # Drop NameError samples (definition-after-use / undefined names) outright.
    if err is not None and err.startswith("NameError"):
        return {"id": rid, "status": "filtered", "err": err}

    # Pure-function check: a second lightweight run (no settrace) under the same
    # wall-clock budget. ``timeout`` here only marks the purity verdict unknown.
    signal.alarm(TIMEOUT_S)
    try:
        signals = check_purity(code, input_str)
        signal.alarm(0)
        ppurity = "pure" if not signals else ("error" if "error" in signals else "impure")
        psignals = sorted(signals - {"error"})
    except Timeout:
        signal.alarm(0)
        ppurity, psignals = "timeout", []

    obj = Counter()
    nonfinite = Counter()
    addr = False
    big_count = 0
    big_sample = None
    for v in iter_values(frames):
        if v == "..":
            continue
        if _ADDR_LEFT.search(v):
            addr = True
        if _OBJ_REPR.match(v):
            obj[_obj_key(v)] += 1
        if v in ("inf", "-inf", "nan"):
            nonfinite[v] += 1
        if len(v) > BIG_VALUE:
            big_count += 1
            if big_sample is None:
                big_sample = (rid, len(v), v[:80])

    return {
        "id": rid,
        "status": "ok",
        "nframes": len(frames),
        "err": err,
        "obj": obj,
        "nonfinite": nonfinite,
        "addr": addr,
        "big_count": big_count,
        "big_sample": big_sample,
        "purity": ppurity,
        "psignals": psignals,
    }


def _report(name, n, results):
    errors = []
    filtered = []
    timeouts = []
    long_traces = []
    frame_total = []
    obj_examples = Counter()
    nonfinite = Counter()
    addr_hits = []
    big_total = 0
    big_samples = []

    for r in results:
        st = r["status"]
        if st == "timeout":
            timeouts.append(r["id"])
            continue
        if st == "filtered":
            filtered.append((r["id"], r["err"]))
            continue
        # ok
        nf = r["nframes"]
        frame_total.append(nf)
        if r["err"] is not None:
            errors.append((r["id"], r["err"]))
        if nf >= LONG_TRACE:
            long_traces.append((r["id"], nf))
        obj_examples.update(r["obj"])
        nonfinite.update(r["nonfinite"])
        if r["addr"]:
            addr_hits.append(r["id"])
        big_total += r["big_count"]
        if r["big_sample"] is not None and len(big_samples) < 5:
            big_samples.append(r["big_sample"])

    print("=" * 78)
    print(f"DATASET {name}: {n} rows")
    print("-" * 78)

    # 1. executability
    ok = n - len(errors) - len(timeouts) - len(filtered)
    print(f"[1] executable cleanly : {ok}/{n}")
    print(f"    filtered (NameError, dropped): {len(filtered)}")
    fl_kinds = Counter(e.split(":")[0] for _, e in filtered)
    for kind, c in fl_kinds.most_common():
        ex = next(rid for rid, e in filtered if e.startswith(kind))
        print(f"        {c:5}  {kind}  (e.g. {ex})")
    print(f"    raised in main()   : {len(errors)}")
    er_kinds = Counter(e.split(":")[0] for _, e in errors)
    for kind, c in er_kinds.most_common(10):
        ex = next(rid for rid, e in errors if e.startswith(kind))
        print(f"        {c:5}  {kind}  (e.g. {ex})")
    print(f"    timed out (>{TIMEOUT_S}s): {len(timeouts)}  {timeouts[:8]}")

    # 2. special formats
    print(f"[2] residual 'at 0x' addresses : {len(addr_hits)}  {addr_hits[:5]}")
    print(f"    default-repr objects (after addr-strip): {sum(obj_examples.values())} values")
    for val, c in obj_examples.most_common(10):
        print(f"        {c:6}  {val}")
    print(f"    non-finite floats : {dict(nonfinite)}")
    print(f"    oversized values (>{BIG_VALUE} chars): {big_total}")
    for rid, ln, snip in big_samples:
        print(f"        {rid}: len={ln}  {snip!r}...")

    # 3. trace length
    if frame_total:
        fts = sorted(frame_total)
        p50 = fts[len(fts) // 2]
        p95 = fts[int(len(fts) * 0.95)]
        print(f"[3] trace frames: median={p50}  p95={p95}  max={fts[-1]}")
        print(f"    long traces (>={LONG_TRACE} frames): {len(long_traces)}")
        for rid, c in sorted(long_traces, key=lambda x: -x[1])[:8]:
            print(f"        {rid}: {c} frames")

    # 4. pure-function vs I/O split (over the executable samples = denominator)
    purity = Counter(r["purity"] for r in results if "purity" in r)
    sig = Counter(s for r in results for s in r.get("psignals", []))
    denom = sum(purity.values()) or 1
    print(f"[4] pure-function: pure={purity['pure']}/{denom} ({100 * purity['pure'] / denom:.1f}%)  "
          f"impure={purity['impure']}  purity-timeout={purity['timeout']}")
    print(f"    impure signals: stdout={sig['stdout']}  file={sig['file']}  syscall={sig['syscall']}")
    print()


def main():
    print(f"workers={NPROC}  timeout={TIMEOUT_S}s  max_frames={MAX_FRAMES}\n")
    with Pool(processes=NPROC, initializer=_init_worker) as pool:
        for name in DATASETS:
            rows = load_rows([name])
            tasks = [(r["id"], r["code"], r["input"]) for r in rows]
            results = pool.map(_scan_one, tasks, chunksize=16)
            _report(name, len(rows), results)


if __name__ == "__main__":
    main()
