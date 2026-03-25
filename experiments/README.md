# Experiments

Each subdirectory is a self-contained experiment with a `REPORT.md` (scientific document:
hypothesis, reproduction commands, results, interpretation) and optionally a `PLAN.md`
(design document written before running, with methods, decision criteria, and expected findings).

| # | Experiment | Status | Key finding |
|---|------------|--------|-------------|
| [01](01_interp_cruxeval/) | Interp on CRUXEval (trace_full) | ✅ Complete | `return_type` linearly decodable +34pp at layer 32; steering +2pp (underpowered) |
| [02](02_bug_trace/) | Bug-fixing representation tracing | ⏳ Running | Extraction job pending (SLURM 15901326) |

## Code

Experiment code lives alongside the experiments in `interp/`:

```
interp/
  extract/          # activation capture (hooks, run_extract)
  probes/           # linear/MLP probe training
  logit_lens/       # logit lens analysis
  steering/         # activation steering
  token_intervention/ # targeted token-level interventions
  bug_trace/        # experiment 02 code (mutation, extraction, analysis)
  scripts/          # SLURM job scripts
```
