# EntropyML Terminology

**Specification:** EntropyML HCMD Specification — Terminology Reference  
**Version:** 1.0  
**Date:** 2026-08-13  
**Scope:** EntropyML and HCMD cross-hut terminology collisions  
**Audience:** Document authors, model developers, and reviewers  
**Status:** Active Constraint  
**Canonical reference:** [EntropyML_HMD.md](EntropyML_HMD.md)

---

## PSC Collision

The term **PSC** has two independent meanings across your systems:

1. **HCMD PSC** — _Pseudo‑Structural Code_

2. **EntropyML PSC** — _Physical State Cycle_

Both are correct within their respective frameworks, but they must remain **strictly separated** to avoid cross‑layer contamination.

## Required Stance

Inside EntropyML documentation:

* **PSC = Physical State Cycle** only.

* HCMD PSC must **never** appear in EntropyML documents.

* If HCMD PSC is referenced, it must be spelled out fully (“Pseudo‑Structural Code”) and never abbreviated.

Inside HCMD documentation:

* **PSC = Pseudo‑Structural Code** only.

* EntropyML PSC must **never** appear in HCMD documents.

* If EntropyML PSC is referenced, it must be spelled out fully (“Physical State Cycle”).

This preserves semantic clarity across both huts.

---

## Deterministic Rules

* PSC abbreviation is **context‑locked**.

* EntropyML documents must not use HCMD PSC.

* HCMD documents must not use EntropyML PSC.

* Cross‑hut references must use full names, never abbreviations.

## Tasks

* Enforce PSC namespace separation during all document reviews.

* Flag any ambiguous PSC usage.

* Update governance notes if new collisions appear.

---

## Structural Boundaries

* PSC (EntropyML) belongs to the EntropyML meaning layer.

* PSC (HCMD) belongs to the HCMD structural layer.

* No shared diagrams, no shared definitions, no shared invariants.

* Abbreviation PSC is **not portable** across huts.

## Invariants

* PSC abbreviation is hut‑local.

* Full names required for cross‑hut references.

* No mixed usage allowed.
