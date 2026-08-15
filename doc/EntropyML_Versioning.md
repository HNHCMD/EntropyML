# EntropyML Versioning Policy

**Version:** 1.0  
**Date:** 2026-08-13
**Scope:** Semantic versioning rules for all EntropyML artifacts and releases  
**Audience:** Maintainers, contributors, release managers  
**Canonical reference:** [EntropyML_HMD.md](EntropyML_HMD.md)  
**Current release:** [EntropyML_ReleaseBundle.md](EntropyML_ReleaseBundle.md)

---

## 1. Versioning Scheme

EntropyML follows **Semantic Versioning 2.0.0** (`MAJOR.MINOR.PATCH`).

```
  v1.0.0
  │ │ └── Patch   — backward-compatible fixes, documentation corrections
  │ └──── Minor   — backward-compatible additions (new model, new check)
  └────── Major   — breaking changes (contract redefinition, PSC reorder, API removal)
```

The version applies to the **specification layer** (HMD, behavioral contract, adapter
interface) and the **code layer** (guardrails, adapters, model API contracts) together.
A change to either layer triggers a version increment.

---

## 2. What Increments Major

A **major version increment** is required when:

- The `ILatentModel` interface changes (method added, removed, or signature changed)
- The `BehavioralContract` invariant groups change in number or semantics
- The PSC phase order or phase definitions change
- The AML term-mapping table changes (e.g., a thermodynamic term is renamed or redefined)
- The decomposition identity formula changes
- A previously passing invariant is removed from the guardrail suite
- An existing adapter's mapping to a model API changes in a breaking way

Major increments require:
- Updated HMD (full re-review of affected sections)
- Updated Spec-Lite, Quick-Start, and API Sheet
- A new release bundle tagged `vX.0.0`

---

## 3. What Increments Minor

A **minor version increment** is required when:

- A new compliant model is added (new model class + new adapter)
- A new guardrail check is added to the suite (235 → N checks)
- A new invariant group is added to `BehavioralContract`
- A new example project is added under `Examples/`
- A new documentation artifact is added (e.g., a new guide or reference page)
- The validation envelope is recalibrated for a new model (bands extended, not tightened)

Minor increments require:
- Updated HMD (new section or extended existing section)
- Updated Quick-Start (new model onboarding instructions)
- Updated API Sheet (new adapter mapping table)
- A new release bundle tagged `v1.X.0`

---

## 4. What Increments Patch

A **patch version increment** is required when:

- A documentation typo, phrasing, or cross-link is corrected
- A guardrail check threshold is adjusted within the same invariant group
- A bug in an adapter that does not change the interface is fixed
- A build or project configuration fix is applied
- An example output log message is corrected

Patch increments require:
- A changelog entry in the release bundle
- A new release bundle tagged `v1.0.X`

---

## 5. Behavioral Contract Changes

| Change | Version impact |
|---|---|
| New invariant group added | Minor |
| Existing invariant group removed | **Major** |
| Invariant threshold tightened (more restrictive) | Minor |
| Invariant threshold relaxed (less restrictive) | Patch (with justification) |
| `ILatentModel` method added | **Major** |
| `ILatentModel` method removed | **Major** |
| `ILatentModel` method signature changed | **Major** |
| New adapter added (existing interface) | Minor |
| Adapter bug fix (no interface change) | Patch |

---

## 6. Guardrail Changes

| Change | Version impact |
|---|---|
| New check added | Minor |
| Existing check removed | **Major** |
| Check threshold adjusted (same semantics) | Patch |
| Check moved to a different invariant group | Patch (with changelog) |
| Total check count changes | Minor (if increase) or **Major** (if decrease) |

The guardrail count (currently 235) is itself a versioned artifact. Any reduction
is treated as a breaking change.

---

## 7. New Model Onboarding

Adding a new model always requires a **minor increment** regardless of model
complexity. The rationale: every new model expands the cross-model consistency
surface and adds new contract checks, which is a forward-compatible addition.

The new model's adapter and example must pass all existing checks **plus** the
new model's contract checks before a minor release may be tagged.

---

## 8. Release Tagging

Git tags follow the pattern: `vMAJOR.MINOR.PATCH`

Examples:
- `v1.0.0` — initial stable release
- `v1.1.0` — new model or new guardrail group added
- `v1.1.1` — documentation fix
- `v2.0.0` — ILatentModel interface changed

Each tag must correspond to a committed [EntropyML_ReleaseBundle.md](EntropyML_ReleaseBundle.md)
that documents the exact state of all artifacts at that version.

---

## 9. Document Version Synchronisation

All five documentation artifacts carry an independent version header. They must be
kept in sync with the specification version:

| Document | Must update on |
|---|---|
| EntropyML_HMD.md | Every increment |
| EntropyML_SpecLite.md | Major or Minor |
| EntropyML_QuickStart.md | Major or Minor |
| EntropyML_API_Sheet.md | Major or Minor |
| EntropyML_Landing.md | Major |
| EntropyML_FolderStructure.md | Major or Minor |
| EntropyML_Versioning.md | Major |
| EntropyML_ReleaseBundle.md | Every increment |

---

*End of EntropyML Versioning Policy v1.0 — release history: [EntropyML_ReleaseBundle.md](EntropyML_ReleaseBundle.md)*
