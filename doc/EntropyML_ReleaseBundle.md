# EntropyML Release Bundle

**Version:** 1.0.0  
**Tag:** v1.0.0  
**Date:** 2026-08-13
**Scope:** Complete distribution record for EntropyML v1.0.0  
**Audience:** Maintainers, integrators, reviewers  
**Versioning policy:** [EntropyML_Versioning.md](EntropyML_Versioning.md)

---

## Release Summary

EntropyML v1.0.0 is the initial stable release of the thermodynamically-grounded
latent variable model framework. It delivers:

- A fully implemented and validated TVAE example (`ExampleTVAE`) following the PSC flow
- A 235-check automated guardrail suite covering six invariant groups
- A model-agnostic behavioral contract with `ILatentModel`, `VaeAdapter`, and `TvaeAdapter`
- A complete eight-document specification and distribution mesh
- Cross-model consistency validation between VAE and TVAE families

---

## Release Notes

### What is new in v1.0.0

- **ExampleTVAE** — fully thermodynamic example with PSC-compliant flow, thermodynamic
  logging (FreeEnergy, Energy, Entropy, eq-range), and passing guardrail suite
- **Guardrail suite** — 235 checks across convergence signature, latent geometry,
  reconstruction, decomposition identity, PSC flow, and stability envelope
- **Behavioral contract** — `ILatentModel` interface, `VaeAdapter`, `TvaeAdapter`,
  and `BehavioralContract.Run(...)` validating both VAE and TVAE families
- **HMD** — canonical hierarchical specification covering AML, PSC, behavioral
  contract, validation envelope, guardrails, cross-model consistency, and onboarding
- **Spec-Lite** — 2-page executive summary
- **Quick-Start** — developer onboarding guide with code snippets and flow diagrams
- **API Sheet** — single-page adapter mapping and interface reference
- **Landing Page** — narrative conceptual introduction with ASCII diagrams
- **Folder Structure** — annotated project tree for contributors
- **Versioning Policy** — semantic versioning governance
- **README** — GitHub front door with installation and run instructions

### Known limitations

- The validation envelope numerical bands are calibrated for the current TVAE and VAE
  configurations only. Adding a new model requires envelope recalibration (minor release).
- Cross-model consistency checks currently cover VAE and TVAE only. Future models
  must be registered explicitly in `RunBehavioralContractGuardrails()`.

### Breaking changes

None — this is the initial stable release.

---

## Artifact Manifest

### Specification artifacts

| Artifact | Path | Role | Version |
|---|---|---|---|
| EntropyML HMD | [EntropyML_HMD.md](EntropyML_HMD.md) | Canonical specification | 2.0 |
| Spec-Lite | [EntropyML_SpecLite.md](EntropyML_SpecLite.md) | Executive summary | 1.0 |
| Quick-Start | [EntropyML_QuickStart.md](EntropyML_QuickStart.md) | Onboarding guide | 1.0 |
| API Sheet | [EntropyML_API_Sheet.md](EntropyML_API_Sheet.md) | API reference card | 1.0 |

### Entry-point artifacts

| Artifact | Path | Role | Version |
|---|---|---|---|
| README | [../README.md](../README.md) | GitHub front door | 1.0 |
| Landing Page | [EntropyML_Landing.md](EntropyML_Landing.md) | Narrative intro | 1.0 |

### Operational artifacts

| Artifact | Path | Role | Version |
|---|---|---|---|
| Folder Structure | [EntropyML_FolderStructure.md](EntropyML_FolderStructure.md) | Architectural map | 1.0 |
| Versioning Policy | [EntropyML_Versioning.md](EntropyML_Versioning.md) | Governance | 1.0 |
| This document | EntropyML_ReleaseBundle.md | Distribution artifact | 1.0.0 |

### Code artifacts

| Artifact | Path | Role |
|---|---|---|
| TVAE model | `EntropyML.TVAE/TVAE.cs` | Model implementation |
| VAE model | `EntropyML.VAE/VAE.cs` | Model implementation |
| ILatentModel + adapters | `Examples/ExampleTVAE/EntropyMLBehavioralContract.cs` | Adapter layer |
| Guardrail suite | `Examples/ExampleTVAE/ExampleTVAEGuardrails.cs` | 235-check suite |
| PSC example | `Examples/ExampleTVAE/ExampleTVAEv1.cs` | Runnable example |
| Entry point | `Examples/ExampleTVAE/Program.cs` | Executable entry |

---

## Compatibility Notes

| Dependency | Required version |
|---|---|
| .NET SDK | 10.0 or later |
| Visual Studio | 2022 17.x or Visual Studio 2026 18.x |
| EntropyML.TVAE | as shipped in v1.0.0 |
| EntropyML.VAE | as shipped in v1.0.0 |
| EntropyML.Data | as shipped in v1.0.0 |
| EntropyML.NN | as shipped in v1.0.0 |

The behavioral contract and guardrail suite are calibrated for the model versions
shipped in this bundle. Updating any model library independently may require envelope
recalibration and a new release (see [EntropyML_Versioning.md](EntropyML_Versioning.md)).

---

## Guardrail Baseline

| Suite | Checks | Status |
|---|---|---|
| Baseline training | included in 235 | All passing |
| Baseline test | included in 235 | All passing |
| Multi-seed | included in 235 | All passing |
| Distribution robustness | included in 235 | All passing |
| Latent-dimension consistency | included in 235 | All passing |
| Cross-model consistency | included in 235 | All passing |
| Long-horizon stability | included in 235 | All passing |
| Behavioral contract (VAE + TVAE) | included in 235 | All passing |
| **Total** | **235** | **235/235** |

---

## Changelog

### v1.0.0 (initial stable release)
- All artifacts listed above created and cross-linked
- 235-check guardrail suite passing on reference hardware
- Behavioral contract validated for both VAE and TVAE adapters
- HCMD documentation pyramid complete

---

*End of EntropyML Release Bundle v1.0.0 — versioning policy: [EntropyML_Versioning.md](EntropyML_Versioning.md) · canonical spec: [EntropyML_HMD.md](EntropyML_HMD.md)*
