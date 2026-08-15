# EntropyML Folder Structure Guide

**Version:** 1.0  
**Date:** 2026-08-13
**Scope:** Complete annotated project tree  
**Audience:** Contributors, model authors, documentation maintainers  
**Canonical reference:** [EntropyML_HMD.md](EntropyML_HMD.md)  
**Narrative overview:** [EntropyML_Landing.md](EntropyML_Landing.md)

---

## Repository Root

```
EntropyML/
 +-- README.md                         GitHub front door; links to all artifacts
 +-- doc/                              All documentation markdown files live here
 +-- solution/
      +-- EntropyML/                    Solution root (all code projects live here)
```

---

## Documentation Layer  (`doc/`)

```
doc/
 +-- EntropyML_HMD.md                   CANONICAL SPECIFICATION — authoritative source
 +-- EntropyML_SpecLite.md              2-page executive summary (derivative)
 +-- EntropyML_QuickStart.md            Developer onboarding guide (derivative)
 +-- EntropyML_API_Sheet.md             Single-page API reference card (derivative)
 +-- EntropyML_Landing.md               Narrative/conceptual landing page (entry point)
 +-- EntropyML_FolderStructure.md       This file — architectural map (operational)
 +-- EntropyML_Versioning.md            Semantic versioning and governance (operational)
 +-- EntropyML_ReleaseBundle.md         Current release artifact (distribution)
 +-- EntropyML_Terminology.md           Terminology collision notice (governance)
```

**Authority order:** HMD > Spec-Lite / Quick-Start / API Sheet > Landing > Operational > Distribution.  
No derivative artifact overrides the HMD.

---

## Solution File

```
solution/EntropyML/
 +-- EntropyML.slnx                     Solution descriptor (references all projects below)
```

---

## Model Libraries

These are the **model layer** — pure implementation, no example code, no adapters.

```
solution/EntropyML/
 +-- EntropyML.TVAE/
 |    +-- EntropyML.TVAE.csproj         Project file
 |    +-- TVAE.cs                      TVAE model implementation
 |                                     ↳ API: Fit, EncodeThermoState,
 |                                             RelaxAndReconstruct, ComputeFreeEnergy
 |
 +-- EntropyML.VAE/
 |    +-- EntropyML.VAE.csproj
 |    +-- VAE.cs                       VAE model implementation
 |                                     ↳ API: Fit, Encode, Reconstruct, ComputeLoss
 |
 +-- EntropyML.AE/
      +-- EntropyML.AE.csproj
      +-- AE.cs                        Autoencoder baseline (no KL term)
```

**Rule:** Model libraries must not be modified to satisfy example or guardrail needs.
All behavioral adaptation happens through the adapter layer (see below).

---

## Infrastructure Libraries

```
solution/EntropyML/
 +-- EntropyML.Data/
 |    +-- EntropyML.Data.csproj
 |    +-- Data.cs                      DataGen, Normalize, measurement utilities
 |
 +-- EntropyML.NN/
      +-- EntropyML.NN.csproj
      +-- NN.cs                        Sequential, DenseLayer, Adam optimizer
```

---

## Example Projects

Each example lives in its own executable project under `Examples/`.

```
solution/EntropyML/Examples/
 |
 +-- ExampleTVAE/                      PRIMARY EXAMPLE — full EntropyML stack
 |    +-- ExampleTVAE.csproj           References: TVAE, VAE, Data, NN
 |    +-- Program.cs                   Entry: TrainTVAE() → TestTVAE() → Guardrails.Run()
 |    +-- ExampleTVAEv1.cs             PSC steps 1-5 (train + test loop)
 |    |                                ↳ thermodynamic logging (FreeEnergy, Energy, Entropy)
 |    +-- ExampleTVAEGuardrails.cs     235-check guardrail suite (Steps 4+5+6 STS)
 |    |                                ↳ RunBaselineTrainingGuardrails()
 |    |                                ↳ RunBaselineTestGuardrails()
 |    |                                ↳ RunMultiSeedGuardrails()
 |    |                                ↳ RunDistributionRobustnessGuardrails()
 |    |                                ↳ RunLatentDimensionGuardrails()
 |    |                                ↳ RunCrossModelConsistencyGuardrails()
 |    |                                ↳ RunLongHorizonGuardrails()
 |    |                                ↳ RunBehavioralContractGuardrails()
 |    +-- EntropyMLBehavioralContract.cs  Adapter layer + model-agnostic contract
 |                                     ↳ ILatentModel (interface)
 |                                     ↳ VaeAdapter (wraps VAE)
 |                                     ↳ TvaeAdapter (wraps TVAE)
 |                                     ↳ BehavioralContract.Run(ILatentModel, ...)
 |
 +-- ExampleVAE/                       VAE reference example (no EntropyML terminology)
 |    +-- ExampleVAE.csproj
 |    +-- Program.cs
 |    +-- ExampleVAEv1.cs
 |    +-- ExampleVAEv2.cs
 |
 +-- ExampleAE/                        Autoencoder baseline example
 |    +-- ExampleAE.csproj
 |    +-- Program.cs
 |
 +-- ExampleData/                      Data generation utility example
 |    +-- ExampleData.csproj
 |    +-- Program.cs
 |
 +-- ExampleNN/                        Neural network primitive example
      +-- ExampleNN.csproj
      +-- Program.cs
```

---

## Where Things Live — Quick Reference

| Thing                    | Location                                             |
| ------------------------ | ---------------------------------------------------- |
| TVAE model               | `EntropyML.TVAE/TVAE.cs`                              |
| VAE model                | `EntropyML.VAE/VAE.cs`                                |
| AE model                 | `EntropyML.AE/AE.cs`                                  |
| ILatentModel interface   | `Examples/ExampleTVAE/EntropyMLBehavioralContract.cs` |
| VaeAdapter               | `Examples/ExampleTVAE/EntropyMLBehavioralContract.cs` |
| TvaeAdapter              | `Examples/ExampleTVAE/EntropyMLBehavioralContract.cs` |
| BehavioralContract       | `Examples/ExampleTVAE/EntropyMLBehavioralContract.cs` |
| Guardrail suite          | `Examples/ExampleTVAE/ExampleTVAEGuardrails.cs`      |
| PSC example (train+test) | `Examples/ExampleTVAE/ExampleTVAEv1.cs`              |
| Entry point              | `Examples/ExampleTVAE/Program.cs`                    |
| Canonical specification  | `doc/EntropyML_HMD.md`                                |

---

## How to Add a New Model

1. Create `EntropyML.MyModel/` as a new class library project under `solution/EntropyML/` (code only; documentation goes in `doc/`).
2. Add `MyModel.cs` implementing `Fit`, `Encode`, `Reconstruct`, `ComputeLoss`.
3. Add `EntropyML.MyModel.csproj` referencing `EntropyML.Data` and `EntropyML.NN`.
4. Add `MyModelAdapter : ILatentModel` in `EntropyMLBehavioralContract.cs` (or a new file).
5. Create `Examples/ExampleMyModel/` as a new executable project.
6. Add a project reference to `EntropyML.MyModel` in `ExampleTVAE.csproj` (or the new example's csproj).
7. Register the new adapter in `RunBehavioralContractGuardrails()`.
8. Run `Guardrails.Run()` — all 235 existing checks plus the new model's contract checks must pass.

For detailed steps see [EntropyML_QuickStart.md § 4](EntropyML_QuickStart.md#4-onboard-a-new-model--five-steps).

---

*End of EntropyML Folder Structure v1.0 — canonical spec: [EntropyML_HMD.md](EntropyML_HMD.md)*
