# EntropyML Hierarchical Model Document (HMD)

**Specification:** EntropyML HCMD Specification  
**Version:** 2.0  
**Date:** 2026-08-13
**Status:** Validated — Steps 1–8 STS complete, 235/235 guardrails passing  
**Scope:** VAE, TVAE, and all future EntropyML latent models  
**Audience:** Model developers, scientific engineers, CI maintainers, future contributors

---

## Abstract

EntropyML is a framework for building thermodynamically-grounded latent variable models
of physical systems. This document is the **Hierarchical Model Document (HMD)** — the
single source of truth for the EntropyML specification, combining semantic meaning,
structural invariants, behavioral correctness criteria, numerical validation baselines,
regression protection, and cross-model consistency rules into one navigable reference.
The document is organised as a layered hierarchy: the Meaning Layer defines what
concepts *are*, the Structure Layer defines how they are *organised*, the Behavioral
Contract defines what *correct behavior* is, and the Validation and Guardrail layers
define how correctness is *measured and protected*. The HCMD methodology (Hierarchical
Conceptual Model Development) is used throughout to ensure that every implementation
decision traces back to a scientific concept, and every scientific concept is enforced
by an automated invariant. The specification applies uniformly to the VAE and TVAE
models currently in the system, and defines the onboarding contract for any future
EntropyML latent model.

---

## Change Log

| Version | Date | Summary                                                                                                                                                                        |
| ------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1.0     | 2026 | Initial HMD: Glossary, Meaning, Structure, Contract, Validation, Guardrails, Cross-Model                                                                                       |
| 2.0     | 2026 | Step 8 STS: Front matter, ToC, diagrams, cross-references, stance preface, onboarding guide, scientific appendices, glossary expansion, regressions appendix, stable numbering |

---

## EntropyML Stance

> *Why does EntropyML exist?*

Physical systems are governed by thermodynamic principles: energy is minimised, entropy
is maximised, and equilibrium is the state where these competing drives are balanced.
Conventional machine learning models — including VAEs — capture this balance implicitly,
encoded in the ELBO objective. EntropyML makes this balance **explicit**: every component
of the model, every variable name, every logged quantity carries a thermodynamic
interpretation.

This matters for three reasons:

**1. Scientific legibility.** When a physicist reads the training output of a TVAE,
they see `FreeEnergy=0.32  Energy=0.13  Entropy=0.19`. These are not abstract loss
numbers — they are physically interpretable quantities with known thermodynamic
relationships.

**2. Structural clarity.** The thermodynamic framing enforces architectural discipline.
The encoder is an **equilibrium finder** and an **entropy estimator**. The decoder is
a **relaxation operator**. This vocabulary makes the separation of concerns precise
and permanent.

**3. Formal correctness.** The thermodynamic interpretation provides a natural
correctness criterion: the free energy must be minimised, the decomposition identity
must hold, the latent geometry must contract. These are not heuristics — they are
theorems about the variational objective, expressed as automated guardrails.

The HCMD methodology structures model development around this principle: meaning comes
first, structure follows from meaning, and behavioral correctness is derived from
structure. Every change must satisfy the contract. The HMD is the artifact that makes
this traceable.

---

## Table of Contents

- [1. Core Glossary](#1-core-glossary)
- [2. Meaning Layer (AML)](#2-meaning-layer-aml)
  - [2.1 What a latent model is](#21-what-a-latent-model-is)
  - [2.2 What latent variables represent](#22-what-latent-variables-represent)
  - [2.3 What reconstruction means](#23-what-reconstruction-means)
  - [2.4 What regularization means](#24-what-regularization-means)
  - [2.5 What free-energy / ELBO means](#25-what-free-energy--elbo-means)
  - [2.6 What equilibrium and entropyPotential mean](#26-what-equilibrium-and-entropypotential-mean)
- [3. Structure Layer (PSC)](#3-structure-layer-psc)
  - [3.1 The six-step PSC flow](#31-the-six-step-psc-flow)
  - [3.2 Encoder/decoder separation](#32-encoderdecoder-separation)
  - [3.3 Dimensionality invariants](#33-dimensionality-invariants)
  - [3.4 No-explosion bounds](#34-no-explosion-bounds)
- [4. Behavioral Contract](#4-behavioral-contract)
  - [4.1 Convergence signature](#41-convergence-signature)
  - [4.2 Latent-geometry invariants](#42-latent-geometry-invariants)
  - [4.3 Reconstruction invariants](#43-reconstruction-invariants)
  - [4.4 Decomposition identity](#44-decomposition-identity)
  - [4.5 PSC flow invariants](#45-psc-flow-invariants)
  - [4.6 Stability envelope](#46-stability-envelope)
- [5. Validation Envelope](#5-validation-envelope)
  - [5.1 Baseline configuration](#51-baseline-configuration)
  - [5.2 Free-energy convergence bands](#52-free-energy-convergence-bands)
  - [5.3 Energy / entropy decomposition bands](#53-energy--entropy-decomposition-bands)
  - [5.4 Latent geometry bands](#54-latent-geometry-bands)
  - [5.5 Reconstruction bands](#55-reconstruction-bands)
  - [5.6 Extended validation bands](#56-extended-validation-bands)
- [6. Regression Guardrails](#6-regression-guardrails)
  - [6.1 Architecture](#61-architecture)
  - [6.2 GuardrailException](#62-guardrailexception)
  - [6.3 CI integration](#63-ci-integration)
  - [6.4 Adding future guardrails](#64-adding-future-guardrails)
- [7. Cross-Model Consistency](#7-cross-model-consistency)
  - [7.1 The adapter surface](#71-the-adapter-surface)
  - [7.2 VAE to TVAE translation table](#72-vae-to-tvae-translation-table)
  - [7.3 Qualitative alignment requirements](#73-qualitative-alignment-requirements)
  - [7.4 Requirements for future models](#74-requirements-for-future-models)
- [8. Future Model Onboarding Guide](#8-future-model-onboarding-guide)
  - [8.1 Step-by-step onboarding](#81-step-by-step-onboarding)
  - [8.2 Adapter template](#82-adapter-template)
  - [8.3 Satisfying the Behavioral Contract](#83-satisfying-the-behavioral-contract)
  - [8.4 Updating the Validation Envelope](#84-updating-the-validation-envelope)
- [9. Scientific Guarantees](#9-scientific-guarantees)
  - [9.1 Decomposition identity (formal)](#91-decomposition-identity-formal)
  - [9.2 Latent contraction (formal)](#92-latent-contraction-formal)
  - [9.3 Stability envelope (formal)](#93-stability-envelope-formal)
  - [9.4 Convergence signature (formal)](#94-convergence-signature-formal)
- [10. Glossary Expansion](#10-glossary-expansion)
- [11. Regressions and Violations](#11-regressions-and-violations)
- [Appendix: File Map](#appendix-file-map)

---

## 1. Core Glossary

*See also [Section 10](#10-glossary-expansion) for the full expanded reference.*

| EntropyML Term        | VAE Analogue                  | Thermodynamic Meaning                                                        |
| --------------------- | ----------------------------- | ---------------------------------------------------------------------------- |
| `equilibrium`         | latent mean mu                | Most probable latent state; free-energy minimum in latent space              |
| `entropyPotential`    | latent log-variance logSigma2 | Log of thermal fluctuation amplitude; encodes uncertainty / entropy capacity |
| `microstate`          | latent sample z               | A specific realisation drawn from the latent thermal distribution            |
| `relaxedMicrostate`   | reconstruction x-hat          | Macrostate recovered from a microstate via the relaxation operator           |
| `energyTerm`          | reconstruction loss           | MSE between observed and reconstructed macrostate; potential energy mismatch |
| `entropyTerm`         | beta * KL divergence          | Regularization toward prior; free entropy of the latent distribution         |
| `freeEnergy`          | ELBO                          | Helmholtz free energy: F = energyTerm + entropyTerm; the quantity minimised  |
| `EncodeThermoState`   | `Encode`                      | Maps macrostate to latent distribution parameters                            |
| `SampleMicrostate`    | `Sample`                      | Draws a microstate via the reparameterization trick                          |
| `RelaxToMicrostate`   | `Decode`                      | Maps a microstate to reconstructed macrostate                                |
| `RelaxAndReconstruct` | `Reconstruct`                 | Full forward pass: observe, encode, sample, relax                            |
| `ComputeFreeEnergy`   | `ComputeLoss`                 | Computes scalar thermodynamic objective for one sample                       |
| `Fit`                 | `Fit`                         | Runs the training loop; returns per-epoch objective history                  |
| `equilibriumEncoder`  | `encoder_mu`                  | Network mapping macrostate to equilibrium                                    |
| `entropyEncoder`      | `encoder_logvar`              | Network mapping macrostate to entropyPotential                               |
| `relaxationOperator`  | `decoder`                     | Network mapping microstate to reconstructed macrostate                       |
| `PSC`                 | PSC                           | Physical Structure Constraint: the shared six-step training and test flow    |
| `ILatentModel`        | —                             | Model-agnostic adapter interface; all EntropyML models must implement it     |
| `BehavioralContract`  | —                             | Unified model-agnostic correctness specification                             |
| `GuardrailException`  | —                             | Thrown when any invariant is violated; signals regression to CI              |

---

## 2. Meaning Layer (AML)

*Structure derived from this layer: [Section 3](#3-structure-layer-psc)*

### 2.1 What a latent model is

A EntropyML latent model is a **generative compression**. It learns a low-dimensional
representation (the *latent space*, or *manifold*) of a high-dimensional observation
space (the *macrostate space*). The model has two coupled roles:

- **Analysis** — given macrostate x, infer the latent distribution it came from.
- **Synthesis** — given latent draw z, generate a plausible macrostate x-hat.

In thermodynamic language: the model learns which microstates are thermally accessible
from each macrostate, and how to relax a microstate back to a macrostate.

### 2.2 What latent variables represent

| Concept             | VAE         | TVAE             | Physical meaning                                          |
| ------------------- | ----------- | ---------------- | --------------------------------------------------------- |
| Latent mean         | mu          | equilibrium      | Most probable latent position; the free-energy attractor  |
| Latent log-variance | logSigma2   | entropyPotential | Log of thermal agitation; breadth of the microstate cloud |
| Latent sample       | z           | microstate       | One specific draw from the thermal cloud                  |
| Latent dimension    | `LatentDim` | `ManifoldDim`    | Intrinsic degrees of freedom of the physical system       |

### 2.3 What reconstruction means

Reconstruction is the model's attempt to recover the original macrostate from a latent
microstate. Quality is measured by the **energy term** (MSE).

- Perfect reconstruction: zero energy term.
- Poor reconstruction: high energy term, lost structural information.

*Reconstruction invariants: [Section 4.3](#43-reconstruction-invariants)*

### 2.4 What regularization means

Regularization (KL divergence / entropyTerm) prevents latent space collapse. It pulls
the latent distribution toward a standard Gaussian prior, ensuring the latent space is
continuous, generative, and stable.

In thermodynamic language: the **free entropy** — the entropic cost of maintaining a
diffuse thermal distribution in latent space.

*Stability envelope: [Section 4.6](#46-stability-envelope)*

### 2.5 What free-energy / ELBO means

```
F = energyTerm + entropyTerm
  = reconstruction_loss + beta * KL
```

Minimising F simultaneously minimises energy (better reconstruction) and maximises
entropy (more diffuse, generative latent space). Beta controls the tradeoff (default 0.1).

*Decomposition identity: [Section 4.4](#44-decomposition-identity) and [Section 9.1](#91-decomposition-identity-formal)*

### 2.6 What equilibrium and entropyPotential mean

**equilibrium** (mu): the latent position minimising local free energy for a given
macrostate. The fixed point of latent relaxation dynamics.

**entropyPotential** (logSigma2): log of thermal fluctuation amplitude around
equilibrium. Large negative value (e.g. -2): well-localised, low entropy. Near zero:
highly diffuse, high entropy. Thermal agitation sigma = exp(0.5 * logSigma2).

*Latent geometry invariants: [Section 4.2](#42-latent-geometry-invariants)*

---

## 3. Structure Layer (PSC)

*Derived from: [Section 2](#2-meaning-layer-aml) | Enforced by: [Section 4](#4-behavioral-contract)*

The **Physical Structure Constraint** defines the invariant structural sequence that
all EntropyML models must follow.

### Diagram A — PSC Flow

```
+------------------------------------------------------------------+
|                      PSC Flow (6 Steps)                          |
+------------------------------------------------------------------+
|                                                                  |
|  Step 1: DATA GENERATION                                         |
|          MakeMeasuredData() --> float[][] X                      |
|          Multi-cluster physically realistic data                 |
|                         |                                        |
|                         v                                        |
|  Step 2: NORMALIZATION                                           |
|          DataGen.Normalize(X) --> float[][] Xn                   |
|          Zero-mean, unit-scale; required for stable training     |
|                         |                                        |
|                         v                                        |
|  Step 3: MODEL CONSTRUCTION                                      |
|          new TVAE(inputDim, latentDim, randSeed)                 |
|          Fixed seed; symmetric architecture; Adam optimizer      |
|                         |                                        |
|                         v                                        |
|  Step 4: TRAINING LOOP                                           |
|          for epoch in [0, 30):                                   |
|            Fit(Xn, epochs=1, verbose=0)                          |
|            Evaluate: energyTerm, entropyTerm, freeEnergy         |
|            Log: FreeEnergy=  Energy=  Entropy=  eq-range=        |
|                         |                                        |
|                         v                                        |
|  Step 5: TEST PASS                                               |
|          Encode test point P1 = [2.0, 0.5, -1.5]                 |
|          Log: raw input, norm input, equilibrium,                |
|               entropyPotential, relaxedMicrostate,               |
|               energyTerm, entropyTerm, freeEnergy                |
|                         |                                        |
|                         v                                        |
|  Step 6: VALIDATION                                              |
|          Guardrails.Run()                                        |
|          235 invariants checked; GuardrailException on failure   |
|                                                                  |
+------------------------------------------------------------------+
```

### Diagram B — Latent Geometry

```
  Epoch 0 (initial)     Epoch 15 (mid)       Epoch 29 (converged)

  +------------+         +--------+            +------+
  |            |         |        |            |      |
  |  .  .      |         |  . .   |            | . .  |
  | .  mu  .   |  --->   |  . mu .|  --->      | .mu. |
  |  .  .      |         |  . .   |            |      |
  |            |         |        |            +------+
  +------------+         +--------+
  Large sigma            Smaller sigma          Small sigma
  High entropy           Contracting            Stable basin

  |eq|_max ~ 2.7         |eq|_max ~ 2.2         |eq|_max ~ 2.0

  VAE:   mu (latent mean)         contracts over training
  TVAE:  equilibrium              contracts over training

  VAE:   logSigma2                stabilises in [-3, -1]
  TVAE:  entropyPotential         stabilises in [-3, -1]
```

### Diagram C — Decomposition Identity

```
  +------------------------------------------------------------+
  |                DECOMPOSITION IDENTITY                      |
  +------------------------------------------------------------+
  |                                                            |
  |  Objective = Reconstruction Component + Reg Component      |
  |                                                            |
  |  VAE:                                                      |
  |  ELBO       = recon_loss (MSE)     + beta * KL             |
  |                                                            |
  |  TVAE:                                                     |
  |  freeEnergy = energyTerm (MSE)     + entropyTerm           |
  |                                                            |
  |  Invariants:                                               |
  |  [1] reconComponent >= 0   (MSE is always non-negative)    |
  |  [2] regComponent   >= 0   (KL is always non-negative)     |
  |  [3] |obj - (recon + reg)| <= 0.01  (identity is exact)    |
  |                                                            |
  +------------------------------------------------------------+
```

### 3.1 The six-step PSC flow

See Diagram A. All six steps are mandatory for all EntropyML models. Steps may not be
reordered. The test pass (Step 5) must use the trained model from Step 4.

*PSC flow invariants: [Section 4.5](#45-psc-flow-invariants)*

### 3.2 Encoder/decoder separation

- **Encoder** maps macrostate to latent distribution parameters. Never touches decoder
  weights.
- **Decoder / relaxation operator** maps microstate to macrostate. Never touches encoder
  weights.
- **Latent sampling** bridges encoder and decoder via the reparameterization trick.

Model internals are private to `VAE.cs` and `TVAE.cs`. Examples and guardrails access
only the public API.

### 3.3 Dimensionality invariants

| Quantity           | Invariant                                                       |
| ------------------ | --------------------------------------------------------------- |
| `inputDim`         | Fixed at construction; cannot change at runtime                 |
| `latentDim`        | Fixed at construction; must be >= 1                             |
| Reconstruction dim | Must equal `inputDim` — always                                  |
| Encoder output dim | 2 x `latentDim` (one mu and one logSigma2 per latent dimension) |
| Decoder output dim | Must equal `inputDim`                                           |

### 3.4 No-explosion bounds (structural)

| Quantity                       | Bound           |
| ------------------------------ | --------------- |
| `                              | equilibrium     |
| `                              | logSigma2       |
| `                              | logSigma2       |
| Reconstruction MSE             | < 5.0           |
| FreeEnergy (trained, baseline) | in [0.25, 0.45] |

*Full validation bands: [Section 5](#5-validation-envelope)*

---

## 4. Behavioral Contract

*Derived from: [Section 3](#3-structure-layer-psc) | Enforced by: [Section 6](#6-regression-guardrails)*  
*Implemented in: `EntropyMLBehavioralContract.cs` — `BehavioralContract.Run(ILatentModel)`*

> **A EntropyML latent model is behaviorally correct if and only if it satisfies
> all six invariant groups defined in this section.**

All thresholds are **relative or structural** — no model-specific numbers appear
inside `BehavioralContract`. Applies uniformly to VAE, TVAE, and any future variant.

### 4.1 Convergence signature

*Formal statement: [Section 9.4](#94-convergence-signature-formal)*

| Invariant          | Relative expression              | Meaning                                |
| ------------------ | -------------------------------- | -------------------------------------- |
| Sharp initial drop | epoch1 < epoch0 x 0.90           | >= 10% relative drop by epoch 1        |
| Convergence        | final < epoch0 x 0.85            | >= 15% total drop by epoch 29          |
| Stable basin       | last-quarter window span <= 0.15 | No oscillation after initial transient |
| No NaN             | all epochs: obj is not NaN       | Numerical stability throughout         |
| No Inf             | all epochs: obj is not Inf       | No divergence                          |

### 4.2 Latent-geometry invariants

*Formal statement: [Section 9.2](#92-latent-contraction-formal)*

| Invariant            | Expression            | Meaning                                        |
| -------------------- | --------------------- | ---------------------------------------------- |
| Mean contraction     | `                     | eq                                             |
| No explosion         | `                     | eq                                             |
| Log-variance bounded | logSigma2 in [-8, +4] | Fluctuation amplitude finite and non-collapsed |

### 4.3 Reconstruction invariants

| Invariant      | Expression                   | Meaning                  |
| -------------- | ---------------------------- | ------------------------ |
| Dimensionality | `recon.Length == x.Length`   | Correct shape            |
| No NaN         | all i: `recon[i]` is not NaN | Reconstruction is real   |
| No Inf         | all i: `recon[i]` is not Inf | Reconstruction is finite |
| MSE bounded    | `MSE(x, recon) < 5.0`        | Not wildly wrong         |

After convergence (>= 30 epochs): trained reconstruction within +/-0.5 of normalised
input, per component.

### 4.4 Decomposition identity

*See Diagram C | Formal statement: [Section 9.1](#91-decomposition-identity-formal)*

```
objective = reconstructionComponent + regularizationComponent
```

| Invariant            | Expression            | Meaning                             |
| -------------------- | --------------------- | ----------------------------------- |
| Recon component >= 0 | `reconComponent >= 0` | MSE is always non-negative          |
| Reg component >= 0   | `regComponent >= 0`   | KL / entropy is always non-negative |
| Identity exact       | `                     | obj - (recon + reg)                 |

### 4.5 PSC flow invariants (structural)

*See Diagram A | Implemented as Section 5 of `BehavioralContract.Run()`*

| Phase              | Invariant                                              |
| ------------------ | ------------------------------------------------------ |
| Data generation    | `Xn.Length > 0`; each sample has `inputDim` components |
| Normalization      | All values in Xn satisfy `                             |
| Model construction | `inputDim > 0`; `latentDim > 0`                        |
| Training loop      | `history.Length == epochs`; no NaN epoch               |
| Per-epoch logging  | All components logged; thermodynamic terminology       |
| Test pass          | Reconstruction returned without exception              |

### 4.6 Stability envelope

*Formal statement: [Section 9.3](#93-stability-envelope-formal)*

| Invariant                   | Condition                                  |
| --------------------------- | ------------------------------------------ |
| No NaN in history           | all epochs: not NaN                        |
| No Inf in history           | all epochs: not Inf                        |
| No latent explosion         | `                                          |
| No reconstruction explosion | `MSE(x, recon) < 5.0`                      |
| No KL / entropy collapse    | regComponent > 0 for at least some samples |
| No oscillatory explosion    | last-quarter window span <= 0.15           |

---

## 5. Validation Envelope

*Derived from: [Section 4](#4-behavioral-contract) | Protected by: [Section 6](#6-regression-guardrails)*

### 5.1 Baseline configuration

```
Model:      TVAE (primary) and VAE (cross-model reference)
InputDim:   3
LatentDim:  2
Seed:       42
Epochs:     30
Data:       MakeMeasuredData() — 430 samples, 4 clusters
Beta:       0.1 (default)
LR:         0.001 (Adam)
Test point: P1 = [2.0, 0.5, -1.5] (normalised before use)
```

### 5.2 Free-energy convergence bands (TVAE baseline)

| Epoch            | Band         | Observed |
| ---------------- | ------------ | -------- |
| 0                | [0.40, 0.90] | ~0.62    |
| 1                | [0.25, 0.55] | ~0.37    |
| Final (epoch 29) | [0.25, 0.45] | ~0.32    |

### 5.3 Energy / entropy decomposition bands (epoch 29 averages)

| Term          | Band         | Observed average |
| ------------- | ------------ | ---------------- |
| `energyTerm`  | [0.05, 0.25] | ~0.09 to 0.16    |
| `entropyTerm` | [0.10, 0.35] | ~0.19 to 0.25    |

### 5.4 Latent geometry bands (trained)

| Quantity                            | Band         | Observed              |
| ----------------------------------- | ------------ | --------------------- |
| `                                   | equilibrium  | _max`                 |
| `entropyPotential` (trained, dim=2) | [-3.0, -1.0] | ~[-2.1, -2.2]         |
| Equilibrium range contraction       | init > final | init ~2.7, final ~2.0 |

### 5.5 Reconstruction bands (trained)

| Quantity                           | Band   | Observed      |
| ---------------------------------- | ------ | ------------- |
| Per-component deviation from input | <= 0.5 | Within +/-0.3 |
| Reconstruction NaN                 | Never  | Not observed  |
| Reconstruction Inf                 | Never  | Not observed  |

### 5.6 Extended validation bands (Step 5)

| Test                                | Band         | Notes                         |
| ----------------------------------- | ------------ | ----------------------------- |
| Multi-seed (42, 123, 999): final FE | [0.25, 0.45] | Same basin across seeds       |
| Distribution robustness: final FE   | [0.01, 5.00] | Wide safety band              |
| Distribution robustness: `          | eq           | `                             |
| Latent-dim (1-4): final FE          | [0.10, 0.80] | Wider for dim=1 (harder)      |
| Latent-dim (1-4): logSigma2         | [-5.0, +1.0] | No-explosion, not convergence |
| Long-horizon (300 epochs): final FE | [0.25, 0.45] | Same basin as 30 epochs       |
| Long-horizon: last-50-epoch window  | <= 0.15      | No slow drift                 |
| Long-horizon: avg entropy           | > 0.02       | No entropy collapse           |

---

## 6. Regression Guardrails

*Derived from: [Section 5](#5-validation-envelope) | Signals via: [Section 11](#11-regressions-and-violations)*

### 6.1 Architecture

```
ExampleTVAEGuardrails.cs       -- Step 4 + Step 5 model-specific checks
EntropyMLBehavioralContract.cs  -- Step 6 model-agnostic contract + adapters
Program.cs                     -- TrainTVAE() -> TestTVAE() -> Guardrails.Run()
```

`Guardrails.Run()` executes eight independent sections in sequence:

| Section  | Method                                  | Checks | Scope                                                        |
| -------- | --------------------------------------- | ------ | ------------------------------------------------------------ |
| Step 4a  | `RunBaselineTrainingGuardrails()`       | 47     | Epoch-by-epoch explosion, convergence, decomposition         |
| Step 4b  | `RunBaselineTestGuardrails()`           | 11     | Reconstruction, latent geometry, decomposition at test point |
| Step 5.1 | `RunMultiSeedGuardrails()`              | 15     | Seeds 42, 123, 999                                           |
| Step 5.2 | `RunDistributionRobustnessGuardrails()` | 30     | 5 distributions x 6 checks                                   |
| Step 5.3 | `RunLatentDimensionGuardrails()`        | 24     | Dims 1-4 x 6 checks                                          |
| Step 5.4 | `RunCrossModelConsistencyGuardrails()`  | 8      | VAE vs TVAE alignment                                        |
| Step 5.5 | `RunLongHorizonGuardrails()`            | 7      | 300 epochs stability                                         |
| Step 6   | `RunBehavioralContractGuardrails()`     | 93     | Contract x 4 model/seed combos                               |

**Total: 235 checks — all passing as of Step 8.**

### 6.2 GuardrailException

*Full operational details: [Section 11](#11-regressions-and-violations)*

```csharp
sealed class GuardrailException : Exception
{
    public int FailureCount { get; }
    // Message: "ExampleTVAE guardrails failed: N invariant(s) violated."
}
```

- All checks run before the exception is thrown (no early abort).
- Full `[PASS]` / `[FAIL]` list always printed before exception surfaces.
- Non-zero process exit code enables CI detection without log parsing.

### 6.3 CI integration

```
dotnet run --project Examples/ExampleTVAE/ExampleTVAE.csproj
```

This single command runs TrainTVAE(), TestTVAE(), and Guardrails.Run() (235 checks).
Exits 0 on full pass; exits non-zero on any failure. No separate test runner required.

### 6.4 Adding future guardrails

To add a **model-specific** check:

1. Add `RunXxx()` to `Guardrails` in `ExampleTVAEGuardrails.cs`.
2. Call it inside `Guardrails.Run()` before `PrintSummary()`.
3. Use the `Check(name, condition, detail)` helper.

To add a **new model** to the contract:

1. Implement `ILatentModel` in a new adapter (see [Section 8](#8-future-model-onboarding-guide)).
2. Add `BehavioralContract.Run(newAdapter, ...)` in `RunBehavioralContractGuardrails()`.

---

## 7. Cross-Model Consistency

*Uses: [Section 7.1](#71-the-adapter-surface) | Enforced by: [Section 6.1](#61-architecture)*

### 7.1 The adapter surface

`ILatentModel` is the single interface abstracting all EntropyML latent models:

```csharp
interface ILatentModel
{
    string ModelName { get; }
    int    InputDim  { get; }
    int    LatentDim { get; }

    List<float> Fit(float[][] X, int epochs, int verbose = 0);

    (float[] latentMean, float[] latentLogVar) EncodeDistribution(float[] x);

    float[] Reconstruct(float[] x);

    float ComputeObjective(float[] x, float[] recon,
                           float[] latentMean, float[] latentLogVar);
}
```

Current adapters:

| Adapter       | Wraps  | Key mappings                                                                                                                   |
| ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `VaeAdapter`  | `VAE`  | `Encode` -> `EncodeDistribution`; `Reconstruct` -> `Reconstruct`; `ComputeLoss` -> `ComputeObjective`                          |
| `TvaeAdapter` | `TVAE` | `EncodeThermoState` -> `EncodeDistribution`; `RelaxAndReconstruct` -> `Reconstruct`; `ComputeFreeEnergy` -> `ComputeObjective` |

### 7.2 VAE to TVAE translation table

| VAE concept          | TVAE concept                   | Physical interpretation                 |
| -------------------- | ------------------------------ | --------------------------------------- |
| Reconstruction loss  | Energy term                    | Potential energy mismatch               |
| beta * KL divergence | Entropy term                   | Free entropy of the latent distribution |
| ELBO                 | Free energy                    | Total thermodynamic cost                |
| mu                   | Equilibrium                    | Fixed point of latent relaxation        |
| logSigma2            | Entropy potential              | Log of thermal agitation amplitude      |
| z                    | Microstate                     | Realised latent configuration           |
| x-hat                | Relaxed microstate             | Recovered macrostate                    |
| Encoder              | Equilibrium + entropy encoders | Macrostate -> latent distribution       |
| Decoder              | Relaxation operator            | Microstate -> macrostate                |

### 7.3 Qualitative alignment requirements

For VAE and TVAE to be considered aligned:

1. Both converge to the same final objective band: [0.25, 0.45].
2. Both show latent contraction: `|latentMean|_max` decreases epoch0 -> epoch29.
3. Both produce coherent reconstructions: within +/-0.5 of normalised test input.
4. Both converge: `final < epoch0 - 0.05`.
5. Both satisfy the full `BehavioralContract` (Section 4).

### 7.4 Requirements for future EntropyML models

*Full guide: [Section 8](#8-future-model-onboarding-guide)*

Any future EntropyML latent model must:

1. Implement `ILatentModel` via a thin adapter.
2. Pass `BehavioralContract.Run()` for at least two seeds.
3. Follow the PSC flow in its corresponding `ExampleXxx` project.
4. Use thermodynamic terminology in per-epoch logs.
5. Be registered in `RunBehavioralContractGuardrails()`.
6. Not modify `VAE.cs`, `TVAE.cs`, `BehavioralContract`, or `ILatentModel`.

---

## 8. Future Model Onboarding Guide

*Required by: [Section 7.4](#74-requirements-for-future-models) | Uses: [Section 4](#4-behavioral-contract)*

### 8.1 Step-by-step onboarding

```
Step 1: Create the model class
        e.g. BetaTVAE.cs in EntropyML.BetaTVAE/
        Implement training, encoding, decoding, and objective computation.
        Do NOT expose internal weights, gradients, or layer structure.

Step 2: Create the adapter
        Add a sealed class implementing ILatentModel.
        Map the model's public API to the four adapter methods.
        See Section 8.2 for the adapter template.

Step 3: Create the Example project
        Add ExampleBetaTVAE/ following the PSC six-step flow.
        Mirror ExampleTVAEv1.cs structure exactly.
        Use thermodynamic terminology throughout.

Step 4: Register in guardrails
        In RunBehavioralContractGuardrails(), add:
        BehavioralContract.Run(new BetaTvaeAdapter(...), Xn, testN, ...)
        Run at seeds 42 and 999 at minimum.

Step 5: Run the contract
        dotnet run --project ExampleBetaTVAE/ExampleBetaTVAE.csproj
        All 235 existing checks must still pass.
        The new model's contract checks must also pass.

Step 6: Calibrate the Validation Envelope
        Record epoch-0, epoch-1, and final objective values.
        Derive model-specific bands (see Section 8.4).
        Add model-specific guardrail methods if needed.

Step 7: Update this document
        Add the new model to the Cross-Model Consistency tables (Section 7).
        Record its validation bands in Section 5.
        Increment the version and change log.
```

### 8.2 Adapter template

```csharp
sealed class MyModelAdapter : ILatentModel
{
    readonly MyModel _model;

    public string ModelName => "MyModel";
    public int    InputDim  => _model.InputDim;
    public int    LatentDim => _model.LatentDim;

    public MyModelAdapter(int inputDim, int latentDim, int randSeed = 42)
        => _model = new MyModel(inputDim, latentDim, randSeed: randSeed);

    public List<float> Fit(float[][] X, int epochs, int verbose = 0)
        => _model.Fit(X, epochs: epochs, verbose: verbose);

    public (float[] latentMean, float[] latentLogVar) EncodeDistribution(float[] x)
    {
        var (mean, logvar) = _model.Encode(x);
        return (mean, logvar);
    }

    public float[] Reconstruct(float[] x) => _model.Reconstruct(x);

    public float ComputeObjective(float[] x, float[] recon,
                                  float[] latentMean, float[] latentLogVar)
        => _model.ComputeLoss(x, recon, latentMean, latentLogVar);
}
```

Rules: thin wrapper only — no logic, no caching, no state. No model internals exposed.

### 8.3 Satisfying the Behavioral Contract

`BehavioralContract.Run()` tests all six invariant groups using only the four adapter
methods. No model-specific code is needed in the contract.

Common failure modes:

| Failure                         | Likely cause                      | Remedy                                          |
| ------------------------------- | --------------------------------- | ----------------------------------------------- |
| Sharp initial drop not achieved | LR too low or bad initialisation  | Increase LR or check weight init                |
| No latent contraction           | Beta too low                      | Increase beta; verify KL gradient is non-zero   |
| logVar out of bounds            | Activation saturating             | Check final encoder activation (must be Linear) |
| Decomposition identity fails    | ComputeObjective formula mismatch | Verify reconComponent = MSE(x,recon)/inputDim   |
| Reconstruction explosion        | Decoder weights diverging         | Check Adam application; reduce LR               |

### 8.4 Updating the Validation Envelope

1. Run 10 independent seeds. Record epoch-0, epoch-1, and final objective.
2. Set `Epoch0_Band = [min(epoch0) * 0.8, max(epoch0) * 1.2]`.
3. Set `Final_Band = [min(final) * 0.8, max(final) * 1.2]`.
4. Set Energy/Entropy bands from epoch-29 averages +/- 50%.
5. Add model-specific `RunXxxBaselineGuardrails()` following the Step 4 pattern.
6. Run full guardrail suite. Confirm 0 failures before merging.

---

## 9. Scientific Guarantees

*Implemented in: `BehavioralContract.Run()` | Checked by: [Section 6](#6-regression-guardrails)*

### 9.1 Decomposition identity (formal)

**Statement.** For any EntropyML latent model M and any input x:

```
L(x; M) = R(x, x-hat) + beta * D(q || p)
```

where:

- `L(x; M)` is the scalar objective (freeEnergy / ELBO)
- `R(x, x-hat) = (1/D) * sum_i (x_i - x-hat_i)^2`  is the reconstruction component (MSE, >= 0)
- `D(q || p) = -0.5 * sum_j (1 + logSigma2_j - mu_j^2 - exp(logSigma2_j))`  is the KL divergence (>= 0)
- `beta > 0` is the regularization weight

**Invariants:**

- `R(x, x-hat) >= 0` — always, by definition of MSE.
- `D(q || p) >= 0` — always, by Gibbs' inequality.
- `|L - (R + beta*D)| <= 0.01` — holds to floating-point precision.

**EntropyML translation:**

- `R(x, x-hat)` is `energyTerm`
- `beta * D(q || p)` is `entropyTerm`
- `L(x; M)` is `freeEnergy`

### 9.2 Latent contraction (formal)

**Statement.** Let `mu_t(x)` be the latent mean after `t` epochs of training on Xn.
Define:

```
Delta(t) = max over x in Xn of max over j of |mu_t(x)_j|
```

**Invariant:**

```
Delta(T) < Delta(1)
```

The maximum absolute latent mean after convergence is strictly smaller than after
the first epoch. This reflects KL / entropy regularization pressure pulling the
latent distribution toward the zero-mean prior.

Note: Delta(0) is near zero (random initialisation). Delta(1) is typically larger
(the model begins to encode structure). Delta(T) is smaller (regularization tightens).
The invariant compares epoch-1 to epoch-T to exclude the transient growth phase.

### 9.3 Stability envelope (formal)

**Statement.** Let `h = [h_0, h_1, ..., h_{T-1}]` be the per-epoch objective history.
Let `W = [h_{floor(3T/4)}, ..., h_{T-1}]` be the last-quarter window. The stability
envelope holds if and only if:

```
[1] for all t: h_t is in R (not NaN, not +/-Inf)
[2] for all t: max over x of |mu_t(x)| < 5.0
[3] for all t: MSE(x, Reconstruct(x)) < 5.0
[4] max(W) - min(W) <= 0.15
[5] there exists x such that regComponent(x) > 0
```

Condition [5] ensures the regularization term has not collapsed to zero across all
samples (entropy / KL collapse).

### 9.4 Convergence signature (formal)

**Statement.** Let `h_0`, `h_1`, `h_T` be epoch-0, epoch-1, and final objective values.
The convergence signature holds if and only if:

```
[1] h_1  < h_0 * 0.90     (sharp initial drop: >= 10% relative by epoch 1)
[2] h_T  < h_0 * 0.85     (convergence: >= 15% total drop)
[3] max(W) - min(W) <= 0.15   (stable basin in last quarter)
[4] for all t: h_t not in {NaN, +/-Inf}   (no divergence)
```

All conditions are **relative** — independent of the absolute magnitude of the
objective. Applicable to any EntropyML model regardless of scale.

---

## 10. Glossary Expansion

*Core glossary: [Section 1](#1-core-glossary)*

### 10.1 EntropyML terms

| Term                  | Definition                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `equilibrium`         | Latent mean mu; fixed point of the latent relaxation dynamics for a given macrostate        |
| `entropyPotential`    | Latent log-variance logSigma2; log of thermal fluctuation amplitude around equilibrium      |
| `microstate`          | A specific draw z from the latent thermal distribution                                      |
| `macrostate`          | An observed data point x; the aggregate of underlying microstates                           |
| `relaxedMicrostate`   | The reconstruction x-hat produced by relaxing a microstate through the relaxation operator  |
| `energyTerm`          | Reconstruction MSE; potential energy mismatch between observed and reconstructed macrostate |
| `entropyTerm`         | beta * KL divergence; free entropy of the latent thermal distribution                       |
| `freeEnergy`          | Helmholtz free energy; the scalar objective F = energyTerm + entropyTerm                    |
| `thermalAgitation`    | sigma = exp(0.5 * logSigma2); std dev of the microstate cloud around equilibrium            |
| `ManifoldDim`         | Latent dimensionality; intrinsic degrees of freedom of the system                           |
| `MicrostateDim`       | Input dimensionality; dimension of the observation space                                    |
| `EncodeThermoState`   | TVAE: x -> (equilibrium, entropyPotential)                                                  |
| `SampleMicrostate`    | TVAE: (equilibrium, entropyPotential) -> microstate z via reparameterization                |
| `RelaxToMicrostate`   | TVAE: z -> relaxedMicrostate                                                                |
| `RelaxAndReconstruct` | TVAE: x -> relaxedMicrostate (full forward pass)                                            |
| `ComputeFreeEnergy`   | TVAE: (x, relaxedMicrostate, equilibrium, entropyPotential) -> scalar freeEnergy            |
| `equilibriumEncoder`  | TVAE network: x -> equilibrium                                                              |
| `entropyEncoder`      | TVAE network: x -> entropyPotential                                                         |
| `relaxationOperator`  | TVAE network: z -> relaxedMicrostate                                                        |
| `AverageFreeEnergy`   | TVAE utility: average freeEnergy over a dataset                                             |
| `ThermodynamicForce`  | TVAE utility: F approx -dFreeEnergy/dequilibrium approx -equilibrium                        |

### 10.2 VAE terms

| Term                  | Definition                                                                |
| --------------------- | ------------------------------------------------------------------------- |
| `mu`                  | Latent mean; centre of the latent Gaussian for a given input              |
| `logvar` (logSigma2)  | Latent log-variance; logarithm of the variance of the latent Gaussian     |
| `z`                   | Latent sample; a draw from the latent Gaussian via z = mu + sigma*epsilon |
| `x_recon`             | Reconstruction; decoder output given latent sample z                      |
| `recon_loss`          | Reconstruction loss; MSE(x, x-hat) / inputDim                             |
| `kl_loss`             | KL divergence; -0.5 * sum_j (1 + logSigma2_j - mu_j^2 - exp(logSigma2_j)) |
| `ELBO`                | Evidence Lower BOund; recon_loss + beta * kl_loss; the VAE objective      |
| `Beta`                | KL weight; controls energy-entropy tradeoff (default 0.1)                 |
| `encoder_mu`          | VAE network: x -> mu                                                      |
| `encoder_logvar`      | VAE network: x -> logSigma2                                               |
| `decoder`             | VAE network: z -> x-hat                                                   |
| `Encode`              | VAE: x -> (mu, logSigma2)                                                 |
| `Sample`              | VAE: (mu, logSigma2) -> z via reparameterization                          |
| `Decode`              | VAE: z -> x-hat                                                           |
| `Reconstruct`         | VAE: x -> x-hat (full forward pass)                                       |
| `ComputeLoss`         | VAE: (x, x-hat, mu, logSigma2) -> scalar ELBO                             |
| `Transform`           | VAE: dataset -> latent means (deterministic, for analysis)                |
| `ReconstructionError` | VAE: average reconstruction MSE over a dataset                            |

### 10.3 Physical analogues

| Physical concept              | VAE analogue               | TVAE analogue            |
| ----------------------------- | -------------------------- | ------------------------ |
| Macrostate                    | Input x                    | Input x                  |
| Microstate                    | Latent sample z            | Microstate z             |
| Equilibrium state             | Latent mean mu             | Equilibrium              |
| Thermal fluctuation amplitude | sigma = exp(0.5*logSigma2) | Thermal agitation        |
| Entropic cost                 | beta * KL                  | Entropy term             |
| Potential energy              | Reconstruction loss        | Energy term              |
| Helmholtz free energy         | ELBO                       | Free energy              |
| Latent manifold               | Latent space               | Manifold                 |
| Degrees of freedom            | LatentDim                  | ManifoldDim              |
| Observation space             | InputDim                   | MicrostateDim            |
| Thermal relaxation            | Decoder pass               | Relaxation operator pass |

### 10.4 Adapter surface definitions

| Symbol                                               | Type                          | Meaning                                                     |
| ---------------------------------------------------- | ----------------------------- | ----------------------------------------------------------- |
| `ILatentModel`                                       | Interface                     | Model-agnostic surface; all EntropyML models must implement |
| `ModelName`                                          | `string`                      | Human-readable model identifier                             |
| `InputDim`                                           | `int`                         | Dimension of the observation space                          |
| `LatentDim`                                          | `int`                         | Dimension of the latent space                               |
| `Fit(X, epochs, verbose)`                            | `List<float>`                 | Train; returns per-epoch objective history                  |
| `EncodeDistribution(x)`                              | `(float[], float[])`          | Returns (latentMean, latentLogVar)                          |
| `Reconstruct(x)`                                     | `float[]`                     | Full forward pass; returns reconstruction                   |
| `ComputeObjective(x, recon, mean, logvar)`           | `float`                       | Returns scalar objective for one sample                     |
| `VaeAdapter`                                         | `sealed class : ILatentModel` | Wraps VAE; maps VAE API to adapter surface                  |
| `TvaeAdapter`                                        | `sealed class : ILatentModel` | Wraps TVAE; maps TVAE API to adapter surface                |
| `BehavioralContract.Run(model, Xn, testSample, ...)` | `static void`                 | Runs 6 invariant groups against any ILatentModel            |

---

## 11. Regressions and Violations

*Exception type: `GuardrailException` | Triggered from: [Section 6.2](#62-guardrailexception)*

### 11.1 What constitutes a regression

A regression is any change causing one or more of the 235 guardrail checks to fail:

| Category                  | Examples                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| Numerical regression      | NaN or Inf in any objective, reconstruction, or latent value                               |
| Convergence regression    | Final objective rises above convergence band; sharp drop disappears                        |
| Geometry regression       | Latent means stop contracting; `                                                           |
| Decomposition regression  | `reconComponent` or `regComponent` becomes negative; identity fails                        |
| Reconstruction regression | NaN/Inf; MSE exceeds ceiling; dimensionality mismatch                                      |
| Stability regression      | Last-quarter window span exceeds 0.15; entropy collapses to near-zero                      |
| Terminology regression    | Log output loses thermodynamic terms (not caught by guardrails; must be reviewed manually) |

### 11.2 How GuardrailException is thrown

```
Guardrails.Run()
  |
  +-- RunBaselineTrainingGuardrails()   -- each Check() calls _fail++ on failure
  +-- RunBaselineTestGuardrails()
  +-- RunMultiSeedGuardrails()
  +-- RunDistributionRobustnessGuardrails()
  +-- RunLatentDimensionGuardrails()
  +-- RunCrossModelConsistencyGuardrails()
  +-- RunLongHorizonGuardrails()
  +-- RunBehavioralContractGuardrails()
  |
  +-- PrintSummary()
       +-- prints "Guardrail result: N/235 passed"
       +-- prints "M FAILURE(s) detected"      (if M > 0)
       +-- throws GuardrailException(M)         (if M > 0)
```

Every failing check is printed before the exception is raised. No early abort.

### 11.3 How CI detects regressions

- **Exit 0**: all 235 checks passed — no regression.
- **Exit non-zero**: at least one check failed — regression detected.

No test runner, test project, or assertion framework is required.

### 11.4 How to interpret failures

Each failure line:

```
  [FAIL] <invariant name>  (<detail with actual values>)
```

Example:

```
  [FAIL] Final FreeEnergy in [0.25, 0.45]  (actual=0.52)
  [FAIL] No latent explosion (|eq| < 5)   (|eq|_max=6.31)
  [FAIL] Reconstruction: no NaN
```

Diagnosis steps:

1. Identify the category (convergence, geometry, reconstruction, etc.).
2. Identify which recent change could affect that category.
3. Check whether the failure is stochastic (run again with same seed to confirm).
4. Check whether the failure is cross-model (both VAE and TVAE fail -> shared
   utility change) or model-specific (only TVAE fails -> check TVAE.cs).

### 11.5 How to fix violations

| Violation type               | Investigation path                                 | Common fixes                                                          |
| ---------------------------- | -------------------------------------------------- | --------------------------------------------------------------------- |
| NaN in objective             | Gradient explosion; log of zero; division by zero  | Clip gradients; add epsilon to log; check normalisation               |
| Convergence band exceeded    | LR or beta misconfigured; architecture change      | Revert change; re-tune beta; reduce LR                                |
| Latent explosion             | Beta too low                                       | Increase beta; verify KL gradient flows through both encoder branches |
| Decomposition identity fails | ComputeObjective formula inconsistent with adapter | Verify MSE uses /inputDim; verify KL sign convention                  |
| Reconstruction NaN           | Activation explosion in decoder                    | Check Tanh saturation; check for NaN in latent sample                 |
| Entropy collapse             | Beta too high                                      | Decrease beta; verify entropyTerm > 0 on at least some samples        |
| Stable basin not achieved    | Model still in transient at epoch 29               | Increase epochs; lower LR; check data normalisation                   |

---

## Appendix: File Map

```
solution/EntropyML/
 +-- EntropyML_HMD.md                         (this document)
 +-- EntropyML.VAE/
 |    +-- VAE.cs                             (VAE model; do not modify)
 +-- EntropyML.TVAE/
 |    +-- TVAE.cs                            (TVAE model; do not modify)
 +-- EntropyML.Data/                          (DataGen, Normalize, Measurements)
 +-- EntropyML.NN/                            (Sequential, DenseLayer, Adam)
 +-- Examples/
      +-- ExampleTVAE/
           +-- ExampleTVAEv1.cs              (TrainTVAE() + TestTVAE(); PSC Steps 1-5)
           +-- ExampleTVAEGuardrails.cs      (Step 4 + Step 5 model-specific guardrails)
           +-- EntropyMLBehavioralContract.cs (ILatentModel, adapters, BehavioralContract)
           +-- Program.cs                    (Entry: TrainTVAE -> TestTVAE -> Guardrails)
```

---

## Derived Artifacts

The following documents are derived from or extend this HMD. They summarise, reference,
or distribute it but do **not** supersede it. This HMD remains the single authoritative source.

### Specification derivatives (summary / reference)

| Artifact                                           | Role                           | Audience                         |
| -------------------------------------------------- | ------------------------------ | -------------------------------- |
| [EntropyML_SpecLite.md](EntropyML_SpecLite.md)     | 2-page executive summary       | Technical leads, reviewers       |
| [EntropyML_QuickStart.md](EntropyML_QuickStart.md) | Developer onboarding guide     | Contributors, new model authors  |
| [EntropyML_API_Sheet.md](EntropyML_API_Sheet.md)   | Single-page API reference card | Developers implementing adapters |

### Entry points

| Artifact                                     | Role                              | Audience                |
| -------------------------------------------- | --------------------------------- | ----------------------- |
| [README.md](../README.md)                    | GitHub front door                 | All visitors            |
| [EntropyML_Landing.md](EntropyML_Landing.md) | Narrative conceptual introduction | Researchers, architects |

### Operational artifacts

| Artifact                                                     | Role                       | Audience                      |
| ------------------------------------------------------------ | -------------------------- | ----------------------------- |
| [EntropyML_FolderStructure.md](EntropyML_FolderStructure.md) | Annotated project tree     | Contributors                  |
| [EntropyML_Versioning.md](EntropyML_Versioning.md)           | Semantic versioning policy | Maintainers, release managers |
| [EntropyML_ReleaseBundle.md](EntropyML_ReleaseBundle.md)     | Current release artifact   | Integrators, reviewers        |

---

*End of EntropyML HMD v2.0*
