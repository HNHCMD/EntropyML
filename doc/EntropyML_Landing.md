# EntropyML — Thermodynamic Meaning Layer for Latent Models

**Version:** 1.0  
**Date:** 2026-08-13  
**Scope:** Conceptual introduction to EntropyML  
**Audience:** Researchers, architects, technical leads, new contributors  
**Canonical reference:** [EntropyML_HMD.md](EntropyML_HMD.md)  
**Onboarding:** [EntropyML_QuickStart.md](EntropyML_QuickStart.md) · [EntropyML_API_Sheet.md](EntropyML_API_Sheet.md) · [EntropyML_SpecLite.md](EntropyML_SpecLite.md)

---

## Why EntropyML?

Standard variational autoencoders are trained to maximise the **E**vidence **L**ower **BO**und
(ELBO). The ELBO decomposes into a reconstruction term and a KL-regularisation term.
These terms are correct, well-studied, and effective — but they carry no physical
meaning on their own.

EntropyML makes a different claim: **the ELBO is the Helmholtz free energy**, and
every component of it is a physically interpretable thermodynamic quantity. This is
not an analogy or a metaphor. It is a deliberate, enforced correspondence between the
mathematics of variational inference and the mathematics of classical thermodynamics:

```
 Statistical Inference         Statistical Thermodynamics
 ─────────────────────         ──────────────────────────
 Latent mean (mu)         ←→   Equilibrium state
 Log-variance (logSigma2) ←→   Entropy potential
 Reconstruction loss      ←→   Energy term (potential energy mismatch)
 KL divergence            ←→   Free entropy (thermal regularisation)
 ELBO                     ←→   Helmholtz free energy
 Fit() training loop      ←→   Free-energy minimisation
 Encode + Reconstruct     ←→   Thermalisation + Relaxation
```

The result is a framework where thermodynamic intuition, software correctness, and
scientific reproducibility are inseparable by construction.

---

## The PSC Flow

Every EntropyML model must follow the **Physical State Cycle** — a six-phase structural
sequence that mirrors thermodynamic preparation of a system:

```
  ┌─────────┐     ┌───────────┐     ┌───────────┐
  │  Data   │────▶│ Normalize │────▶│ Construct │
  └─────────┘     └───────────┘     └─────┬─────┘
                                          │
                  ┌───────────┐     ┌─────▼─────┐
                  │ Validate  │◀────│   Test    │
                  └───────────┘     └─────▲─────┘
                                          │
                                    ┌─────┴─────┐
                                    │   Train   │
                                    └───────────┘
```

1. **Data** — raw thermodynamic measurements are collected  
2. **Normalize** — the microstate space is standardised  
3. **Construct** — the latent model is instantiated  
4. **Train** — free-energy minimisation drives the system toward equilibrium  
5. **Test** — the system is probed with held-out microstates  
6. **Validate** — the behavioral contract and guardrail suite confirm all invariants  

No compliant model may skip or reorder these phases.

---

## Latent Geometry

At convergence, the latent manifold of a EntropyML model has a specific thermodynamic
shape:

```
  Epoch 0 (initial)               Epoch T (converged)
  ─────────────────               ───────────────────

  ·  ·   ·    ·  ·                     ·  ·
 ·  ·  ·   ·   ·  ·                  · ·  · ·
  ·   ·       ·  ·          →           · ·
   ·    ·  ·     ·                    · · ·
     ·   ·  ·  ·                        · ·

  Equilibria scattered,            Equilibria contracted,
  no thermal structure             well-separated, bounded
```

The behavioral contract enforces:

- Latent means **contract** from epoch 0 to convergence (no expansion)
- No latent explosion (`|mean| < 5.0` across all dimensions)
- Log-variances remain bounded in `[-8, +4]`
- Entropy potential decreases as the system thermalises

---

## The Decomposition Identity

Every EntropyML model must satisfy, for every input sample:

```
  FreeEnergy  =  EnergyTerm  +  EntropyTerm

  where:
    EnergyTerm   = (1/D) * Σ (x_i - recon_i)²       ≥ 0
    EntropyTerm  = beta * KL(q(z|x) ‖ p(z))          ≥ 0
    FreeEnergy   = EnergyTerm + EntropyTerm           ≥ 0
```

This is not a convention — it is a hard invariant checked at runtime. Any model
implementation that violates decomposition identity fails the guardrail suite and
must not be deployed.

---

## Scientific Stance

EntropyML takes the following position:

> The correct semantics of a variational autoencoder are thermodynamic, not
> purely statistical. Encoding is thermalisation. Reconstruction is relaxation.
> The ELBO is free energy. Training is minimisation of free energy. This
> correspondence is exact within the AML (Abstract Meaning Layer), and any
> deviation from it constitutes a behavioral contract violation.

This stance is enforced computationally, not merely asserted. The 235-check guardrail
suite, the adapter interface, and the behavioral contract together constitute a
machine-checkable proof of thermodynamic consistency for every model in the framework.

---

## Cross-Model Consistency

EntropyML enforces that all compliant models produce **equivalent thermodynamic
behaviour** on equivalent inputs, regardless of internal implementation:

- All models converge (objective drops; no explosion or collapse)
- All models produce bounded latent geometry (contracted equilibria, bounded entropy potentials)
- All models satisfy decomposition identity (energy + entropy = free energy)
- All models satisfy PSC flow (all six phases complete without error)
- All models pass the stability envelope (no NaN, no Inf, across seeds, dimensions, and distributions)

The `ILatentModel` adapter interface and `BehavioralContract.Run(...)` method are the
technical mechanisms that enforce this equivalence across the VAE and TVAE families,
and will extend to all future models added to EntropyML.

---

## HCMD Artifact Pyramid

The EntropyML documentation is organised as a **Hierarchical Canonical Model Document**
pyramid. Each layer is more detailed and more authoritative than the one above it:

```
                  ┌───────────────────┐
                  │  README / Landing │  Entry points
                  └────────┬──────────┘
                           │
              ┌────────────▼─────────────┐
              │  Spec-Lite / Quick-Start  │  Derivative guides
              │       API Sheet           │
              └────────────┬─────────────┘
                           │
                  ┌────────▼────────┐
                  │   EntropyML HMD  │  Canonical specification
                  └─────────────────┘
                  (authoritative source)
```

| Layer        | Document                                                                                                       | Authority           |
| ------------ | -------------------------------------------------------------------------------------------------------------- | ------------------- |
| Entry        | [README.md](../README.md) · this page                                                                       | Orientation         |
| Guide        | [Spec-Lite](EntropyML_SpecLite.md) · [Quick-Start](EntropyML_QuickStart.md) · [API Sheet](EntropyML_API_Sheet.md) | Summary / Reference |
| Canonical    | [EntropyML HMD](EntropyML_HMD.md)                                                                                | **Authoritative**   |
| Operational  | [Folder Structure](EntropyML_FolderStructure.md) · [Versioning](EntropyML_Versioning.md)                         | Governance          |
| Distribution | [Release Bundle](EntropyML_ReleaseBundle.md)                                                                    | Publication         |

---

*End of EntropyML Landing Page v1.0 — canonical spec: [EntropyML_HMD.md](EntropyML_HMD.md)*
