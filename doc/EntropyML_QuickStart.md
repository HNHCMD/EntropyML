# EntropyML Quick-Start Guide

**Specification:** EntropyML HCMD Specification — Developer Onboarding  
**Version:** 1.0  
**Date:** 2026-08-13  
**Scope:** VAE, TVAE, and all future EntropyML latent models  
**Audience:** Developers onboarding to EntropyML; contributors adding new models  
**Canonical reference:** [EntropyML_HMD.md](EntropyML_HMD.md)  
**API reference:** [EntropyML_API_Sheet.md](EntropyML_API_Sheet.md)  
**Executive summary:** [EntropyML_SpecLite.md](EntropyML_SpecLite.md)

---

## What is EntropyML?

EntropyML is a cross‑platform C# implementation of thermodynamically‑grounded latent variable models. It gives standard VAE components physical names and enforces their correctness

```
VAE term          EntropyML term          Physical meaning
-----------------------------------------------------------------
latent mean mu    equilibrium            Free-energy attractor
logSigma2         entropyPotential       Log of thermal agitation
recon loss        energyTerm             Potential energy mismatch
beta * KL         entropyTerm            Free entropy of latent dist.
ELBO              freeEnergy             Helmholtz free energy
```

The framework contains two models (`VAE`, `TVAE`), one behavioral contract
(`BehavioralContract`), and one adapter interface (`ILatentModel`) that unifies them.

---

## 1. Run ExampleTVAE

```
cd S:\GitHub\EntropyML
dotnet run --project solution\EntropyML\Examples\ExampleTVAE\ExampleTVAE.csproj
```

Expected output (abbreviated):

```
** ExampleTVAE.TrainTVAE (Realistic Data) **
Epoch  0  FreeEnergy=0.6202  Energy=0.1538  Entropy=0.2527  eq-range=[-2.67,2.96]
...
Epoch 29  FreeEnergy=0.3189  Energy=0.1177  Entropy=0.1970  eq-range=[-1.82,2.01]

Test Input (raw):                    [2.000, 0.500, -1.500]
Latent mu (equilibrium):             [-1.199, 1.261]
Latent logSigma2 (entropy potential):[-2.084, -2.173]
Energy/Reconstruction (norm):        [1.726, 0.907, -1.574]
Energy term:                         0.0350
Entropy term (Beta*KL):              0.2762
FreeEnergy (energy + entropy):       0.3112

** ExampleTVAE.TestTVAE **
...

====================================================
  ExampleTVAE Guardrails (Step 4 + Step 5 + Step 6 STS)
====================================================
...
  Guardrail result: 235/235 passed
====================================================
```

Three things run in one command: `TrainTVAE()`, `TestTVAE()`, `Guardrails.Run()`.

---

## 2. Understand the PSC Flow

Every EntropyML model follows this six-step sequence:

```
Data --> Normalize --> Construct --> Train --> Test --> Validate
```

In code (`ExampleTVAEv1.cs`):

```csharp
var X  = MakeMeasuredData();                        // Step 1
var Xn = DataGen.Normalize(X);                      // Step 2
var tvae = new TVAE(inputDim: 3, latentDim: 2);     // Step 3

for (int epoch = 0; epoch < 30; epoch++)            // Step 4
{
    tvae.Fit(Xn, epochs: 1, verbose: 0);
    // evaluate and log: FreeEnergy, Energy, Entropy, eq-range
}

var (eq, entPot) = tvae.EncodeThermoState(testN);   // Step 5
var recon = tvae.RelaxAndReconstruct(testN);

Guardrails.Run();                                   // Step 6
```

---

## 3. Interpret Guardrail Output

Each check emits one of:

```
  [PASS] <invariant name>
  [FAIL] <invariant name>  (<actual values>)
```

On any failure the run ends with:

```
  Guardrail result: N/235 passed
  M FAILURE(s) — regression detected
```

and a non-zero exit code. Common failures and their causes:

| Output                                       | Likely cause                      |
| -------------------------------------------- | --------------------------------- |
| `[FAIL] Final FreeEnergy in [0.25, 0.45]`    | Architecture or beta changed      |
| `[FAIL] No latent explosion`                 | Beta too low; KL not regularising |
| `[FAIL] Reconstruction: no NaN`              | Gradient explosion in decoder     |
| `[FAIL] Latent means contract over training` | Regularisation not active         |
| `[FAIL] FreeEnergy approx energy + entropy`  | ComputeObjective formula mismatch |

For the full diagnosis guide see [HMD Section 11](EntropyML_HMD.md#11-regressions-and-violations).

---

## 4. Onboard a New Model — Five Steps

### Step 4.1 — Create the model class

Create `EntropyML.MyModel/MyModel.cs`. Implement:

- `Fit(float[][] X, int epochs, int verbose)` returning `List<float>`
- `Encode(float[] x)` returning `(float[] mean, float[] logvar)`
- `Reconstruct(float[] x)` returning `float[]`
- `ComputeLoss(float[] x, float[] recon, float[] mean, float[] logvar)` returning `float`

Do **not** expose weights, gradients, or layer internals.

### Step 4.2 — Write the adapter

```csharp
sealed class MyModelAdapter : ILatentModel
{
    readonly MyModel _m;
    public string ModelName => "MyModel";
    public int    InputDim  => _m.InputDim;
    public int    LatentDim => _m.LatentDim;

    public MyModelAdapter(int inputDim, int latentDim, int randSeed = 42)
        => _m = new MyModel(inputDim, latentDim, randSeed: randSeed);

    public List<float> Fit(float[][] X, int epochs, int verbose = 0)
        => _m.Fit(X, epochs: epochs, verbose: verbose);

    public (float[] latentMean, float[] latentLogVar) EncodeDistribution(float[] x)
    {
        var (mean, logvar) = _m.Encode(x);
        return (mean, logvar);
    }

    public float[] Reconstruct(float[] x) => _m.Reconstruct(x);

    public float ComputeObjective(float[] x, float[] recon,
                                  float[] latentMean, float[] latentLogVar)
        => _m.ComputeLoss(x, recon, latentMean, latentLogVar);
}
```

Rule: the adapter is a **thin wrapper** — no logic, no state, no caching.

### Step 4.3 — Create the Example project

Copy `ExampleTVAE/` as `ExampleMyModel/`. Follow the PSC six-step flow exactly.
Use thermodynamic terminology in all logs (FreeEnergy, Energy, Entropy, eq-range).

### Step 4.4 — Register in the contract

In `ExampleTVAEGuardrails.cs`, inside `RunBehavioralContractGuardrails()`:

```csharp
var myModel  = new MyModelAdapter(inputDim: 3, latentDim: 2, randSeed: 42);
var myModel2 = new MyModelAdapter(inputDim: 3, latentDim: 2, randSeed: 999);
BehavioralContract.Run(myModel,  Xn, testN, ref _pass, ref _fail, "standard");
BehavioralContract.Run(myModel2, Xn, testN, ref _pass, ref _fail, "seed=999");
```

### Step 4.5 — Run and confirm

```
dotnet run --project solution\EntropyML\Examples\ExampleTVAE\ExampleTVAE.csproj
```

All existing 235 checks must still pass. The new model's contract checks must also pass.
Then calibrate the Validation Envelope — see [HMD Section 8.4](EntropyML_HMD.md#84-updating-the-validation-envelope).

---

## 5. The ILatentModel Interface at a Glance

For the full API mapping tables see [EntropyML_API_Sheet.md](EntropyML_API_Sheet.md).

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

`BehavioralContract.Run(model, Xn, testSample, ref pass, ref fail)` tests all
six invariant groups automatically against any `ILatentModel`.

---

## 6. The Behavioral Contract in One Sentence per Group

| Group                     | One sentence                                                           |
| ------------------------- | ---------------------------------------------------------------------- |
| 1. Convergence signature  | Objective drops by at least 10% in the first epoch and 15% total.      |
| 2. Latent geometry        | Latent means contract from epoch 0 to convergence; no explosion.       |
| 3. Reconstruction         | Output matches input dimensionality, is finite, and stays close.       |
| 4. Decomposition identity | Objective = reconComponent + regComponent; both non-negative.          |
| 5. PSC flow               | All six structural phases complete without error.                      |
| 6. Stability envelope     | No NaN, no Inf, no explosion, no collapse — across all configurations. |

---

## 7. Key Files

```
solution/EntropyML/
 +-- EntropyML_HMD.md                         Full specification (canonical)
 +-- EntropyML_SpecLite.md                    Executive summary
 +-- EntropyML_QuickStart.md                  This file
 +-- EntropyML_API_Sheet.md                   API reference card
 +-- Examples/ExampleTVAE/
      +-- ExampleTVAEv1.cs                   PSC Steps 1-5 (train + test)
      +-- ExampleTVAEGuardrails.cs           235-check guardrail suite
      +-- EntropyMLBehavioralContract.cs      ILatentModel + adapters + contract
      +-- Program.cs                         Entry point
```

---

*End of EntropyML Quick-Start v1.0 — canonical spec: [EntropyML_HMD.md](EntropyML_HMD.md)*
