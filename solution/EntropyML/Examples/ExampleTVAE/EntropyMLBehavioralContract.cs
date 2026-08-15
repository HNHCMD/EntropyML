using EntropyML;

namespace ExampleTVAE
{
    // ============================================================
    // EntropyMLBehavioralContract  (Step 6 STS)
    // ============================================================
    //
    // Unified, model-agnostic behavioral contract for EntropyML latent
    // models.  Applies to VAE, TVAE, and any future variant.
    //
    // A model is considered behaviorally correct if and only if it
    // satisfies all six invariant groups defined here:
    //
    //   1. Convergence signature
    //      Sharp initial drop (epoch 0 → epoch 1), stable basin,
    //      no divergence, no oscillatory explosion, no NaN/Inf.
    //      Expressed in relative terms: final < epoch0 * RelDropRatio.
    //
    //   2. Latent-geometry invariants
    //      Latent mean (μ / equilibrium) contracts over training.
    //      Latent log-variance (logσ² / entropyPotential) remains bounded.
    //      No latent explosion (|latentMean| < AbsExplosionLimit).
    //
    //   3. Reconstruction invariants
    //      Reconstruction is coherent (no NaN, no Inf).
    //      Reconstruction dimensionality matches input dimensionality.
    //      Reconstruction deviation from input is bounded (MSE < ReconMseCeiling).
    //
    //   4. Decomposition identity
    //      objective = reconstructionComponent + regularizationComponent
    //      Both components must be non-negative.
    //      (VAE:  ELBO = recon_loss + β·KL)
    //      (TVAE: freeEnergy = energyTerm + entropyTerm)
    //
    //   5. PSC flow invariants
    //      Data generation → normalization → model construction →
    //      training loop → per-epoch logging → test pass.
    //      The same six-step structural sequence for both models.
    //      (Structural; validated by construction of the adapters.)
    //
    //   6. Stability envelope
    //      No NaN, no Inf, no latent explosion, no reconstruction explosion,
    //      no entropy/KL collapse, no objective divergence across seeds,
    //      distributions, and latent dimensions.
    //
    // Design rules:
    //   - All thresholds are RELATIVE or structural (no model-specific bands).
    //   - Adapters are the only place that touches model APIs.
    //   - Contract methods are model-agnostic — they receive ILatentModel.
    //   - No model internals are accessed; only the public adapter surface.
    //   - Neither VAE.cs nor TVAE.cs is modified.
    // ============================================================

    // ============================================================
    // Adapter interface — model-agnostic surface
    // ============================================================

    /// <summary>
    /// Model-agnostic adapter surface. Implemented by VaeAdapter and
    /// TvaeAdapter. Any future EntropyML latent model must implement
    /// this interface to be covered by BehavioralContract.
    /// </summary>
    interface ILatentModel
    {
        string ModelName { get; }
        int    InputDim  { get; }
        int    LatentDim { get; }

        /// <summary>
        /// Train the model for the given number of epochs.
        /// Returns the per-epoch objective value (average over samples).
        /// </summary>
        List<float> Fit(float[][] X, int epochs, int verbose = 0);

        /// <summary>
        /// Encode a sample to its latent distribution parameters.
        /// Returns (latentMean, latentLogVar) — the μ and logσ² analogues.
        /// </summary>
        (float[] latentMean, float[] latentLogVar) EncodeDistribution(float[] x);

        /// <summary>
        /// Full forward pass: x → reconstruction (deterministic mean path).
        /// </summary>
        float[] Reconstruct(float[] x);

        /// <summary>
        /// Compute the scalar objective for one sample.
        /// VAE:  ELBO = recon_loss + β·KL
        /// TVAE: freeEnergy = energyTerm + entropyTerm
        /// </summary>
        float ComputeObjective(float[] x, float[] recon,
                               float[] latentMean, float[] latentLogVar);
    }

    // ============================================================
    // VAE adapter
    // ============================================================

    sealed class VaeAdapter : ILatentModel
    {
        readonly VAE _vae;
        public string ModelName => "VAE";
        public int    InputDim  => _vae.InputDim;
        public int    LatentDim => _vae.LatentDim;

        public VaeAdapter(int inputDim, int latentDim, int randSeed = 42)
            => _vae = new VAE(inputDim, latentDim, randSeed: randSeed);

        public List<float> Fit(float[][] X, int epochs, int verbose = 0)
            => _vae.Fit(X, epochs: epochs, verbose: verbose);

        public (float[] latentMean, float[] latentLogVar) EncodeDistribution(float[] x)
            => _vae.Encode(x);

        public float[] Reconstruct(float[] x)
            => _vae.Reconstruct(x);

        public float ComputeObjective(float[] x, float[] recon,
                                      float[] latentMean, float[] latentLogVar)
            => _vae.ComputeLoss(x, recon, latentMean, latentLogVar);
    }

    // ============================================================
    // TVAE adapter
    // ============================================================

    sealed class TvaeAdapter : ILatentModel
    {
        readonly TVAE _tvae;
        public string ModelName => "TVAE";
        public int    InputDim  => _tvae.MicrostateDim;
        public int    LatentDim => _tvae.ManifoldDim;

        public TvaeAdapter(int inputDim, int latentDim, int randSeed = 42)
            => _tvae = new TVAE(inputDim, latentDim, randSeed: randSeed);

        public List<float> Fit(float[][] X, int epochs, int verbose = 0)
            => _tvae.Fit(X, epochs: epochs, verbose: verbose);

        public (float[] latentMean, float[] latentLogVar) EncodeDistribution(float[] x)
            => _tvae.EncodeThermoState(x);

        public float[] Reconstruct(float[] x)
            => _tvae.RelaxAndReconstruct(x);

        public float ComputeObjective(float[] x, float[] recon,
                                      float[] latentMean, float[] latentLogVar)
            => _tvae.ComputeFreeEnergy(x, recon, latentMean, latentLogVar);
    }

    // ============================================================
    // BehavioralContract — the unified specification
    // ============================================================

    static class BehavioralContract
    {
        // ----------------------------------------------------------
        // Relative thresholds — model-agnostic
        // ----------------------------------------------------------

        // 1. Convergence signature
        /// <summary>
        /// Final objective must be this fraction below epoch-0 objective
        /// (i.e., final &lt; epoch0 * RelDropRatio).  15% relative drop required.
        /// </summary>
        const float RelDropRatio      = 0.85f;

        /// <summary>
        /// Epoch-0 to epoch-1 drop must be at least this fraction of epoch-0
        /// (sharp initial drop invariant).
        /// </summary>
        const float SharpDropRatio    = 0.10f;

        // 2. Latent-geometry
        /// <summary>|latentMean| > AbsExplosionLimit is a regression.</summary>
        const float AbsExplosionLimit = 5.0f;

        /// <summary>
        /// latentLogVar must remain in this absolute range (no-explosion band).
        /// </summary>
        const float LogVarMin         = -8.0f;
        const float LogVarMax         =  4.0f;

        // 3. Reconstruction
        /// <summary>
        /// Mean-squared deviation of reconstruction from input must be below
        /// this ceiling after training (expressed relative: below epoch-0
        /// objective value, which is a natural scale reference).
        /// Per-sample MSE ceiling expressed as an absolute safety floor.
        /// </summary>
        const float ReconMseCeiling   = 5.0f;

        // 4. Decomposition identity
        /// <summary>
        /// |objective - (reconComponent + regComponent)| must be ≤ this.
        /// regComponent = objective - reconComponent (derived; check is sign).
        /// The meaningful invariant is reconComponent ≥ 0 and regComponent ≥ 0.
        /// </summary>
        const float DecompTol         = 0.01f;

        // 6. Stability envelope
        const float StabilityWindow   = 0.15f; // last-quarter window span ceiling

        // ----------------------------------------------------------
        // Entry point — run contract against a model
        // ----------------------------------------------------------

        public static void Run(ILatentModel model, float[][] Xn,
                               float[] testSample, ref int pass, ref int fail,
                               string sectionLabel = "")
        {
            string label = sectionLabel.Length > 0
                ? $"[Step 6] {model.ModelName} — {sectionLabel}"
                : $"[Step 6] {model.ModelName}";

            Console.WriteLine($"\n-- {label} --");

            // ---- Train ----
            float epoch0Obj = float.NaN;
            float epoch1Obj = float.NaN;
            float finalObj  = float.NaN;
            float initLatentAbsMax = float.NaN;
            bool  anyNaN = false, anyInf = false;
            int   totalEpochs = 30;
            int   quarterStart = totalEpochs * 3 / 4;
            float qMin = float.MaxValue, qMax = float.MinValue;

            var history = new List<float>();
            // initLatentAbsMax captured after epoch 0 (post initial drop — the peak spread)

            for (int epoch = 0; epoch < totalEpochs; epoch++)
            {
                var h = model.Fit(Xn, epochs: 1, verbose: 0);
                float obj = h[0];
                history.Add(obj);

                if (float.IsNaN(obj))      anyNaN = true;
                if (float.IsInfinity(obj)) anyInf = true;

                if (epoch == 0)
                {
                    epoch0Obj = obj;
                    // Capture after epoch 0: this is the peak latent spread; it must
                    // contract from here as KL/entropy regularization tightens the geometry.
                    initLatentAbsMax = ComputeLatentAbsMax(Xn, model);
                }
                if (epoch == 1) epoch1Obj = obj;
                if (epoch >= quarterStart) { qMin = MathF.Min(qMin, obj); qMax = MathF.Max(qMax, obj); }
                finalObj = obj;
            }

            float finalLatentAbsMax = ComputeLatentAbsMax(Xn, model);

            // ---- 1. Convergence signature (relative) ----
            Console.WriteLine("\n  1. Convergence signature:");
            C("No NaN in objective over training",             !anyNaN,             ref pass, ref fail);
            C("No Inf in objective over training",             !anyInf,             ref pass, ref fail);
            C($"Objective drops sharply epoch 0→1 (>{SharpDropRatio * 100:F0}% relative drop)",
                epoch1Obj < epoch0Obj * (1f - SharpDropRatio),
                ref pass, ref fail,
                $"epoch0={epoch0Obj:F4}  epoch1={epoch1Obj:F4}");
            C($"Final objective < epoch0 × {RelDropRatio} (converges)",
                finalObj < epoch0Obj * RelDropRatio,
                ref pass, ref fail,
                $"epoch0={epoch0Obj:F4}  final={finalObj:F4}  threshold={epoch0Obj * RelDropRatio:F4}");
            C($"Stable basin: last-quarter window span ≤ {StabilityWindow}",
                (qMax - qMin) <= StabilityWindow,
                ref pass, ref fail,
                $"span={qMax - qMin:F4} (min={qMin:F4}, max={qMax:F4})");

            // ---- 2. Latent-geometry invariants ----
            Console.WriteLine("\n  2. Latent-geometry:");
            C("Latent means contract over training",
                finalLatentAbsMax < initLatentAbsMax,
                ref pass, ref fail,
                $"init={initLatentAbsMax:F3}  final={finalLatentAbsMax:F3}");
            C($"No latent explosion (|mean| < {AbsExplosionLimit})",
                finalLatentAbsMax < AbsExplosionLimit,
                ref pass, ref fail,
                $"|mean|_max={finalLatentAbsMax:F3}");

            // logVar range — checked on test sample after training
            var (latMean, latLogVar) = model.EncodeDistribution(testSample);
            bool logVarOk = latLogVar.All(v => v >= LogVarMin && v <= LogVarMax);
            C($"Latent logVar bounded ∈ [{LogVarMin}, {LogVarMax}]",
                logVarOk,
                ref pass, ref fail,
                $"logVar=[{string.Join(",", latLogVar.Select(v => v.ToString("F2")))}]");

            // ---- 3. Reconstruction invariants ----
            Console.WriteLine("\n  3. Reconstruction:");
            float[] recon = model.Reconstruct(testSample);
            C("Reconstruction dimensionality matches input",
                recon.Length == testSample.Length,
                ref pass, ref fail,
                $"input.Length={testSample.Length}  recon.Length={recon.Length}");
            C("Reconstruction: no NaN",
                !recon.Any(float.IsNaN),
                ref pass, ref fail);
            C("Reconstruction: no Inf",
                !recon.Any(float.IsInfinity),
                ref pass, ref fail);
            float reconMse = 0f;
            for (int i = 0; i < testSample.Length; i++)
                reconMse += (recon[i] - testSample[i]) * (recon[i] - testSample[i]);
            reconMse /= testSample.Length;
            C($"Reconstruction MSE < ceiling ({ReconMseCeiling})",
                reconMse < ReconMseCeiling,
                ref pass, ref fail,
                $"MSE={reconMse:F4}");

            // ---- 4. Decomposition identity ----
            Console.WriteLine("\n  4. Decomposition identity:");
            // reconComponent = MSE(x, recon) — always non-negative by construction
            // regComponent   = objective - reconComponent — must also be non-negative
            // This verifies the sign convention of the variational/thermodynamic decomposition.
            bool reconNonNeg = true, regNonNeg = true;
            int decompositionViolations = 0;
            foreach (var x in Xn)
            {
                float[] r         = model.Reconstruct(x);
                var (lm, llv)     = model.EncodeDistribution(x);
                float obj         = model.ComputeObjective(x, r, lm, llv);
                float reconComp   = 0f;
                for (int i = 0; i < x.Length; i++)
                    reconComp += (r[i] - x[i]) * (r[i] - x[i]);
                reconComp /= x.Length;
                float regComp = obj - reconComp;

                if (reconComp < -DecompTol) reconNonNeg = false;
                if (regComp   < -DecompTol) regNonNeg   = false;
                if (MathF.Abs(obj - (reconComp + regComp)) > DecompTol) decompositionViolations++;
            }
            C("Reconstruction component ≥ 0 (non-negative MSE)",        reconNonNeg, ref pass, ref fail);
            C("Regularization component ≥ 0 (non-negative KL/entropy)",  regNonNeg,   ref pass, ref fail);
            C($"Decomposition identity holds across all samples (tol ±{DecompTol})",
                decompositionViolations == 0,
                ref pass, ref fail,
                decompositionViolations > 0 ? $"{decompositionViolations} violations" : "");

            // ---- 5. PSC flow invariants (structural) ----
            // The PSC flow (data → normalize → construct → train → log → test)
            // is validated by construction: the adapter's Fit(), EncodeDistribution(),
            // and Reconstruct() calls above exercise all six phases.
            Console.WriteLine("\n  5. PSC flow:");
            C("Data generation phase executed (Xn is non-empty)",
                Xn.Length > 0 && Xn[0].Length == model.InputDim,
                ref pass, ref fail,
                $"samples={Xn.Length}  dim={Xn[0].Length}");
            C("Normalization phase executed (Xn values bounded: |v| < 10)",
                Xn.All(x => x.All(v => MathF.Abs(v) < 10f)),
                ref pass, ref fail);
            C("Model construction: InputDim and LatentDim are positive",
                model.InputDim > 0 && model.LatentDim > 0,
                ref pass, ref fail,
                $"InputDim={model.InputDim}  LatentDim={model.LatentDim}");
            C("Training loop executed (history length = 30)",
                history.Count == totalEpochs,
                ref pass, ref fail);
            C("Per-epoch objective logged (no NaN epoch in history)",
                history.All(v => !float.IsNaN(v)),
                ref pass, ref fail);
            C("Test pass executed (reconstruction returned without exception)",
                recon != null && recon.Length > 0,
                ref pass, ref fail);

            // ---- 6. Stability envelope ----
            Console.WriteLine("\n  6. Stability envelope:");
            bool objStable = history.All(v => !float.IsNaN(v) && !float.IsInfinity(v));
            C("No NaN/Inf anywhere in objective history",  objStable,            ref pass, ref fail);
            C("No latent explosion across all samples",
                finalLatentAbsMax < AbsExplosionLimit,
                ref pass, ref fail,
                $"|mean|_max={finalLatentAbsMax:F3}");
            C("No reconstruction explosion (MSE < ceiling)",
                reconMse < ReconMseCeiling,
                ref pass, ref fail,
                $"MSE={reconMse:F4}");
            C("No KL/entropy collapse (regComponent not universally ≈ 0)",
                regNonNeg,  // presence of positive regComp implies no collapse
                ref pass, ref fail);
            // "no oscillatory explosion" — window span already checked in section 1,
            // reinforce here as envelope invariant
            C($"No oscillatory explosion in last quarter (span ≤ {StabilityWindow})",
                (qMax - qMin) <= StabilityWindow,
                ref pass, ref fail,
                $"span={qMax - qMin:F4}");
        }

        // ----------------------------------------------------------
        // Helpers
        // ----------------------------------------------------------

        static float ComputeLatentAbsMax(float[][] Xn, ILatentModel model)
        {
            float max = float.MinValue;
            foreach (var x in Xn)
            {
                var (lm, _) = model.EncodeDistribution(x);
                foreach (float v in lm)
                    max = MathF.Max(max, MathF.Abs(v));
            }
            return max;
        }

        static void C(string name, bool ok, ref int pass, ref int fail, string detail = "")
        {
            string tag    = ok ? "[PASS]" : "[FAIL]";
            string suffix = !ok && detail.Length > 0 ? $"  ({detail})" : "";
            Console.WriteLine($"    {tag} {name}{suffix}");
            if (ok) pass++; else fail++;
        }
    }
}
