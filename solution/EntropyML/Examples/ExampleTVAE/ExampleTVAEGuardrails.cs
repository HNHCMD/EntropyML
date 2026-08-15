using EntropyML;

namespace ExampleTVAE
{
    // ============================================================
    // ExampleTVAEGuardrails  (Step 4 + Step 5 STS)
    // ============================================================
    //
    // Numerical stability baselines and behavioral invariants for
    // ExampleTVAE, established from the validated Step 3 run.
    //
    // Step 4 checks (baseline):
    //   Epoch 0 FreeEnergy ~ 0.62  →  acceptable band [0.40, 0.90]
    //   Epoch 1 FreeEnergy ~ 0.37  →  acceptable band [0.25, 0.55]
    //   Final convergence  ~ 0.32  →  acceptable band [0.25, 0.45]
    //   Avg energy term    ~ 0.09–0.16  → acceptable band [0.05, 0.25]
    //   Avg entropy term   ~ 0.19–0.25  → acceptable band [0.10, 0.35]
    //   eq-range tightens: initMax → finalMax (must contract)
    //   entropyPotential (trained) ~ [-2.1, -2.2]  → band [-3.0, -1.0]
    //   |equilibrium| explosion threshold: 5.0
    //   Reconstruction tolerance from input: ±0.5
    //
    // Step 5 checks (extended):
    //   Multi-seed stability:      seeds 42, 123, 999
    //   Input-distribution:        measured, uniform-random, Gaussian, skewed, scaled
    //   Latent-dimension:          latentDim ∈ {1, 2, 3, 4}
    //   Cross-model consistency:   VAE ELBO vs TVAE free-energy qualitative match
    //   Long-horizon stability:    300 epochs, no slow drift or entropy collapse
    //
    // Design:
    //   Each section is independent. No shared model state.
    //   Throws GuardrailException after all sections if any check failed.
    // ============================================================

    static class Guardrails
    {
        // ------------------------------------------------------------
        // Shared thresholds
        // ------------------------------------------------------------
        const float Epoch0_FE_Min    = 0.40f;
        const float Epoch0_FE_Max    = 0.90f;
        const float Epoch1_FE_Min    = 0.25f;
        const float Epoch1_FE_Max    = 0.55f;
        const float Final_FE_Min     = 0.25f;
        const float Final_FE_Max     = 0.45f;
        const float Energy_Min       = 0.05f;
        const float Energy_Max       = 0.25f;
        const float Entropy_Min      = 0.10f;
        const float Entropy_Max      = 0.35f;
        const float FE_Decomp_Tol    = 0.01f;
        const float Eq_Explosion     = 5.0f;
        const float EntPot_Min       = -3.0f;
        const float EntPot_Max       = -1.0f;
        const float Recon_Tolerance  = 0.5f;

        // Step 5 — extended thresholds
        // Multi-seed: same convergence band as baseline
        // Multi-distribution: generous bounds (unnormalized inputs may have wider ranges)
        const float Robust_FE_Min    = 0.01f;
        const float Robust_FE_Max    = 5.00f;   // allow wide range; block NaN/Inf/explosion
        const float Robust_Eq_Expl   = 10.0f;
        // Latent-dim: convergence band relaxed for dim=1 (harder) and dim=4 (wider)
        const float LatDim_FE_Min    = 0.10f;
        const float LatDim_FE_Max    = 0.80f;
        // Latent-dim entropy potential: no-explosion band (wider than baseline dim=2 band)
        // Baseline [-3,-1] is valid only for dim=2; other dims may settle anywhere in [-5, 1]
        const float LatDim_EntPot_Min = -5.0f;
        const float LatDim_EntPot_Max =  1.0f;
        // Cross-model: both VAE and TVAE must converge from epoch0 to final
        const float CrossModel_Drop  = 0.05f;   // final must be at least this much below epoch0
        // Long-horizon: no drift upward over last 50 epochs, no entropy collapse
        const float LongHorizon_Eps  = 0.05f;   // allowed upward drift window
        const float LongHorizon_EntMin = 0.02f; // entropy must not collapse to near-zero

        static int _pass, _fail;

        // ============================================================
        // Entry point
        // ============================================================
        public static void Run()
        {
            _pass = 0;
            _fail = 0;

            Console.WriteLine("\n====================================================");
            Console.WriteLine("  ExampleTVAE Guardrails (Step 4 + Step 5 + Step 6 STS)");
            Console.WriteLine("====================================================");

            // Step 4 — baseline
            RunBaselineTrainingGuardrails();
            RunBaselineTestGuardrails();

            // Step 5 — extended
            RunMultiSeedGuardrails();
            RunDistributionRobustnessGuardrails();
            RunLatentDimensionGuardrails();
            RunCrossModelConsistencyGuardrails();
            RunLongHorizonGuardrails();

            // Step 6 — unified behavioral contract (model-agnostic)
            RunBehavioralContractGuardrails();

            PrintSummary();
        }

        // ============================================================
        // STEP 4 — BASELINE
        // ============================================================

        static void RunBaselineTrainingGuardrails()
        {
            Console.WriteLine("\n-- [Step 4] Baseline training invariants (seed=42) --");

            var X  = MakeMeasuredData();
            var Xn = DataGen.Normalize(X);
            var tvae = new TVAE(inputDim: 3, latentDim: 2, randSeed: 42);

            float epoch0FE     = float.NaN;
            float epoch1FE     = float.NaN;
            float finalFE      = float.NaN;
            float initEqAbsMax = float.NaN;
            bool  anyNaN       = false;
            bool  anyInf       = false;
            bool  decompositionOk = true;
            float epochEnergySum  = 0f;
            float epochEntropySum = 0f;
            int   epochSamples    = 0;

            for (int epoch = 0; epoch < 30; epoch++)
            {
                var history = tvae.Fit(Xn, epochs: 1, verbose: 0);
                float avgFE = history[0];

                if (float.IsNaN(avgFE))      anyNaN = true;
                if (float.IsInfinity(avgFE)) anyInf = true;

                if (epoch == 0) epoch0FE = avgFE;
                if (epoch == 1) epoch1FE = avgFE;
                finalFE = avgFE;

                float eqAbsMax = ComputeEqAbsMax(Xn, tvae);
                if (epoch == 0) initEqAbsMax = eqAbsMax;

                Check($"Epoch {epoch,2}: no latent explosion (|eq| < {Eq_Explosion})",
                    eqAbsMax < Eq_Explosion, $"|eq|_max={eqAbsMax:F3}");

                if (epoch == 29)
                {
                    foreach (var x in Xn)
                    {
                        var (eq, entPot) = tvae.EncodeThermoState(x);
                        float[] relaxed = tvae.RelaxAndReconstruct(x);
                        float et  = ComputeEnergyTerm(x, relaxed);
                        float fe  = tvae.ComputeFreeEnergy(x, relaxed, eq, entPot);
                        float st  = fe - et;

                        if (float.IsNaN(fe) || float.IsNaN(et) || float.IsNaN(st)) anyNaN = true;
                        if (float.IsInfinity(fe) || float.IsInfinity(et))          anyInf = true;

                        if (MathF.Abs(fe - (et + st)) > FE_Decomp_Tol) decompositionOk = false;

                        epochEnergySum  += et;
                        epochEntropySum += st;
                        epochSamples++;
                    }
                }
            }

            float finalEqAbsMax = ComputeEqAbsMax(Xn, tvae);
            float avgEnergy  = epochSamples > 0 ? epochEnergySum  / epochSamples : float.NaN;
            float avgEntropy = epochSamples > 0 ? epochEntropySum / epochSamples : float.NaN;

            Console.WriteLine("\n  Free-energy convergence:");
            Check($"Epoch 0 FreeEnergy ∈ [{Epoch0_FE_Min}, {Epoch0_FE_Max}]",
                epoch0FE >= Epoch0_FE_Min && epoch0FE <= Epoch0_FE_Max, $"actual={epoch0FE:F4}");
            Check($"Epoch 1 FreeEnergy ∈ [{Epoch1_FE_Min}, {Epoch1_FE_Max}]",
                epoch1FE >= Epoch1_FE_Min && epoch1FE <= Epoch1_FE_Max, $"actual={epoch1FE:F4}");
            Check($"Final FreeEnergy ∈ [{Final_FE_Min}, {Final_FE_Max}]",
                finalFE >= Final_FE_Min && finalFE <= Final_FE_Max, $"actual={finalFE:F4}");

            Console.WriteLine("\n  Numerical safety:");
            Check("No NaN in free-energy or decomposition", !anyNaN);
            Check("No Inf in free-energy or decomposition", !anyInf);

            Console.WriteLine("\n  Energy/entropy decomposition (epoch 29 averages):");
            Check($"Avg energy ∈ [{Energy_Min}, {Energy_Max}]",
                avgEnergy >= Energy_Min && avgEnergy <= Energy_Max, $"avg={avgEnergy:F4}");
            Check($"Avg entropy ∈ [{Entropy_Min}, {Entropy_Max}]",
                avgEntropy >= Entropy_Min && avgEntropy <= Entropy_Max, $"avg={avgEntropy:F4}");
            Check($"FreeEnergy ≈ energy + entropy (±{FE_Decomp_Tol})", decompositionOk);

            Console.WriteLine("\n  Latent geometry:");
            Check("Equilibrium range contracts over training",
                finalEqAbsMax < initEqAbsMax,
                $"init={initEqAbsMax:F3}  final={finalEqAbsMax:F3}");
        }

        static void RunBaselineTestGuardrails()
        {
            Console.WriteLine("\n-- [Step 4] Baseline test-pass invariants (seed=42, trained) --");

            var X  = MakeMeasuredData();
            var Xn = DataGen.Normalize(X);
            var tvae = new TVAE(inputDim: 3, latentDim: 2, randSeed: 42);
            tvae.Fit(Xn, epochs: 30, verbose: 0);

            float[] test  = { 2.0f, 0.5f, -1.5f };
            float[] testN = DataGen.NormalizeSingle(test, X);

            var (eq, entPot) = tvae.EncodeThermoState(testN);
            float[] relaxed  = tvae.RelaxAndReconstruct(testN);
            float fe         = tvae.ComputeFreeEnergy(testN, relaxed, eq, entPot);
            float et         = ComputeEnergyTerm(testN, relaxed);
            float st         = fe - et;

            Console.WriteLine("\n  Reconstruction safety:");
            Check("Reconstruction: no NaN", !relaxed.Any(float.IsNaN));
            Check("Reconstruction: no Inf", !relaxed.Any(float.IsInfinity));

            bool reconClose = true;
            for (int i = 0; i < testN.Length; i++)
                if (MathF.Abs(relaxed[i] - testN[i]) > Recon_Tolerance) reconClose = false;
            Check($"Trained reconstruction within ±{Recon_Tolerance} of norm input",
                reconClose,
                $"input=[{testN[0]:F3},{testN[1]:F3},{testN[2]:F3}]  " +
                $"recon=[{relaxed[0]:F3},{relaxed[1]:F3},{relaxed[2]:F3}]");

            Console.WriteLine("\n  Latent geometry (test point):");
            Check($"Equilibrium within ±{Eq_Explosion} (no explosion)",
                MathF.Abs(eq[0]) < Eq_Explosion && MathF.Abs(eq[1]) < Eq_Explosion,
                $"eq=[{eq[0]:F3},{eq[1]:F3}]");
            Check($"Entropy potential (logσ²) ∈ [{EntPot_Min}, {EntPot_Max}]",
                entPot[0] >= EntPot_Min && entPot[0] <= EntPot_Max &&
                entPot[1] >= EntPot_Min && entPot[1] <= EntPot_Max,
                $"entPot=[{entPot[0]:F3},{entPot[1]:F3}]");

            Console.WriteLine("\n  Free-energy decomposition (test point):");
            Check("Energy term ≥ 0", et >= 0f, $"et={et:F4}");
            Check("Entropy term ≥ 0", st >= 0f, $"st={st:F4}");
            Check($"FreeEnergy ≈ energy + entropy (±{FE_Decomp_Tol})",
                MathF.Abs(fe - (et + st)) <= FE_Decomp_Tol,
                $"fe={fe:F4}  et={et:F4}  st={st:F4}");
        }

        // ============================================================
        // STEP 5.1 — MULTI-SEED STABILITY
        // ============================================================

        static void RunMultiSeedGuardrails()
        {
            Console.WriteLine("\n-- [Step 5.1] Multi-seed stability (seeds: 42, 123, 999) --");

            var X  = MakeMeasuredData();
            var Xn = DataGen.Normalize(X);

            foreach (int seed in new[] { 42, 123, 999 })
            {
                var tvae = new TVAE(inputDim: 3, latentDim: 2, randSeed: seed);

                float epoch0FE     = float.NaN;
                float finalFE      = float.NaN;
                float initEqAbsMax = float.NaN;
                bool  anyNaN       = false;
                bool  anyInf       = false;

                for (int epoch = 0; epoch < 30; epoch++)
                {
                    var history = tvae.Fit(Xn, epochs: 1, verbose: 0);
                    float avgFE = history[0];

                    if (float.IsNaN(avgFE))      anyNaN = true;
                    if (float.IsInfinity(avgFE)) anyInf = true;

                    if (epoch == 0) { epoch0FE = avgFE; initEqAbsMax = ComputeEqAbsMax(Xn, tvae); }
                    finalFE = avgFE;
                }

                float finalEqAbsMax = ComputeEqAbsMax(Xn, tvae);

                Console.WriteLine($"\n  Seed {seed}:");
                Check($"  No NaN", !anyNaN);
                Check($"  No Inf", !anyInf);
                Check($"  Final FreeEnergy ∈ [{Final_FE_Min}, {Final_FE_Max}]",
                    finalFE >= Final_FE_Min && finalFE <= Final_FE_Max, $"actual={finalFE:F4}");
                Check($"  Equilibrium range contracts",
                    finalEqAbsMax < initEqAbsMax,
                    $"init={initEqAbsMax:F3}  final={finalEqAbsMax:F3}");
                Check($"  No latent explosion (|eq| < {Eq_Explosion})",
                    finalEqAbsMax < Eq_Explosion, $"|eq|_max={finalEqAbsMax:F3}");
            }
        }

        // ============================================================
        // STEP 5.2 — INPUT-DISTRIBUTION ROBUSTNESS
        // ============================================================

        static void RunDistributionRobustnessGuardrails()
        {
            Console.WriteLine("\n-- [Step 5.2] Input-distribution robustness --");

            var distributions = new (string name, float[][] data)[]
            {
                ("Measured (baseline)",  DataGen.Normalize(MakeMeasuredData())),
                ("Uniform random",       MakeUniformRandom(430, 3, seed: 7)),
                ("Gaussian",             MakeGaussian(430, 3, seed: 7)),
                ("Slightly skewed",      MakeSkewed(430, 3, seed: 7)),
                ("Slightly scaled",      MakeScaled(DataGen.Normalize(MakeMeasuredData()), scale: 2.0f)),
            };

            foreach (var (name, Xn) in distributions)
            {
                var tvae = new TVAE(inputDim: 3, latentDim: 2, randSeed: 42);
                bool anyNaN = false, anyInf = false;
                float finalFE = float.NaN;
                float finalEqAbsMax = float.NaN;

                for (int epoch = 0; epoch < 30; epoch++)
                {
                    var history = tvae.Fit(Xn, epochs: 1, verbose: 0);
                    float avgFE = history[0];
                    if (float.IsNaN(avgFE))      anyNaN = true;
                    if (float.IsInfinity(avgFE)) anyInf = true;
                    finalFE = avgFE;
                }

                finalEqAbsMax = ComputeEqAbsMax(Xn, tvae);

                // Verify reconstruction on first sample
                bool reconNaN = false, reconInf = false;
                float[] testX = Xn[0];
                float[] recon = tvae.RelaxAndReconstruct(testX);
                if (recon.Any(float.IsNaN))      reconNaN = true;
                if (recon.Any(float.IsInfinity)) reconInf = true;

                Console.WriteLine($"\n  Distribution: {name}");
                Check($"  No NaN in free-energy", !anyNaN);
                Check($"  No Inf in free-energy", !anyInf);
                Check($"  Final FreeEnergy ∈ [{Robust_FE_Min}, {Robust_FE_Max}]",
                    finalFE >= Robust_FE_Min && finalFE <= Robust_FE_Max, $"actual={finalFE:F4}");
                Check($"  No latent explosion (|eq| < {Robust_Eq_Expl})",
                    finalEqAbsMax < Robust_Eq_Expl, $"|eq|_max={finalEqAbsMax:F3}");
                Check($"  Reconstruction: no NaN", !reconNaN);
                Check($"  Reconstruction: no Inf", !reconInf);
            }
        }

        // ============================================================
        // STEP 5.3 — LATENT-DIMENSION CONSISTENCY
        // ============================================================

        static void RunLatentDimensionGuardrails()
        {
            Console.WriteLine("\n-- [Step 5.3] Latent-dimension consistency (dims: 1, 2, 3, 4) --");

            var X  = MakeMeasuredData();
            var Xn = DataGen.Normalize(X);

            foreach (int latDim in new[] { 1, 2, 3, 4 })
            {
                var tvae = new TVAE(inputDim: 3, latentDim: latDim, randSeed: 42);
                bool anyNaN = false, anyInf = false;
                float finalFE = float.NaN;
                float finalEqAbsMax = float.NaN;
                bool  decompositionOk = true;

                for (int epoch = 0; epoch < 30; epoch++)
                {
                    var history = tvae.Fit(Xn, epochs: 1, verbose: 0);
                    float avgFE = history[0];
                    if (float.IsNaN(avgFE))      anyNaN = true;
                    if (float.IsInfinity(avgFE)) anyInf = true;
                    finalFE = avgFE;
                }

                // Post-training decomposition and geometry check
                foreach (var x in Xn)
                {
                    var (eq, entPot) = tvae.EncodeThermoState(x);
                    float[] relaxed  = tvae.RelaxAndReconstruct(x);
                    float et  = ComputeEnergyTerm(x, relaxed);
                    float fe  = tvae.ComputeFreeEnergy(x, relaxed, eq, entPot);
                    float st  = fe - et;

                    if (float.IsNaN(fe) || float.IsNaN(et)) anyNaN = true;
                    if (MathF.Abs(fe - (et + st)) > FE_Decomp_Tol) decompositionOk = false;

                    // Compute per-dim eq abs max for this sample
                    for (int d = 0; d < latDim; d++)
                    {
                        float cur = float.IsNaN(finalEqAbsMax) ? 0f : finalEqAbsMax;
                        finalEqAbsMax = MathF.Max(cur, MathF.Abs(eq[d]));
                    }
                }

                // entropyPotential check on test point P1
                float[] testN = DataGen.NormalizeSingle(new[] { 2.0f, 0.5f, -1.5f }, X);
                var (eqTest, entPotTest) = tvae.EncodeThermoState(testN);
                bool entPotOk = entPotTest.All(v => v >= LatDim_EntPot_Min && v <= LatDim_EntPot_Max);

                Console.WriteLine($"\n  LatentDim={latDim}:");
                Check($"  No NaN", !anyNaN);
                Check($"  No Inf", !anyInf);
                Check($"  Final FreeEnergy ∈ [{LatDim_FE_Min}, {LatDim_FE_Max}]",
                    finalFE >= LatDim_FE_Min && finalFE <= LatDim_FE_Max, $"actual={finalFE:F4}");
                Check($"  No latent explosion (|eq| < {Eq_Explosion})",
                    finalEqAbsMax < Eq_Explosion, $"|eq|_max={finalEqAbsMax:F3}");
                Check($"  FreeEnergy ≈ energy + entropy (±{FE_Decomp_Tol})", decompositionOk);
                Check($"  EntropyPotential ∈ [{LatDim_EntPot_Min}, {LatDim_EntPot_Max}] (no explosion)", entPotOk,
                    $"entPot=[{string.Join(",", entPotTest.Select(v => v.ToString("F2")))}]");
            }
        }

        // ============================================================
        // STEP 5.4 — CROSS-MODEL CONSISTENCY (VAE ↔ TVAE)
        // ============================================================

        static void RunCrossModelConsistencyGuardrails()
        {
            Console.WriteLine("\n-- [Step 5.4] Cross-model consistency (VAE ↔ TVAE) --");

            var X  = MakeMeasuredData();
            var Xn = DataGen.Normalize(X);

            // --- VAE side ---
            var vae = new VAE(inputDim: 3, latentDim: 2, randSeed: 42);
            float vaeEpoch0FE = float.NaN, vaeFinalFE = float.NaN;
            float vaeInitMuMax = float.NaN, vaeFinalMuMax = float.NaN;

            for (int epoch = 0; epoch < 30; epoch++)
            {
                var history = vae.Fit(Xn, epochs: 1, verbose: 0);
                float avgLoss = history[0];
                if (epoch == 0) { vaeEpoch0FE = avgLoss; vaeInitMuMax = ComputeVaeMuAbsMax(Xn, vae); }
                vaeFinalFE = avgLoss;
            }
            vaeFinalMuMax = ComputeVaeMuAbsMax(Xn, vae);

            // Reconstruction coherence: VAE
            float[] testN = DataGen.NormalizeSingle(new[] { 2.0f, 0.5f, -1.5f }, X);
            float[] vaeRecon = vae.Reconstruct(testN);
            bool vaeReconClose = true;
            for (int i = 0; i < testN.Length; i++)
                if (MathF.Abs(vaeRecon[i] - testN[i]) > Recon_Tolerance) vaeReconClose = false;

            // --- TVAE side ---
            var tvae = new TVAE(inputDim: 3, latentDim: 2, randSeed: 42);
            float tvaeEpoch0FE = float.NaN, tvaeFinalFE = float.NaN;
            float tvaeInitEqMax = float.NaN, tvaeFinalEqMax = float.NaN;

            for (int epoch = 0; epoch < 30; epoch++)
            {
                var history = tvae.Fit(Xn, epochs: 1, verbose: 0);
                float avgFE = history[0];
                if (epoch == 0) { tvaeEpoch0FE = avgFE; tvaeInitEqMax = ComputeEqAbsMax(Xn, tvae); }
                tvaeFinalFE = avgFE;
            }
            tvaeFinalEqMax = ComputeEqAbsMax(Xn, tvae);

            float[] tvaeRecon = tvae.RelaxAndReconstruct(testN);
            bool tvaeReconClose = true;
            for (int i = 0; i < testN.Length; i++)
                if (MathF.Abs(tvaeRecon[i] - testN[i]) > Recon_Tolerance) tvaeReconClose = false;

            Console.WriteLine($"\n  VAE:  epoch0={vaeEpoch0FE:F4}  final={vaeFinalFE:F4}  " +
                              $"μ-range [{vaeInitMuMax:F2}→{vaeFinalMuMax:F2}]");
            Console.WriteLine($"  TVAE: epoch0={tvaeEpoch0FE:F4}  final={tvaeFinalFE:F4}  " +
                              $"eq-range [{tvaeInitEqMax:F2}→{tvaeFinalEqMax:F2}]");

            // Both must converge (final < epoch0 - crossModelDrop)
            Check("VAE converges (final < epoch0 - threshold)",
                vaeFinalFE < vaeEpoch0FE - CrossModel_Drop,
                $"epoch0={vaeEpoch0FE:F4}  final={vaeFinalFE:F4}");
            Check("TVAE converges (final < epoch0 - threshold)",
                tvaeFinalFE < tvaeEpoch0FE - CrossModel_Drop,
                $"epoch0={tvaeEpoch0FE:F4}  final={tvaeFinalFE:F4}");

            // Both must converge to same qualitative basin
            Check($"VAE final ELBO ∈ [{Final_FE_Min}, {Final_FE_Max}]",
                vaeFinalFE >= Final_FE_Min && vaeFinalFE <= Final_FE_Max, $"actual={vaeFinalFE:F4}");
            Check($"TVAE final FreeEnergy ∈ [{Final_FE_Min}, {Final_FE_Max}]",
                tvaeFinalFE >= Final_FE_Min && tvaeFinalFE <= Final_FE_Max, $"actual={tvaeFinalFE:F4}");

            // Latent contraction in both
            Check("VAE latent μ range contracts",
                vaeFinalMuMax < vaeInitMuMax, $"init={vaeInitMuMax:F3}  final={vaeFinalMuMax:F3}");
            Check("TVAE equilibrium range contracts",
                tvaeFinalEqMax < tvaeInitEqMax, $"init={tvaeInitEqMax:F3}  final={tvaeFinalEqMax:F3}");

            // Reconstruction coherence in both
            Check("VAE reconstruction coherent (within ±0.5 of input)", vaeReconClose,
                $"recon=[{vaeRecon[0]:F3},{vaeRecon[1]:F3},{vaeRecon[2]:F3}]");
            Check("TVAE reconstruction coherent (within ±0.5 of input)", tvaeReconClose,
                $"recon=[{tvaeRecon[0]:F3},{tvaeRecon[1]:F3},{tvaeRecon[2]:F3}]");
        }

        // ============================================================
        // STEP 5.5 — LONG-HORIZON STABILITY (300 epochs)
        // ============================================================

        static void RunLongHorizonGuardrails()
        {
            Console.WriteLine("\n-- [Step 5.5] Long-horizon stability (300 epochs, seed=42) --");

            const int TotalEpochs = 300;
            const int WindowStart = 250;   // last 50 epochs for drift check

            var X  = MakeMeasuredData();
            var Xn = DataGen.Normalize(X);
            var tvae = new TVAE(inputDim: 3, latentDim: 2, randSeed: 42);

            bool anyNaN = false, anyInf = false;
            float windowMinFE = float.MaxValue;
            float windowMaxFE = float.MinValue;
            float epoch0FE    = float.NaN;
            float finalFE     = float.NaN;
            float finalEqAbsMax = float.NaN;
            float finalAvgEntropy = float.NaN;

            var history = tvae.Fit(Xn, epochs: TotalEpochs, verbose: 0);

            for (int epoch = 0; epoch < TotalEpochs; epoch++)
            {
                float fe = history[epoch];
                if (float.IsNaN(fe))      anyNaN = true;
                if (float.IsInfinity(fe)) anyInf = true;
                if (epoch == 0) epoch0FE = fe;
                if (epoch >= WindowStart)
                {
                    windowMinFE = MathF.Min(windowMinFE, fe);
                    windowMaxFE = MathF.Max(windowMaxFE, fe);
                }
            }
            finalFE = history[TotalEpochs - 1];

            // Evaluate geometry and entropy at final state
            finalEqAbsMax = ComputeEqAbsMax(Xn, tvae);

            float entropySum = 0f;
            int   entropyCount = 0;
            foreach (var x in Xn)
            {
                var (eq, entPot) = tvae.EncodeThermoState(x);
                float[] relaxed  = tvae.RelaxAndReconstruct(x);
                float et = ComputeEnergyTerm(x, relaxed);
                float fe = tvae.ComputeFreeEnergy(x, relaxed, eq, entPot);
                entropySum += fe - et;
                entropyCount++;
            }
            finalAvgEntropy = entropyCount > 0 ? entropySum / entropyCount : float.NaN;

            // Reconstruction coherence at final state
            float[] testN = DataGen.NormalizeSingle(new[] { 2.0f, 0.5f, -1.5f }, X);
            float[] recon = tvae.RelaxAndReconstruct(testN);
            bool reconClose = true;
            for (int i = 0; i < testN.Length; i++)
                if (MathF.Abs(recon[i] - testN[i]) > Recon_Tolerance) reconClose = false;

            float drift = windowMaxFE - windowMinFE;

            Check("No NaN over 300 epochs", !anyNaN);
            Check("No Inf over 300 epochs", !anyInf);
            Check($"Final FreeEnergy ∈ [{Final_FE_Min}, {Final_FE_Max}]",
                finalFE >= Final_FE_Min && finalFE <= Final_FE_Max, $"actual={finalFE:F4}");
            Check($"No upward drift in last 50 epochs (window span ≤ {LongHorizon_Eps + 0.10f:F2})",
                drift <= LongHorizon_Eps + 0.10f,
                $"window span={drift:F4} (min={windowMinFE:F4}, max={windowMaxFE:F4})");
            Check($"No latent explosion after 300 epochs (|eq| < {Eq_Explosion})",
                finalEqAbsMax < Eq_Explosion, $"|eq|_max={finalEqAbsMax:F3}");
            Check($"Entropy does not collapse (avg entropy > {LongHorizon_EntMin})",
                finalAvgEntropy > LongHorizon_EntMin, $"avgEntropy={finalAvgEntropy:F4}");
            Check("Reconstruction coherent after 300 epochs (within ±0.5 of input)",
                reconClose,
                $"recon=[{recon[0]:F3},{recon[1]:F3},{recon[2]:F3}]");
        }

        // ============================================================
        // STEP 6 — UNIFIED BEHAVIORAL CONTRACT
        // ============================================================

        static void RunBehavioralContractGuardrails()
        {
            Console.WriteLine("\n====================================================");
            Console.WriteLine("  [Step 6] Unified Behavioral Contract");
            Console.WriteLine("  Applies to: VAE, TVAE, any future EntropyML model");
            Console.WriteLine("====================================================");

            var X  = MakeMeasuredData();
            var Xn = DataGen.Normalize(X);
            float[] testN = DataGen.NormalizeSingle(new[] { 2.0f, 0.5f, -1.5f }, X);

            // Run contract against VAE (seed=42)
            var vae = new VaeAdapter(inputDim: 3, latentDim: 2, randSeed: 42);
            BehavioralContract.Run(vae, Xn, testN, ref _pass, ref _fail, "standard config");

            // Run contract against TVAE (seed=42)
            var tvae = new TvaeAdapter(inputDim: 3, latentDim: 2, randSeed: 42);
            BehavioralContract.Run(tvae, Xn, testN, ref _pass, ref _fail, "standard config");

            // Run contract against both with a different seed — tests seed-independence
            var vae2  = new VaeAdapter( inputDim: 3, latentDim: 2, randSeed: 999);
            var tvae2 = new TvaeAdapter(inputDim: 3, latentDim: 2, randSeed: 999);
            BehavioralContract.Run(vae2,  Xn, testN, ref _pass, ref _fail, "seed=999");
            BehavioralContract.Run(tvae2, Xn, testN, ref _pass, ref _fail, "seed=999");
        }

        // ============================================================
        // Helpers
        // ============================================================

        static float ComputeEnergyTerm(float[] x, float[] relaxed)
        {
            float mse = 0f;
            for (int i = 0; i < x.Length; i++)
                mse += (x[i] - relaxed[i]) * (x[i] - relaxed[i]);
            return mse / x.Length;
        }

        static float ComputeEqAbsMax(float[][] Xn, TVAE tvae)
        {
            float max = float.MinValue;
            foreach (var x in Xn)
            {
                var (eq, _) = tvae.EncodeThermoState(x);
                foreach (float v in eq)
                    max = MathF.Max(max, MathF.Abs(v));
            }
            return max;
        }

        static float ComputeVaeMuAbsMax(float[][] Xn, VAE vae)
        {
            float max = float.MinValue;
            foreach (var x in Xn)
            {
                var (mu, _) = vae.Encode(x);
                foreach (float v in mu)
                    max = MathF.Max(max, MathF.Abs(v));
            }
            return max;
        }

        static float[][] MakeMeasuredData()
        {
            var d1 = DataGen.Measurements(new[] {  2.0f,  0.5f, -1.5f }, 100, 0.3f);
            var d2 = DataGen.Measurements(new[] {  1.0f,  1.5f,  2.0f },  50, 0.1f);
            var d3 = DataGen.Measurements(new[] {  0.0f, -0.5f,  0.5f }, 200, 0.2f);
            var d4 = DataGen.Measurements(new[] { -0.2f,  0.0f,  1.0f },  80, 0.3f);
            return d1.Concat(d2).Concat(d3).Concat(d4).ToArray();
        }

        // Synthetic distributions for Step 5.2
        static float[][] MakeUniformRandom(int n, int dim, int seed)
        {
            var rng = new Random(seed);
            return Enumerable.Range(0, n)
                .Select(_ => Enumerable.Range(0, dim)
                    .Select(__ => (float)(rng.NextDouble() * 2.0 - 1.0))
                    .ToArray())
                .ToArray();
        }

        static float[][] MakeGaussian(int n, int dim, int seed)
        {
            var rng = new Random(seed);
            return Enumerable.Range(0, n)
                .Select(_ => Enumerable.Range(0, dim)
                    .Select(__ =>
                    {
                        // Box-Muller
                        double u1 = rng.NextDouble() + 1e-10;
                        double u2 = rng.NextDouble();
                        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                    })
                    .ToArray())
                .ToArray();
        }

        static float[][] MakeSkewed(int n, int dim, int seed)
        {
            // Slightly skewed: Gaussian + small bias
            var base_ = MakeGaussian(n, dim, seed);
            return base_.Select(x => x.Select((v, i) => v + 0.3f * i).ToArray()).ToArray();
        }

        static float[][] MakeScaled(float[][] Xn, float scale)
        {
            return Xn.Select(x => x.Select(v => v * scale).ToArray()).ToArray();
        }

        static bool Check(string name, bool condition, string detail = "")
        {
            string tag    = condition ? "[PASS]" : "[FAIL]";
            string suffix = !condition && detail.Length > 0 ? $"  ({detail})" : "";
            Console.WriteLine($"  {tag} {name}{suffix}");
            if (condition) _pass++; else _fail++;
            return condition;
        }

        static void PrintSummary()
        {
            int total = _pass + _fail;
            Console.WriteLine($"\n====================================================");
            Console.WriteLine($"  Guardrail result: {_pass}/{total} passed");
            if (_fail > 0)
                Console.WriteLine($"  {_fail} FAILURE(s) — regression detected");
            Console.WriteLine("====================================================\n");

            if (_fail > 0)
                throw new GuardrailException(_fail);
        }
    }

    // ============================================================
    // GuardrailException
    // ============================================================
    sealed class GuardrailException : Exception
    {
        public int FailureCount { get; }
        public GuardrailException(int failures)
            : base($"ExampleTVAE guardrails failed: {failures} invariant(s) violated.")
        {
            FailureCount = failures;
        }
    }
}
