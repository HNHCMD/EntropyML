using EntropyML;

namespace ExampleTVAE
{
    class ExampleTVAE
    {
        // ------------------------------------------------------------
        // True underlying physical states (3-dimensional)
        // ------------------------------------------------------------
        static readonly float[] P1 = { 2.0f, 0.5f, -1.5f };
        static readonly float[] P2 = { 1.0f, 1.5f, 2.0f };
        static readonly float[] P3 = { 0.0f, -0.5f, 0.5f };
        static readonly float[] P4 = { -0.2f, 0.0f, 1.0f };

        // ------------------------------------------------------------
        // Generate realistic measured data
        // ------------------------------------------------------------
        static float[][] MakeMeasuredData()
        {
            var d1 = DataGen.Measurements(P1, 100, 0.3f);
            var d2 = DataGen.Measurements(P2, 50, 0.1f);
            var d3 = DataGen.Measurements(P3, 200, 0.2f);
            var d4 = DataGen.Measurements(P4, 80, 0.3f);

            return d1.Concat(d2).Concat(d3).Concat(d4).ToArray();
        }

        // ------------------------------------------------------------
        // Compute energy term (MSE reconstruction) from public API outputs
        // energyTerm = (1/D) * Σ(x - relaxedMicrostate)²
        // ------------------------------------------------------------
        static float ComputeEnergyTerm(float[] x, float[] relaxedMicrostate)
        {
            float mse = 0f;
            for (int i = 0; i < x.Length; i++)
                mse += (x[i] - relaxedMicrostate[i]) * (x[i] - relaxedMicrostate[i]);
            return mse / x.Length;
        }

        // ------------------------------------------------------------
        // Train TVAE
        // ------------------------------------------------------------
        public static void TrainTVAE()
        {
            Console.WriteLine("\n** ExampleTVAE.TrainTVAE (Realistic Data) **");

            // 1. Generate realistic measured data
            var X = MakeMeasuredData();

            // 2. Normalize
            var Xn = DataGen.Normalize(X);

            // 3. Construct TVAE model
            var tvae = new TVAE(inputDim: 3, latentDim: 2);

            // 4. Train one epoch at a time to emit thermodynamic per-epoch logs
            for (int epoch = 0; epoch < 30; epoch++)
            {
                // Advance one epoch (suppress Fit's internal logging)
                var history = tvae.Fit(Xn, epochs: 1, verbose: 0);

                // history[0] is the average free energy for this epoch (from Fit)
                float avgFreeEnergy = history[0];

                // Post-epoch evaluation pass: decompose and inspect latent geometry
                float totalEnergy = 0f;
                float totalEntropy = 0f;
                float eqMin = float.MaxValue, eqMax = float.MinValue;

                foreach (var x in Xn)
                {
                    var (eq, entPot) = tvae.EncodeThermoState(x);
                    float[] relaxed = tvae.RelaxAndReconstruct(x);

                    // energyTerm: MSE reconstruction (non-negative by construction)
                    float energyTerm = ComputeEnergyTerm(x, relaxed);

                    // freeEnergy from TVAE API
                    float fe = tvae.ComputeFreeEnergy(x, relaxed, eq, entPot);

                    // entropyTerm: derived as Beta * KL = freeEnergy - energyTerm (non-negative)
                    float entropyTerm = fe - energyTerm;

                    totalEnergy += energyTerm;
                    totalEntropy += entropyTerm;

                    // Track equilibrium (μ analogue) range across all samples and both dims
                    eqMin = MathF.Min(eqMin, MathF.Min(eq[0], eq[1]));
                    eqMax = MathF.Max(eqMax, MathF.Max(eq[0], eq[1]));
                }

                float n = Xn.Length;
                Console.WriteLine(
                    $"Epoch {epoch,2}  FreeEnergy={avgFreeEnergy:F4}  " +
                    $"Energy={totalEnergy / n:F4}  Entropy={totalEntropy / n:F4}  " +
                    $"eq-range=[{eqMin:F2},{eqMax:F2}]");
            }

            // 5. Test on a true physical state
            float[] test = P1;
            float[] testN = DataGen.NormalizeSingle(test, X);

            var (equilibrium, entropyPotential) = tvae.EncodeThermoState(testN);
            float[] relaxedMicrostate = tvae.RelaxAndReconstruct(testN);
            float freeEnergyTest = tvae.ComputeFreeEnergy(testN, relaxedMicrostate, equilibrium, entropyPotential);
            float energyTest = ComputeEnergyTerm(testN, relaxedMicrostate);
            float entropyTest = freeEnergyTest - energyTest;

            Console.WriteLine($"\nTest Input (raw):                    [{test[0]:F3}, {test[1]:F3}, {test[2]:F3}]");
            Console.WriteLine($"Test Input (norm):                   [{testN[0]:F3}, {testN[1]:F3}, {testN[2]:F3}]");
            Console.WriteLine($"Latent μ (equilibrium):              [{equilibrium[0]:F3}, {equilibrium[1]:F3}]");
            Console.WriteLine($"Latent logσ² (entropy potential):    [{entropyPotential[0]:F3}, {entropyPotential[1]:F3}]");
            Console.WriteLine($"Energy/Reconstruction (norm):        [{relaxedMicrostate[0]:F3}, {relaxedMicrostate[1]:F3}, {relaxedMicrostate[2]:F3}]");
            Console.WriteLine($"Energy term:                         {energyTest:F4}");
            Console.WriteLine($"Entropy term (Beta*KL):              {entropyTest:F4}");
            Console.WriteLine($"FreeEnergy (energy + entropy):       {freeEnergyTest:F4}");
        }

        // ------------------------------------------------------------
        // Test TVAE
        // ------------------------------------------------------------
        public static void TestTVAE()
        {
            Console.WriteLine("\n** ExampleTVAE.TestTVAE **");

            // 1. Generate realistic measured data
            var X = MakeMeasuredData();

            // 2. Construct TVAE model
            var tvae = new TVAE(inputDim: 3, latentDim: 2);

            // 3. Test on a true physical state
            float[] test = P1;
            float[] testN = DataGen.NormalizeSingle(test, X);

            var (equilibrium, entropyPotential) = tvae.EncodeThermoState(testN);
            float[] relaxedMicrostate = tvae.RelaxAndReconstruct(testN);
            float freeEnergy = tvae.ComputeFreeEnergy(testN, relaxedMicrostate, equilibrium, entropyPotential);
            float energyTerm = ComputeEnergyTerm(testN, relaxedMicrostate);
            float entropyTerm = freeEnergy - energyTerm;

            Console.WriteLine($"\nTest Input (raw):                    [{test[0]:F3}, {test[1]:F3}, {test[2]:F3}]");
            Console.WriteLine($"Test Input (norm):                   [{testN[0]:F3}, {testN[1]:F3}, {testN[2]:F3}]");
            Console.WriteLine($"Latent μ (equilibrium):              [{equilibrium[0]:F3}, {equilibrium[1]:F3}]");
            Console.WriteLine($"Latent logσ² (entropy potential):    [{entropyPotential[0]:F3}, {entropyPotential[1]:F3}]");
            Console.WriteLine($"Energy/Reconstruction (norm):        [{relaxedMicrostate[0]:F3}, {relaxedMicrostate[1]:F3}, {relaxedMicrostate[2]:F3}]");
            Console.WriteLine($"Energy term:                         {energyTerm:F4}");
            Console.WriteLine($"Entropy term (Beta*KL):              {entropyTerm:F4}");
            Console.WriteLine($"FreeEnergy (energy + entropy):       {freeEnergy:F4}");
        }
    }
}
