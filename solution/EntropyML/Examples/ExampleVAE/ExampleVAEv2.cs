using System.Data;
using EntropyML;

namespace ExampleVAE
{

    //
    // ============================================================
    // ExampleVAEv2 — Clean, pedagogical, stable VAE implementation
    // ============================================================
    //
    // Architecture:
    //
    //   Encoder:
    //       x(3) → Dense(3,Tanh) → Dense(4,Linear)
    //       Output: [μ1, μ2, logσ²1, logσ²2]
    //
    //   Decoder:
    //       z(2) → Dense(4,Tanh) → Dense(6,Tanh) → Dense(3,Linear)
    //       Output: reconstructed x̂(3)
    //
    // Latent dimension: 2
    // KL weight (β):    0.01
    // Optimizer:        Adam (lr = 0.001)
    // Epochs:           30
    //
    // ============================================================
    //

    class ExampleVAEv2
    {
        // ------------------------------------------------------------
        // True underlying physical states (3D)
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
        // Encoder: x → (μ, logσ²)
        // ------------------------------------------------------------
        static Sequential BuildEncoder()
        {
            return new Sequential()
                .Dense(3, Activation.Tanh, inputSize: 3)
                .Dense(4, Activation.Linear);   // [μ1, μ2, logσ²1, logσ²2]
        }

        // ------------------------------------------------------------
        // Decoder: z → x̂
        // ------------------------------------------------------------
        static Sequential BuildDecoder()
        {
            return new Sequential()
                .Dense(4, Activation.Tanh, inputSize: 2)
                .Dense(6, Activation.Tanh)
                .Dense(3, Activation.Linear);   // reconstruct 3D input
        }

        // ------------------------------------------------------------
        // Reparameterization: z = μ + σ * ε
        // ------------------------------------------------------------
        static float[] SampleZ(float[] mu, float[] logvar)
        {
            float[] z = new float[mu.Length];
            var rnd = new Random();

            for (int i = 0; i < mu.Length; i++)
            {
                float eps = DataGen.NextGaussian(rnd);
                float sigma = MathF.Exp(0.5f * logvar[i]);
                z[i] = mu[i] + sigma * eps;
            }
            return z;
        }

        // ------------------------------------------------------------
        // Compute KL divergence (scaled by β)
        // ------------------------------------------------------------
        static float ComputeKL(float[] mu, float[] logvar, float beta)
        {
            float kl = 0f;
            for (int i = 0; i < mu.Length; i++)
            {
                kl += -0.5f * (1 + logvar[i] - mu[i] * mu[i] - MathF.Exp(logvar[i]));
            }
            return beta * kl;
        }

        // ------------------------------------------------------------
        // Compute reconstruction loss (MSE)
        // ------------------------------------------------------------
        static float ComputeRecLoss(float[] x, float[] xhat)
        {
            float s = 0f;
            for (int i = 0; i < x.Length; i++)
                s += (xhat[i] - x[i]) * (xhat[i] - x[i]);
            return s;
        }

        // ------------------------------------------------------------
        // Train VAE
        // ------------------------------------------------------------
        public static void TrainVAE()
        {
            Console.WriteLine("\n** ExampleVAEv2.TrainVAE (Realistic Data) **");

            // 1. Generate realistic measured data
            var X = MakeMeasuredData();

            // 2. Normalize
            var Xn = DataGen.Normalize(X);

            // 3. Build encoder + decoder
            var encoder = BuildEncoder();
            var decoder = BuildDecoder();

            float lr = 0.001f;
            float beta = 0.01f;
            int t = 1;   // Adam timestep

            // 4. Train
            for (int epoch = 0; epoch < 30; epoch++)
            {
                float totalLoss = 0f;
                float totalKL = 0f;
                float muMin = float.MaxValue, muMax = float.MinValue;

                foreach (var x in Xn)
                {
                    // Forward encoder
                    float[] enc = encoder.Forward(x);
                    float[] mu = { enc[0], enc[1] };
                    float[] logvar = { enc[2], enc[3] };

                    // Track μ range
                    muMin = MathF.Min(muMin, MathF.Min(mu[0], mu[1]));
                    muMax = MathF.Max(muMax, MathF.Max(mu[0], mu[1]));

                    // Sample latent z
                    float[] z = SampleZ(mu, logvar);

                    // Forward decoder
                    float[] recon = decoder.Forward(z);

                    // Losses
                    float recLoss = ComputeRecLoss(x, recon);
                    float kl = ComputeKL(mu, logvar, beta);
                    float loss = recLoss + kl;

                    totalLoss += loss;
                    totalKL += kl;

                    // Gradients
                    float[] gradRecon = new float[3];
                    for (int i = 0; i < 3; i++)
                        gradRecon[i] = 2f * (recon[i] - x[i]);

                    float[] gradZ = decoder.Backward(gradRecon, lr, ref t);

                    float[] gradEnc = new float[4];
                    gradEnc[0] = gradZ[0];  // dL/dμ1
                    gradEnc[1] = gradZ[1];  // dL/dμ2
                    gradEnc[2] = 0.5f * (MathF.Exp(logvar[0]) - 1); // dL/dlogσ²1
                    gradEnc[3] = 0.5f * (MathF.Exp(logvar[1]) - 1); // dL/dlogσ²2

                    encoder.Backward(gradEnc, lr, ref t);
                }

                Console.WriteLine(
                    $"Epoch {epoch,2}  Loss={totalLoss / Xn.Length:F4}  KL={totalKL / Xn.Length:F4}  μ-range=[{muMin:F2},{muMax:F2}]");
            }

            // 5. Test on a true physical state
            float[] test = P1;
            float[] testN = DataGen.NormalizeSingle(test, X);

            float[] encTest = encoder.Forward(testN);
            float[] muTest = { encTest[0], encTest[1] };
            float[] logvarTest = { encTest[2], encTest[3] };
            float[] zTest = SampleZ(muTest, logvarTest);
            float[] reconTest = decoder.Forward(zTest);

            Console.WriteLine($"\nTest Input (raw):   [{test[0]:F3}, {test[1]:F3}, {test[2]:F3}]");
            Console.WriteLine($"Test Input (norm):    [{testN[0]:F3}, {testN[1]:F3}, {testN[2]:F3}]");
            Console.WriteLine($"Latent μ:             [{muTest[0]:F3}, {muTest[1]:F3}]");
            Console.WriteLine($"Latent logσ²:         [{logvarTest[0]:F3}, {logvarTest[1]:F3}]");
            Console.WriteLine($"Reconstruction (norm):[{reconTest[0]:F3}, {reconTest[1]:F3}, {reconTest[2]:F3}]");
        }
    }
}
