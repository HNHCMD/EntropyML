using System.Data;
using EntropyML;

namespace ExampleVAE
{
    class ExampleVAE
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
        // VAE components: encoder, decoder, sampling
        // ------------------------------------------------------------

        // Encoder: x → (μ, logσ²)
        static Sequential BuildEncoder()
        {
            //return new Sequential()
            //    .Dense(6, Activation.Tanh, inputSize: 3)
            //    .Dense(4, Activation.Tanh)
            //    .Dense(4, Activation.Linear); // outputs: [μ1, μ2, logσ²1, logσ²2]

            //return new Sequential()
            //    .Dense(4, Activation.Tanh, inputSize: 3)
            //    .Dense(4, Activation.Linear); // outputs: [μ1, μ2, logσ²1, logσ²2]

            return new Sequential()
                .Dense(3, Activation.Tanh, inputSize: 3)
                .Dense(4, Activation.Linear); // outputs: [μ1, μ2, logσ²1, logσ²2]
        }

        // Decoder: z → x̂
        static Sequential BuildDecoder()
        {
            return new Sequential()
                .Dense(4, Activation.Tanh, inputSize: 2)
                .Dense(6, Activation.Tanh)
                .Dense(3, Activation.Linear); // reconstruct 3D input

            //return new Sequential()
            //    .Dense(6, Activation.Tanh, inputSize: 2)
            //    .Dense(6, Activation.Tanh)
            //    .Dense(3, Activation.Linear); // reconstruct 3D input

            //return new Sequential()
            //    .Dense(8, Activation.Tanh, inputSize: 2)
            //    .Dense(6, Activation.Tanh)
            //    .Dense(3, Activation.Linear); // reconstruct 3D input

            //return new Sequential()
            //    .Dense(4, Activation.Tanh, inputSize: 2)
            //    .Dense(3, Activation.Linear); // reconstruct 3D input

        }

        // Reparameterization: z = μ + σ * ε
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
        // Train VAE
        // ------------------------------------------------------------       
        public static void TrainVAE()
        {
            Console.WriteLine("\n** ExampleVAE.TrainVAE (Realistic Data) **");

            // 1. Generate realistic measured data
            var X = MakeMeasuredData();

            // 2. Normalize
            var Xn = DataGen.Normalize(X);

            // 3. Build encoder + decoder
            var encoder = BuildEncoder();
            var decoder = BuildDecoder();

            int t = 1;        // Adam timestep
            float lr = 0.001f;

            // 4. Train for a few epochs
            for (int epoch = 0; epoch < 30; epoch++)
            {
                float totalLoss = 0f;

                foreach (var x in Xn)
                {
                    // Forward encoder
                    float[] enc = encoder.Forward(x);

                    float[] mu = new float[] { enc[0], enc[1] };
                    float[] logvar = new float[] { enc[2], enc[3] };

                    // Sample latent z
                    float[] z = SampleZ(mu, logvar);

                    // Forward decoder
                    float[] recon = decoder.Forward(z);

                    // Reconstruction loss (MSE)
                    float recLoss = 0f;
                    for (int i = 0; i < 3; i++)
                        recLoss += (recon[i] - x[i]) * (recon[i] - x[i]);

                    // KL divergence
                    //float kl = -0.5f * (1 + logvar[0] - mu[0] * mu[0] - MathF.Exp(logvar[0])
                    //                  + 1 + logvar[1] - mu[1] * mu[1] - MathF.Exp(logvar[1]));

                    //float beta = 0.1f;
                    //float beta = epoch / 30.0f;
                    //float tt = epoch / 30.0f;
                    //float beta = tt * tt;
                    float beta = 0.01f;

                    float kl = beta * (
                        -0.5f * (
                            1 + logvar[0] - mu[0] * mu[0] - MathF.Exp(logvar[0]) +
                            1 + logvar[1] - mu[1] * mu[1] - MathF.Exp(logvar[1])
                        )
                    );

                    float loss = recLoss + kl;
                    totalLoss += loss;

                    // 1. Reconstruction gradient
                    float[] gradRecon = new float[3];
                    for (int i = 0; i < 3; i++)
                        gradRecon[i] = 2f * (recon[i] - x[i]);

                    // 2. Backprop through decoder
                    float[] gradZ = decoder.Backward(gradRecon, lr, ref t);

                    // 3. Backprop through encoder
                    float[] gradEnc = new float[4];
                    gradEnc[0] = gradZ[0]; // dL/dμ1
                    gradEnc[1] = gradZ[1]; // dL/dμ2
                    gradEnc[2] = 0.5f * (MathF.Exp(logvar[0]) - 1); // dL/dlogσ²1
                    gradEnc[3] = 0.5f * (MathF.Exp(logvar[1]) - 1); // dL/dlogσ²2

                    encoder.Backward(gradEnc, lr, ref t);
                }

                Console.WriteLine($"Epoch {epoch,2}  Loss = {totalLoss / Xn.Length:F4}");
            }

            // 5. Test on a true physical state
            float[] test = P1;
            float[] testN = DataGen.NormalizeSingle(test, X);

            float[] encTest = encoder.Forward(testN);
            float[] muTest = new float[] { encTest[0], encTest[1] };
            float[] logvarTest = new float[] { encTest[2], encTest[3] };
            float[] zTest = SampleZ(muTest, logvarTest);
            float[] reconTest = decoder.Forward(zTest);

            Console.WriteLine($"\nTest Input (raw):     [{test[0]:F3}, {test[1]:F3}, {test[2]:F3}]");
            Console.WriteLine($"Test Input (norm):    [{testN[0]:F3}, {testN[1]:F3}, {testN[2]:F3}]");
            Console.WriteLine($"Latent μ:             [{muTest[0]:F3}, {muTest[1]:F3}]");
            Console.WriteLine($"Latent logσ²:         [{logvarTest[0]:F3}, {logvarTest[1]:F3}]");
            Console.WriteLine($"Reconstruction (norm):[{reconTest[0]:F3}, {reconTest[1]:F3}, {reconTest[2]:F3}]");
        }
    }
}
