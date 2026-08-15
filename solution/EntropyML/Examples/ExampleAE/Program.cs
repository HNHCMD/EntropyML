using System.Data;
using EntropyML;

namespace ExampleAE
{
    class ExampleAE
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
        // Train Autoencoder
        // ------------------------------------------------------------
        public static void TrainAE()
        {
            Console.WriteLine("\n** ExampleAE.TrainAE (Realistic Data) **");

            // 1. Generate realistic measured data
            var X = MakeMeasuredData();

            // 2. Normalize (important for tanh AE)
            var Xn = DataGen.Normalize(X);

            // 3. Build a tiny autoencoder
            // Encoder: 3 → 2
            // Decoder: 2 → 3
            var ae = new Sequential()
                .Dense(4, Activation.Tanh, inputSize: 3)
                .Dense(2, Activation.Tanh)       // latent space
                .Dense(4, Activation.Tanh)
                .Dense(3, Activation.Linear);    // reconstruction

            int t = 1;        // Adam timestep
            float lr = 0.001f;

            // 4. Train for a few epochs
            for (int epoch = 0; epoch < 30; epoch++)
            {
                float loss = ae.TrainBatch(Xn, Xn, ref t);
                Console.WriteLine($"Epoch {epoch,2}  Loss = {loss:F4}");
            }

            // 5. Test reconstruction on a true physical state
            float[] test = P1;

            // Normalize test point using same mean/std
            float[] testN = DataGen.NormalizeSingle(test, X);

            float[] recon = ae.Forward(testN);

            Console.WriteLine($"\nTest Input (raw):     [{test[0]:F3}, {test[1]:F3}, {test[2]:F3}]");
            Console.WriteLine($"Test Input (norm):    [{testN[0]:F3}, {testN[1]:F3}, {testN[2]:F3}]");
            Console.WriteLine($"Reconstruction (norm):[{recon[0]:F3}, {recon[1]:F3}, {recon[2]:F3}]");
        }

        static void Main(string[] args)
        {
            TrainAE();
        }
    }
}
