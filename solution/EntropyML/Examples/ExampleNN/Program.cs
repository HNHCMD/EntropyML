using EntropyML;

namespace ExampleNN
{
    class ExampleNN
    {
        static class Ex01
        {
            static float[][] MakeInputs(int n, int dim)
            {
                return DataGen.Gaussian(n, dim);
            }

            static float[][] MakeTargets(float[][] X)
            {
                // Simple target: y = x0 + x1
                return X.Select(x => new float[] { x[0] + x[1] }).ToArray();
            }

            public static void TrainNN()
            {
                Console.WriteLine("\n** ExampleNN.Ex01.TrainNN() **");

                int samples = 2000;
                int dim = 2;

                // 1. Generate synthetic Gaussian data
                var X = MakeInputs(samples, dim);
                var Y = MakeTargets(X);

                // 2. Build a tiny neural network
                var nn = new Sequential()
                    .Dense(8, Activation.Tanh, inputSize: dim)
                    .Dense(1, Activation.Linear);

                int t = 1; // Adam timestep
                float lr = 0.001f;

                // 3. Train for a few epochs
                for (int epoch = 0; epoch < 20; epoch++)
                {
                    float loss = nn.TrainBatch(X, Y, ref t);
                    Console.WriteLine($"Epoch {epoch,2}  Loss = {loss,6:F4}");
                }

                // 4. Test on a new sample
                float[] test = new float[] { 0.5f, -1.2f };
                float[] pred = nn.Forward(test);
                float target = test[0] + test[1];

                Console.WriteLine($"\nTest Input:  [{test[0]:F3}, {test[1]:F3}]");
                Console.WriteLine($"True y = {target:F3}");
                Console.WriteLine($"Pred y = {pred[0]:F3}");
            }
        }

        static class Ex02
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
            // Simple target: sum of coordinates (for demonstration)
            // ------------------------------------------------------------
            static float[][] MakeTargets(float[][] X)
            {
                return X.Select(x => new float[] { x[0] + x[1] + x[2] }).ToArray();
            }

            // ------------------------------------------------------------
            // Train NN on realistic measured data
            // ------------------------------------------------------------
            public static void TrainNN()
            {
                Console.WriteLine("\n** ExampleNN.Ex02.TrainNN (Realistic Data) **");

                // 1. Generate realistic measured data
                var X = MakeMeasuredData();
                var Y = MakeTargets(X);

                // 2. Build a tiny neural network
                var nn = new Sequential()
                    .Dense(8, Activation.Tanh, inputSize: 3)
                    .Dense(1, Activation.Linear);

                int t = 1;        // Adam timestep
                float lr = 0.001f;

                // 3. Train for a few epochs
                for (int epoch = 0; epoch < 20; epoch++)
                {
                    float loss = nn.TrainBatch(X, Y, ref t);
                    Console.WriteLine($"Epoch {epoch,2}  Loss = {loss:F4}");
                }

                // 4. Test on a true physical state
                float[] test = P1;
                float[] pred = nn.Forward(test);
                float target = P1[0] + P1[1] + P1[2];

                Console.WriteLine($"\nTest Input:  [{test[0]:F3}, {test[1]:F3}, {test[2]:F3}]");
                Console.WriteLine($"True y = {target:F3}");
                Console.WriteLine($"Pred y = {pred[0]:F3}");
            }
        }
        static void Main(string[] args)
        {
            //Ex01.TrainNN();
            Ex02.TrainNN();
        }
    }
}
