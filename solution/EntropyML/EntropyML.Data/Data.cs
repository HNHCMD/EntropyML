namespace EntropyML
{
    public static class DataGen
    {
        private static readonly Random Rnd = new Random(123);

        // --------------------------------------------------------------------
        // Gaussian(...) is the general-purpose synthetic data generator.
        // Microstates(...) intentionally calls the same function.
        //
        // This dual naming is pedagogical:
        //   - AE/VAE examples use "Gaussian" to emphasize ML context.
        //   - Episodes use "Microstates" to emphasize thermodynamic meaning.
        //
        // Both return identical data by design.
        // --------------------------------------------------------------------

        public static float[][] Gaussian(int n, int dim)
        {
            var rnd = new Random(123);
            var data = new float[n][];

            for (int i = 0; i < n; i++)
            {
                float[] x = new float[dim];
                for (int d = 0; d < dim; d++)
                    x[d] = NextGaussian(rnd);

                data[i] = x;
            }

            return data;
        }

        public static float[][] Microstates(int n, int dim)
        {
            return Gaussian(n, dim);
        }

        // --------------------------------------------------------------------
        // Measurements(...) generates realistic measured data around a true
        // physical state (center). Noise is Gaussian with specified std.
        //
        // This prepares beginners for AE/VAE/TVAE:
        //   - AE: compress noisy measured data
        //   - VAE: learn distributions of clusters
        //   - TVAE: interpret clusters as thermodynamic microstates
        // --------------------------------------------------------------------

        public static float[][] Measurements(float[] center, int count, float std)
        {
            var rnd = new Random();
            var data = new float[count][];

            for (int i = 0; i < count; i++)
            {
                float[] x = new float[center.Length];
                for (int d = 0; d < center.Length; d++)
                    x[d] = center[d] + std * NextGaussian(rnd);

                data[i] = x;
            }

            return data;
        }

        // --------------------------------------------------------------------
        // Gaussian noise generator (Box–Muller)
        // --------------------------------------------------------------------
        public static float NextGaussian(Random rnd)
        {
            float u1 = 1f - (float)rnd.NextDouble(); // avoid log(0)
            float u2 = 1f - (float)rnd.NextDouble();
            float r = MathF.Sqrt(-2f * MathF.Log(u1));
            float theta = 2f * MathF.PI * u2;
            return r * MathF.Cos(theta);
        }

        public static float[] NormalizeSingle(float[] x, float[][] data)
        {
            int dim = x.Length;

            float[] mean = new float[dim];
            float[] std = new float[dim];

            // Compute mean
            for (int d = 0; d < dim; d++)
                mean[d] = data.Average(v => v[d]);

            // Compute std
            for (int d = 0; d < dim; d++)
                std[d] = MathF.Sqrt(data.Average(v => (v[d] - mean[d]) * (v[d] - mean[d])));

            // Normalize single point
            float[] xn = new float[dim];
            for (int d = 0; d < dim; d++)
                xn[d] = (x[d] - mean[d]) / std[d];

            return xn;
        }


        public static float[][] Normalize(float[][] data)
        {
            int n = data.Length;
            int dim = data[0].Length;

            float[] mean = new float[dim];
            float[] std = new float[dim];

            // Compute mean
            for (int d = 0; d < dim; d++)
                mean[d] = data.Average(x => x[d]);

            // Compute std
            for (int d = 0; d < dim; d++)
                std[d] = MathF.Sqrt(data.Average(x => (x[d] - mean[d]) * (x[d] - mean[d])));

            // Normalize
            var norm = new float[n][];
            for (int i = 0; i < n; i++)
            {
                float[] x = new float[dim];
                for (int d = 0; d < dim; d++)
                    x[d] = (data[i][d] - mean[d]) / std[d];
                norm[i] = x;
            }

            return norm;
        }


    }
}
