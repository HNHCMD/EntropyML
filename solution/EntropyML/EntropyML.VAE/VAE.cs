namespace EntropyML
{
    public class VAE
    {
        public int InputDim { get; private set; }
        public int LatentDim { get; private set; }
        public float Beta { get; set; } = 0.1f;  // ← Sweet spot

        private Sequential encoder_mu;      // x → μ
        private Sequential encoder_logvar;  // x → log(σ²)
        private Sequential decoder;         // z → x

        // Training state (for backprop)
        private Random random;
        private int t = 0;  // Adam timestep

        public VAE(int inputDim, int latentDim, int hiddenDim = 64, int randSeed = 42)
        {
            InputDim = inputDim;
            LatentDim = latentDim;
            random = new Random(Seed: randSeed);  // Fixed seed for reproducibility

            // Build encoder for μ (reuses your DenseLayer!)
            encoder_mu = new Sequential()
                .Dense(hiddenDim, Activation.Tanh, inputSize: inputDim)
                .Dense(latentDim, Activation.Linear);

            // Build encoder for log(σ²) (same architecture)
            encoder_logvar = new Sequential()
                .Dense(hiddenDim, Activation.Tanh, inputSize: inputDim)
                .Dense(latentDim, Activation.Linear);

            // Build decoder (symmetric)
            decoder = new Sequential()
                .Dense(hiddenDim, Activation.Tanh, inputSize: latentDim)
                .Dense(inputDim, Activation.Linear);
        }

        // ====================================================================
        // FORWARD PASS METHODS
        // ====================================================================

        /// <summary>
        /// Encode input to latent distribution parameters.
        /// </summary>
        /// <param name="x">Input vector</param>
        /// <returns>Tuple of (μ, log_σ²)</returns>
        public (float[] mu, float[] logvar) Encode(float[] x)
        {
            float[] mu = encoder_mu.Forward(x);
            float[] logvar = encoder_logvar.Forward(x);
            return (mu, logvar);
        }

        /// <summary>
        /// Sample from latent distribution using reparameterization trick.
        /// z = μ + σ * ε, where ε ~ N(0,1)
        /// </summary>
        public float[] Sample(float[] mu, float[] logvar)
        {
            float[] z = new float[LatentDim];

            for (int i = 0; i < LatentDim; i++)
            {
                float epsilon = (float)GaussianRandom(random);  // ε ~ N(0,1)
                float sigma = (float)Math.Exp(0.5 * logvar[i]); // σ = exp(0.5 * log_σ²)
                z[i] = mu[i] + sigma * epsilon;                 // z = μ + σ * ε
            }

            return z;
        }

        /// <summary>
        /// Sample with stored epsilon (for backprop).
        /// Returns both z and epsilon for gradient computation.
        /// </summary>
        private (float[] z, float[] epsilon) SampleWithEpsilon(float[] mu, float[] logvar)
        {
            float[] z = new float[LatentDim];
            float[] epsilon = new float[LatentDim];

            for (int i = 0; i < LatentDim; i++)
            {
                epsilon[i] = (float)GaussianRandom(random);     // ε ~ N(0,1)
                float sigma = (float)Math.Exp(0.5 * logvar[i]); // σ = exp(0.5 * log_σ²)
                z[i] = mu[i] + sigma * epsilon[i];              // z = μ + σ * ε
            }

            return (z, epsilon);
        }

        /// <summary>
        /// Decode latent vector to reconstruction.
        /// </summary>
        public float[] Decode(float[] z)
        {
            return decoder.Forward(z);
        }

        /// <summary>
        /// Full forward pass: x → μ, log_σ² → z → x_recon
        /// </summary>
        public float[] Reconstruct(float[] x)
        {
            var (mu, logvar) = Encode(x);
            float[] z = Sample(mu, logvar);
            return Decode(z);
        }

        // ====================================================================
        // LOSS COMPUTATION
        // ====================================================================

        /// <summary>
        /// Compute VAE loss: Reconstruction + β * KL
        /// </summary>
        public float ComputeLoss(float[] x, float[] x_recon, float[] mu, float[] logvar)
        {
            // Reconstruction loss (MSE)
            float recon_loss = 0;
            for (int i = 0; i < InputDim; i++)
            {
                float diff = x[i] - x_recon[i];
                recon_loss += diff * diff;
            }
            recon_loss /= InputDim;

            // KL divergence: -0.5 * Σ(1 + log_σ² - μ² - σ²)
            float kl_loss = 0;
            for (int i = 0; i < LatentDim; i++)
            {
                kl_loss += 1 + logvar[i] - mu[i] * mu[i] - (float)Math.Exp(logvar[i]);
            }
            kl_loss *= -0.5f;

            return recon_loss + Beta * kl_loss;
        }

        // ====================================================================
        // BACKPROPAGATION (EDUCATIONAL IMPLEMENTATION)
        // ====================================================================

        /// <summary>
        /// Compute gradients and update parameters for one training sample.
        /// This is the heart of VAE training - showing all gradient computations.
        /// </summary>
        private float TrainSample(float[] x, float learningRate)
        {
            t++;  // Increment Adam timestep

            // ================================================================
            // FORWARD PASS (with saved intermediate values)
            // ================================================================

            // 1. Encode: x → (μ, log_σ²)
            var (mu, logvar) = Encode(x);

            // 2. Sample: (μ, log_σ²) → z (save ε for backprop!)
            var (z, epsilon) = SampleWithEpsilon(mu, logvar);

            // 3. Decode: z → x_recon
            float[] x_recon = Decode(z);

            // 4. Compute loss
            float loss = ComputeLoss(x, x_recon, mu, logvar);

            // ================================================================
            // BACKWARD PASS - GRADIENT COMPUTATION
            // ================================================================

            // ────────────────────────────────────────────────────────────────
            // STEP 1: Compute ∂L_recon/∂x_recon (reconstruction gradient)
            // ────────────────────────────────────────────────────────────────
            // L_recon = (1/D) * Σ(x - x_recon)²
            // ∂L_recon/∂x_recon = -2/D * (x - x_recon)

            float[] grad_x_recon = new float[InputDim];
            for (int i = 0; i < InputDim; i++)
            {
                grad_x_recon[i] = -2.0f / InputDim * (x[i] - x_recon[i]);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 2: Backprop through decoder: ∂L_recon/∂z
            // ────────────────────────────────────────────────────────────────
            // Use decoder's backward pass to compute gradient w.r.t. z
            float[] grad_z = decoder.Layers.Last().Backward(grad_x_recon, learningRate, t);
            for (int i = decoder.Layers.Count - 2; i >= 0; i--)
            {
                grad_z = decoder.Layers[i].Backward(grad_z, learningRate, t);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 3: Compute KL divergence gradients
            // ────────────────────────────────────────────────────────────────
            // L_KL = -0.5 * Σ(1 + log_σ² - μ² - exp(log_σ²))
            //
            // ∂L_KL/∂μ = -0.5 * (-2μ) = μ
            // ∂L_KL/∂log_σ² = -0.5 * (1 - exp(log_σ²))
            //               = 0.5 * (exp(log_σ²) - 1)

            float[] grad_mu_kl = new float[LatentDim];
            float[] grad_logvar_kl = new float[LatentDim];

            for (int i = 0; i < LatentDim; i++)
            {
                grad_mu_kl[i] = Beta * mu[i];

                // FIXED: Correct gradient sign
                float exp_logvar = (float)Math.Exp(logvar[i]);
                grad_logvar_kl[i] = Beta * 0.5f * (exp_logvar - 1.0f);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 4: Reparameterization trick gradients
            // ────────────────────────────────────────────────────────────────
            // z = μ + σ * ε, where σ = exp(0.5 * log_σ²)
            //
            // ∂z/∂μ = 1
            // ∂z/∂log_σ² = ∂z/∂σ * ∂σ/∂log_σ²
            //            = ε * (0.5 * exp(0.5 * log_σ²))
            //            = 0.5 * σ * ε
            //
            // Chain rule from decoder:
            // ∂L_recon/∂μ = ∂L_recon/∂z * ∂z/∂μ = grad_z
            // ∂L_recon/∂log_σ² = ∂L_recon/∂z * ∂z/∂log_σ²

            float[] grad_mu_recon = new float[LatentDim];
            float[] grad_logvar_recon = new float[LatentDim];

            for (int i = 0; i < LatentDim; i++)
            {
                // Gradient w.r.t. μ (from reconstruction)
                grad_mu_recon[i] = grad_z[i];  // ∂z/∂μ = 1

                // Gradient w.r.t. log_σ² (from reconstruction)
                float sigma = (float)Math.Exp(0.5 * logvar[i]);
                grad_logvar_recon[i] = grad_z[i] * 0.5f * sigma * epsilon[i];
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 5: Combine gradients (reconstruction + KL)
            // ────────────────────────────────────────────────────────────────

            float[] grad_mu = new float[LatentDim];
            float[] grad_logvar = new float[LatentDim];

            for (int i = 0; i < LatentDim; i++)
            {
                grad_mu[i] = grad_mu_recon[i] + grad_mu_kl[i];
                grad_logvar[i] = grad_logvar_recon[i] + grad_logvar_kl[i];
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 6: Backprop through encoder_mu
            // ────────────────────────────────────────────────────────────────

            float[] grad_mu_input = grad_mu;
            for (int i = encoder_mu.Layers.Count - 1; i >= 0; i--)
            {
                grad_mu_input = encoder_mu.Layers[i].Backward(grad_mu_input, learningRate, t);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 7: Backprop through encoder_logvar
            // ────────────────────────────────────────────────────────────────

            float[] grad_logvar_input = grad_logvar;
            for (int i = encoder_logvar.Layers.Count - 1; i >= 0; i--)
            {
                grad_logvar_input = encoder_logvar.Layers[i].Backward(grad_logvar_input, learningRate, t);
            }

            // Note: grad_mu_input and grad_logvar_input are gradients w.r.t. input x
            // We don't need them (input is fixed), but they could be useful for
            // gradient-based input optimization or saliency maps.

            return loss;
        }

        // ====================================================================
        // TRAINING
        // ====================================================================

        /// <summary>
        /// Train VAE on dataset using full backpropagation.
        /// </summary>
        /// <param name="X">Training data [num_samples × input_dim]</param>
        /// <param name="epochs">Number of training epochs</param>
        /// <param name="batchSize">Batch size (currently processes samples individually)</param>
        /// <param name="learningRate">Learning rate for Adam optimizer</param>
        /// <param name="verbose">Print frequency (0 = silent, N = every N epochs)</param>
        /// <returns>Training loss history</returns>
        public List<float> Fit(float[][] X, int epochs = 500, int batchSize = 32,
                              float learningRate = 0.001f, int verbose = 100)
        {
            // Set learning rates for all networks
            encoder_mu.LearningRate = learningRate;
            encoder_logvar.LearningRate = learningRate;
            decoder.LearningRate = learningRate;

            var lossHistory = new List<float>();

            Console.WriteLine("Training Variational Autoencoder...");
            Console.WriteLine($"  Samples: {X.Length}");
            Console.WriteLine($"  Architecture: {InputDim} → ({LatentDim} μ, {LatentDim} log_σ²) → {InputDim}");
            Console.WriteLine($"  Epochs: {epochs}");
            Console.WriteLine($"  Learning rate: {learningRate}");
            Console.WriteLine($"  Beta (KL weight): {Beta}\n");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float epoch_loss = 0;

                // Shuffle data (simple random permutation)
                var indices = Enumerable.Range(0, X.Length).OrderBy(x => random.Next()).ToArray();

                // Train on each sample
                foreach (int idx in indices)
                {
                    float sample_loss = TrainSample(X[idx], learningRate);
                    epoch_loss += sample_loss;
                }

                float avg_loss = epoch_loss / X.Length;
                lossHistory.Add(avg_loss);

                // Print progress
                if (verbose > 0 && (epoch % verbose == 0 || epoch == epochs - 1))
                {
                    Console.WriteLine($"Epoch {epoch,4}: Loss = {avg_loss:F6}");
                }
            }

            Console.WriteLine($"\n✓ Training complete! Final loss: {lossHistory.Last():F6}");
            return lossHistory;
        }

        // ====================================================================
        // UTILITY METHODS
        // ====================================================================

        /// <summary>
        /// Gaussian random number generator using Box-Muller transform.
        /// </summary>
        private static double GaussianRandom(Random random)
        {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }

        /// <summary>
        /// Transform dataset to latent space (using mean μ, not sampling).
        /// Useful for visualization and analysis.
        /// </summary>
        public float[][] Transform(float[][] X)
        {
            var Z = new float[X.Length][];
            for (int i = 0; i < X.Length; i++)
            {
                var (mu, logvar) = Encode(X[i]);
                Z[i] = mu;  // Use mean, not sample (deterministic)
            }
            return Z;
        }

        /// <summary>
        /// Compute reconstruction error on dataset.
        /// </summary>
        public float ReconstructionError(float[][] X)
        {
            float total_error = 0;

            foreach (var x in X)
            {
                var x_recon = Reconstruct(x);

                for (int i = 0; i < InputDim; i++)
                {
                    float diff = x[i] - x_recon[i];
                    total_error += diff * diff;
                }
            }

            return total_error / (X.Length * InputDim);
        }

        /// <summary>
        /// Extract learned uncertainty (σ) for each sample.
        /// </summary>
        public float[][] ExtractSigma(float[][] X)
        {
            var Sigma = new float[X.Length][];

            for (int i = 0; i < X.Length; i++)
            {
                var (mu, logvar) = Encode(X[i]);
                Sigma[i] = new float[LatentDim];

                for (int j = 0; j < LatentDim; j++)
                {
                    Sigma[i][j] = (float)Math.Sqrt(Math.Exp(logvar[j]));  // σ = sqrt(exp(log_σ²))
                }
            }

            return Sigma;
        }

        /// <summary>
        /// Print model summary.
        /// </summary>
        public void Summary()
        {
            Console.WriteLine("\n╔════════════════════════════════════════════════════════╗");
            Console.WriteLine("║  Variational Autoencoder Architecture                  ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════╝");
            Console.WriteLine($"\nInput dimension:   {InputDim}");
            Console.WriteLine($"Latent dimension:  {LatentDim}");
            Console.WriteLine($"Beta (KL weight):  {Beta}");
            Console.WriteLine("\nEncoder μ:");
            foreach (var layer in encoder_mu.Layers)
            {
                Console.WriteLine($"  {layer.InputSize} → {layer.OutputSize} ({layer.Act})");
            }
            Console.WriteLine("\nEncoder log_σ²:");
            foreach (var layer in encoder_logvar.Layers)
            {
                Console.WriteLine($"  {layer.InputSize} → {layer.OutputSize} ({layer.Act})");
            }
            Console.WriteLine("\nDecoder:");
            foreach (var layer in decoder.Layers)
            {
                Console.WriteLine($"  {layer.InputSize} → {layer.OutputSize} ({layer.Act})");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Estimate input-space noise from reconstruction residuals.
        /// This is the σ_input you want to compare with true noise!
        /// </summary>
        public float EstimateInputNoise(float[][] X)
        {
            float total_variance = 0;

            foreach (var x in X)
            {
                var (mu, logvar) = Encode(x);
                var x_recon = Decode(mu);  // Use μ, not sampling

                for (int i = 0; i < InputDim; i++)
                {
                    float residual = x[i] - x_recon[i];
                    total_variance += residual * residual;
                }
            }

            float mse = total_variance / (X.Length * InputDim);
            return (float)Math.Sqrt(mse);  // RMSE ≈ σ_input estimate
        }
    }
}
