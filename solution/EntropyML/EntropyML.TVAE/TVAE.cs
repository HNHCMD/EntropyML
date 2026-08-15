namespace EntropyML
{
    public class TVAE
    {
        public int MicrostateDim { get; private set; }
        public int ManifoldDim { get; private set; }
        public float Beta { get; set; } = 0.1f;  // ← Sweet spot

        private Sequential equilibriumEncoder;      // x → μ
        private Sequential entropyEncoder;  // x → log(σ²)
        private Sequential relaxationOperator;         // z → x

        // Training state (for backprop)
        private Random random;
        private int t = 0;  // Adam timestep

        public TVAE(int inputDim, int latentDim, int hiddenDim = 64, int randSeed = 42)
        {
            MicrostateDim = inputDim;
            ManifoldDim = latentDim;
            random = new Random(Seed: randSeed);  // Fixed seed for reproducibility

            // Build encoder for μ (reuses your DenseLayer!)
            equilibriumEncoder = new Sequential()
                .Dense(hiddenDim, Activation.Tanh, inputSize: inputDim)
                .Dense(latentDim, Activation.Linear);

            // Build encoder for log(σ²) (same architecture)
            entropyEncoder = new Sequential()
                .Dense(hiddenDim, Activation.Tanh, inputSize: inputDim)
                .Dense(latentDim, Activation.Linear);

            // Build relaxationOperator (symmetric)
            relaxationOperator = new Sequential()
                .Dense(hiddenDim, Activation.Tanh, inputSize: latentDim)
                .Dense(inputDim, Activation.Linear);
        }

        // ====================================================================
        // FORWARD PASS METHODS
        // ====================================================================

        /// <summary>
        /// EncodeThermoState input to latent distribution parameters.
        /// </summary>
        /// <param name="x">Input vector</param>
        /// <returns>Tuple of (μ, log_σ²)</returns>
        public (float[] equilibrium, float[] entropyPotential) EncodeThermoState(float[] x)
        {
            float[] equilibrium = equilibriumEncoder.Forward(x);
            float[] entropyPotential = entropyEncoder.Forward(x);
            return (equilibrium, entropyPotential);
        }

        /// <summary>
        /// SampleMicrostate from latent distribution using reparameterization trick.
        /// z = μ + σ * ε, where ε ~ N(0,1)
        /// </summary>
        public float[] SampleMicrostate(float[] equilibrium, float[] entropyPotential)
        {
            float[] z = new float[ManifoldDim];

            for (int i = 0; i < ManifoldDim; i++)
            {
                float microstateNoise = (float)GaussianRandom(random);  // ε ~ N(0,1)
                float thermalAgitation = (float)Math.Exp(0.5 * entropyPotential[i]); // σ = exp(0.5 * log_σ²)
                z[i] = equilibrium[i] + thermalAgitation * microstateNoise;                 // z = μ + σ * ε
            }

            return z;
        }

        /// <summary>
        /// SampleMicrostate with stored microstateNoise (for backprop).
        /// Returns both z and microstateNoise for gradient computation.
        /// </summary>
        private (float[] z, float[] microstateNoise) SampleMicrostateWithNoise(float[] equilibrium, float[] entropyPotential)
        {
            float[] z = new float[ManifoldDim];
            float[] microstateNoise = new float[ManifoldDim];

            for (int i = 0; i < ManifoldDim; i++)
            {
                microstateNoise[i] = (float)GaussianRandom(random);     // ε ~ N(0,1)
                float thermalAgitation = (float)Math.Exp(0.5 * entropyPotential[i]); // σ = exp(0.5 * log_σ²)
                z[i] = equilibrium[i] + thermalAgitation * microstateNoise[i];              // z = μ + σ * ε
            }

            return (z, microstateNoise);
        }

        /// <summary>
        /// RelaxToMicrostate latent vector to reconstruction.
        /// </summary>
        public float[] RelaxToMicrostate(float[] z)
        {
            return relaxationOperator.Forward(z);
        }

        /// <summary>
        /// Full forward pass: x → μ, log_σ² → z → relaxedMicrostate
        /// </summary>
        public float[] RelaxAndReconstruct(float[] x)
        {
            var (equilibrium, entropyPotential) = EncodeThermoState(x);
            float[] z = SampleMicrostate(equilibrium, entropyPotential);
            return RelaxToMicrostate(z);
        }

        // ====================================================================
        // LOSS COMPUTATION
        // ====================================================================

        /// <summary>
        /// Compute TVAE loss: Reconstruction + β * KL
        /// </summary>
        public float ComputeFreeEnergy(float[] x, float[] relaxedMicrostate, float[] equilibrium, float[] entropyPotential)
        {
            // Reconstruction loss (MSE)
            float energyMismatch = 0;
            for (int i = 0; i < MicrostateDim; i++)
            {
                float diff = x[i] - relaxedMicrostate[i];
                energyMismatch += diff * diff;
            }
            energyMismatch /= MicrostateDim;

            // KL divergence: -0.5 * Σ(1 + log_σ² - μ² - σ²)
            float entropyRegularizer = 0;
            for (int i = 0; i < ManifoldDim; i++)
            {
                entropyRegularizer += 1 + entropyPotential[i] - equilibrium[i] * equilibrium[i] - (float)Math.Exp(entropyPotential[i]);
            }
            entropyRegularizer *= -0.5f;

            return energyMismatch + Beta * entropyRegularizer;
        }

        // ====================================================================
        // BACKPROPAGATION (EDUCATIONAL IMPLEMENTATION)
        // ====================================================================

        /// <summary>
        /// Compute gradients and update parameters for one training sample.
        /// This is the heart of TVAE training - showing all gradient computations.
        /// </summary>
        private float RelaxSample(float[] x, float learningRate)
        {
            t++;  // Increment Adam timestep

            // ================================================================
            // FORWARD PASS (with saved intermediate values)
            // ================================================================

            // 1. EncodeThermoState: x → (μ, log_σ²)
            var (equilibrium, entropyPotential) = EncodeThermoState(x);

            // 2. SampleMicrostate: (μ, log_σ²) → z (save ε for backprop!)
            var (z, microstateNoise) = SampleMicrostateWithNoise(equilibrium, entropyPotential);

            // 3. RelaxToMicrostate: z → relaxedMicrostate
            float[] relaxedMicrostate = RelaxToMicrostate(z);

            // 4. Compute loss
            float loss = ComputeFreeEnergy(x, relaxedMicrostate, equilibrium, entropyPotential);

            // ================================================================
            // BACKWARD PASS - GRADIENT COMPUTATION
            // ================================================================

            // ────────────────────────────────────────────────────────────────
            // STEP 1: Compute ∂L_recon/∂relaxedMicrostate (reconstruction gradient)
            // ────────────────────────────────────────────────────────────────
            // L_recon = (1/D) * Σ(x - relaxedMicrostate)²
            // ∂L_recon/∂relaxedMicrostate = -2/D * (x - relaxedMicrostate)

            float[] grad_x_recon = new float[MicrostateDim];
            for (int i = 0; i < MicrostateDim; i++)
            {
                grad_x_recon[i] = -2.0f / MicrostateDim * (x[i] - relaxedMicrostate[i]);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 2: Backprop through relaxationOperator: ∂L_recon/∂z
            // ────────────────────────────────────────────────────────────────
            // Use relaxationOperator's backward pass to compute gradient w.r.t. z
            float[] grad_z = relaxationOperator.Layers.Last().Backward(grad_x_recon, learningRate, t);
            for (int i = relaxationOperator.Layers.Count - 2; i >= 0; i--)
            {
                grad_z = relaxationOperator.Layers[i].Backward(grad_z, learningRate, t);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 3: Compute KL divergence gradients
            // ────────────────────────────────────────────────────────────────
            // L_KL = -0.5 * Σ(1 + log_σ² - μ² - exp(log_σ²))
            //
            // ∂L_KL/∂μ = -0.5 * (-2μ) = μ
            // ∂L_KL/∂log_σ² = -0.5 * (1 - exp(log_σ²))
            //               = 0.5 * (exp(log_σ²) - 1)

            float[] grad_mu_kl = new float[ManifoldDim];
            float[] grad_logvar_kl = new float[ManifoldDim];

            for (int i = 0; i < ManifoldDim; i++)
            {
                grad_mu_kl[i] = Beta * equilibrium[i];

                // FIXED: Correct gradient sign
                float exp_logvar = (float)Math.Exp(entropyPotential[i]);
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
            // Chain rule from relaxationOperator:
            // ∂L_recon/∂μ = ∂L_recon/∂z * ∂z/∂μ = grad_z
            // ∂L_recon/∂log_σ² = ∂L_recon/∂z * ∂z/∂log_σ²

            float[] grad_mu_recon = new float[ManifoldDim];
            float[] grad_logvar_recon = new float[ManifoldDim];

            for (int i = 0; i < ManifoldDim; i++)
            {
                // Gradient w.r.t. μ (from reconstruction)
                grad_mu_recon[i] = grad_z[i];  // ∂z/∂μ = 1

                // Gradient w.r.t. log_σ² (from reconstruction)
                float thermalAgitation = (float)Math.Exp(0.5 * entropyPotential[i]);
                grad_logvar_recon[i] = grad_z[i] * 0.5f * thermalAgitation * microstateNoise[i];
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 5: Combine gradients (reconstruction + KL)
            // ────────────────────────────────────────────────────────────────

            float[] grad_mu = new float[ManifoldDim];
            float[] grad_logvar = new float[ManifoldDim];

            for (int i = 0; i < ManifoldDim; i++)
            {
                grad_mu[i] = grad_mu_recon[i] + grad_mu_kl[i];
                grad_logvar[i] = grad_logvar_recon[i] + grad_logvar_kl[i];
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 6: Backprop through equilibriumEncoder
            // ────────────────────────────────────────────────────────────────

            float[] grad_mu_input = grad_mu;
            for (int i = equilibriumEncoder.Layers.Count - 1; i >= 0; i--)
            {
                grad_mu_input = equilibriumEncoder.Layers[i].Backward(grad_mu_input, learningRate, t);
            }

            // ────────────────────────────────────────────────────────────────
            // STEP 7: Backprop through entropyEncoder
            // ────────────────────────────────────────────────────────────────

            float[] grad_logvar_input = grad_logvar;
            for (int i = entropyEncoder.Layers.Count - 1; i >= 0; i--)
            {
                grad_logvar_input = entropyEncoder.Layers[i].Backward(grad_logvar_input, learningRate, t);
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
        /// Train TVAE on dataset using full backpropagation.
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
            equilibriumEncoder.LearningRate = learningRate;
            entropyEncoder.LearningRate = learningRate;
            relaxationOperator.LearningRate = learningRate;

            var lossHistory = new List<float>();

            Console.WriteLine("Training Variational Autoencoder...");
            Console.WriteLine($"  Samples: {X.Length}");
            Console.WriteLine($"  Architecture: {MicrostateDim} → ({ManifoldDim} μ, {ManifoldDim} log_σ²) → {MicrostateDim}");
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
                    float sample_loss = RelaxSample(X[idx], learningRate);
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
                var (equilibrium, entropyPotential) = EncodeThermoState(X[i]);
                Z[i] = equilibrium;  // Use mean, not sample (deterministic)
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
                var relaxedMicrostate = RelaxAndReconstruct(x);

                for (int i = 0; i < MicrostateDim; i++)
                {
                    float diff = x[i] - relaxedMicrostate[i];
                    total_error += diff * diff;
                }
            }

            return total_error / (X.Length * MicrostateDim);
        }

        /// <summary>
        /// Extract learned uncertainty (σ) for each sample.
        /// </summary>
        public float[][] ExtractSigma(float[][] X)
        {
            var Sigma = new float[X.Length][];

            for (int i = 0; i < X.Length; i++)
            {
                var (equilibrium, entropyPotential) = EncodeThermoState(X[i]);
                Sigma[i] = new float[ManifoldDim];

                for (int j = 0; j < ManifoldDim; j++)
                {
                    Sigma[i][j] = (float)Math.Sqrt(Math.Exp(entropyPotential[j]));  // σ = sqrt(exp(log_σ²))
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
            Console.WriteLine($"\nInput dimension:   {MicrostateDim}");
            Console.WriteLine($"Latent dimension:  {ManifoldDim}");
            Console.WriteLine($"Beta (KL weight):  {Beta}");
            Console.WriteLine("\nEncoder μ:");
            foreach (var layer in equilibriumEncoder.Layers)
            {
                Console.WriteLine($"  {layer.InputSize} → {layer.OutputSize} ({layer.Act})");
            }
            Console.WriteLine("\nEncoder log_σ²:");
            foreach (var layer in entropyEncoder.Layers)
            {
                Console.WriteLine($"  {layer.InputSize} → {layer.OutputSize} ({layer.Act})");
            }
            Console.WriteLine("\nDecoder:");
            foreach (var layer in relaxationOperator.Layers)
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
                var (equilibrium, entropyPotential) = EncodeThermoState(x);
                var relaxedMicrostate = RelaxToMicrostate(equilibrium);  // Use μ, not sampling

                for (int i = 0; i < MicrostateDim; i++)
                {
                    float residual = x[i] - relaxedMicrostate[i];
                    total_variance += residual * residual;
                }
            }

            float mse = total_variance / (X.Length * MicrostateDim);
            return (float)Math.Sqrt(mse);  // RMSE ≈ σ_input estimate
        }


        // Free energy already exists inside TVAE: ComputeFreeEnergy
        // Here we expose it as an operator on a batch.

        public static float AverageFreeEnergy(TVAE model, float[][] microstates)
        {
            float total = 0;

            foreach (var x in microstates)
            {
                var (eq, ent) = model.EncodeThermoState(x);
                var z = model.SampleMicrostate(eq, ent);
                var relaxed = model.RelaxToMicrostate(z);

                total += model.ComputeFreeEnergy(x, relaxed, eq, ent);
            }

            return total / microstates.Length;
        }

        // Simple “thermodynamic force” estimate:
        // F ≈ −∂FreeEnergy/∂equilibrium  ≈ −equilibrium (from KL term)
        public static float[] ThermodynamicForce(float[] equilibrium)
        {
            var F = new float[equilibrium.Length];
            for (int i = 0; i < equilibrium.Length; i++)
                F[i] = -equilibrium[i];
            return F;
        }
    }

    public static class ThermoOps
    {
        // Free energy already exists inside TVAE: ComputeFreeEnergy
        // Here we expose it as an operator on a batch.

        public static float AverageFreeEnergy(TVAE model, float[][] microstates)
        {
            float total = 0;

            foreach (var x in microstates)
            {
                var (eq, ent) = model.EncodeThermoState(x);
                var z = model.SampleMicrostate(eq, ent);
                var relaxed = model.RelaxToMicrostate(z);

                total += model.ComputeFreeEnergy(x, relaxed, eq, ent);
            }

            return total / microstates.Length;
        }

        // Simple “thermodynamic force” estimate:
        // F ≈ −∂FreeEnergy/∂equilibrium  ≈ −equilibrium (from KL term)
        public static float[] ThermodynamicForce(float[] equilibrium)
        {
            var F = new float[equilibrium.Length];
            for (int i = 0; i < equilibrium.Length; i++)
                F[i] = -equilibrium[i];
            return F;
        }

        // Simple entropy flow proxy: use norm of thermodynamic force
        public static float EntropyFlow(float[] force)
        {
            float sum = 0;
            for (int i = 0; i < force.Length; i++)
                sum += force[i] * force[i];
            return MathF.Sqrt(sum);
        }

        // One relaxation step in microstate space: x_{t+1} = x_t + η F
        public static float[] RelaxMicrostate(float[] x, float[] force, float eta)
        {
            var next = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
                next[i] = x[i] + eta * force[i];
            return next;
        }

        // One relaxation step in latent space: z_{t+1} = z_t + η F(μ)
        public static float[] RelaxLatent(float[] z, float[] equilibrium, float eta)
        {
            var next = new float[z.Length];
            var F = ThermodynamicForce(equilibrium);

            for (int i = 0; i < z.Length; i++)
                next[i] = z[i] + eta * F[i];

            return next;
        }

    }

}

