namespace EntropyML
{
    public enum Activation
    {
        Tanh,
        Linear
    }

    public static class Activations
    {
        public static float[] Forward(float[] x, Activation act)
        {
            return act switch
            {
                Activation.Tanh => x.Select(MathF.Tanh).ToArray(),
                Activation.Linear => x.ToArray(),
                _ => throw new NotImplementedException()
            };
        }

        public static float[] Backward(float[] activated, float[] grad, Activation act)
        {
            return act switch
            {
                Activation.Tanh => grad.Zip(activated, (g, v) => g * (1 - v * v)).ToArray(),
                Activation.Linear => grad.ToArray(),
                _ => throw new NotImplementedException()
            };
        }
    }

    public class DenseLayer
    {
        /// <summary>
        /// Gets or sets a value indicating whether to use parallel processing for the Adam optimizer update.
        /// When <c>true</c>, the update loop over output neurons is parallelized.
        /// Default is <c>false</c>.
        /// </summary>
        public bool UseParallelism { get; set; } = false;

        /// <summary>Gets the number of input features for this layer.</summary>
        public int InputSize { get; private set; }

        /// <summary>Gets the number of output units (neurons) in this layer.</summary>
        public int OutputSize { get; private set; }

        /// <summary>The weight matrix of the layer, with dimensions [InputSize, OutputSize].</summary>
        public float[,] W { get; set; }

        /// <summary>The bias vector of the layer, with dimensions [OutputSize].</summary>
        public float[] B { get; set; }

        /// <summary>The activation function used by the layer.</summary>
        public Activation Act { get; private set; }

        /// <summary>The cached input from the last forward pass, used during backpropagation.</summary>
        public float[] LastInput { get; private set; }

        /// <summary>The cached output from the last forward pass (after activation), used during backpropagation.</summary>
        public float[] LastOutput { get; private set; }

        #region Adam Optimizer State
        /// <summary>Adam optimizer's first moment (moving average of the gradients) for weights.</summary>
        public float[,] mW { get; set; }

        /// <summary>Adam optimizer's second moment (moving average of the squared gradients) for weights.</summary>
        public float[,] vW { get; set; }

        /// <summary>Adam optimizer's first moment for biases.</summary>
        public float[] mB { get; set; }

        /// <summary>Adam optimizer's second moment for biases.</summary>
        public float[] vB { get; set; }
        #endregion

        /// <summary>
        /// Initializes a new instance of the <see cref="DenseLayer_"/> class with Xavier weight initialization.
        /// </summary>
        /// <param name="input">The number of input features.</param>
        /// <param name="output">The number of output units (neurons).</param>
        /// <param name="act">The activation function to use for this layer.</param>
        public DenseLayer(int input, int output, Activation act)
        {
            InputSize = input;
            OutputSize = output;
            Act = act;

            var rand = new Random();
            W = new float[input, output];
            B = new float[output];

            float scale = MathF.Sqrt(2f / (input + output));
            for (int i = 0; i < input; i++)
                for (int j = 0; j < output; j++)
                    W[i, j] = (float)(rand.NextDouble() * 2 - 1) * scale;

            // Initialize Adam optimizer state arrays
            mW = new float[input, output];
            vW = new float[input, output];
            mB = new float[output];
            vB = new float[output];
        }

        public float[] Forward(float[] x)
        {
            LastInput = x;

            // Compute z = W^T * x + b
            float[] z = new float[OutputSize];
            for (int j = 0; j < OutputSize; j++)
            {
                float sum = B[j];
                for (int i = 0; i < InputSize; i++)
                    sum += x[i] * W[i, j];
                z[j] = sum;
            }

            // Apply activation function and cache the result
            LastOutput = Activations.Forward(z, Act);
            return LastOutput;
        }

        public float[] Backward(float[] gradOutput, float lr, int t)
        {
            // 1. Compute gradient of the loss with respect to the pre-activation output (dL/dz).
            // This is done by chaining the incoming gradient with the derivative of the activation function.
            float[] gradAct = Activations.Backward(LastOutput, gradOutput, Act);

            float[] gradInput = new float[InputSize];

            // 2. Compute gradient with respect to the layer's input (dL/dx).
            // This will be passed to the previous layer. dL/dx = W * dL/dz
            for (int j = 0; j < OutputSize; j++)
                for (int i = 0; i < InputSize; i++)
                    gradInput[i] += gradAct[j] * W[i, j];

            // 3. Update weights and biases using the Adam optimizer.
            AdamUpdate(gradAct, lr, t);
            return gradInput;
        }

        private void AdamUpdate(float[] gradAct, float lr, int t)
        {
            const float beta1 = 0.9f;   // Decay rate for the first moment (mean)
            const float beta2 = 0.999f; // Decay rate for the second moment (uncentered variance)
            const float eps = 1e-8f;    // Small constant for numerical stability

            if (UseParallelism)
            {
                // This loop is safe to parallelize because each iteration 'j' operates on a unique
                // slice of the weight and bias arrays (W[*, j], B[j], etc.), so there are no write conflicts between threads.
                Parallel.For(0, OutputSize, j =>
                {
                    // Update bias for neuron j
                    float mB_j = beta1 * mB[j] + (1 - beta1) * gradAct[j];
                    float vB_j = beta2 * vB[j] + (1 - beta2) * gradAct[j] * gradAct[j];
                    float mHatB = mB_j / (1 - MathF.Pow(beta1, t));
                    float vHatB = vB_j / (1 - MathF.Pow(beta2, t));
                    B[j] -= lr * mHatB / (MathF.Sqrt(vHatB) + eps);
                    mB[j] = mB_j;
                    vB[j] = vB_j;

                    // Update all weights connected to neuron j
                    for (int i = 0; i < InputSize; i++)
                    {
                        float g = gradAct[j] * LastInput[i]; // Gradient of weight W[i,j]
                        float mW_ij = beta1 * mW[i, j] + (1 - beta1) * g;
                        float vW_ij = beta2 * vW[i, j] + (1 - beta2) * g * g;
                        float mHat = mW_ij / (1 - MathF.Pow(beta1, t));
                        float vHat = vW_ij / (1 - MathF.Pow(beta2, t));
                        W[i, j] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
                        mW[i, j] = mW_ij;
                        vW[i, j] = vW_ij;
                    }
                });
            }
            else
            {
                // Original sequential implementation
                for (int j = 0; j < OutputSize; j++)
                {
                    // Update bias for neuron j
                    mB[j] = beta1 * mB[j] + (1 - beta1) * gradAct[j];
                    vB[j] = beta2 * vB[j] + (1 - beta2) * gradAct[j] * gradAct[j];

                    float mHatB = mB[j] / (1 - MathF.Pow(beta1, t));
                    float vHatB = vB[j] / (1 - MathF.Pow(beta2, t));

                    B[j] -= lr * mHatB / (MathF.Sqrt(vHatB) + eps);

                    // Update all weights connected to neuron j
                    for (int i = 0; i < InputSize; i++)
                    {
                        float g = gradAct[j] * LastInput[i]; // Gradient of weight W[i,j]

                        mW[i, j] = beta1 * mW[i, j] + (1 - beta1) * g;
                        vW[i, j] = beta2 * vW[i, j] + (1 - beta2) * g * g;

                        float mHat = mW[i, j] / (1 - MathF.Pow(beta1, t));
                        float vHat = vW[i, j] / (1 - MathF.Pow(beta2, t));

                        W[i, j] -= lr * mHat / (MathF.Sqrt(vHat) + eps);
                    }
                }
            }
        }

        public void Save(BinaryWriter writer)
        {
            writer.Write(InputSize);
            writer.Write(OutputSize);
            writer.Write((int)Act);

            for (int i = 0; i < InputSize; i++)
                for (int j = 0; j < OutputSize; j++)
                    writer.Write(W[i, j]);

            for (int j = 0; j < OutputSize; j++)
                writer.Write(B[j]);

            for (int i = 0; i < InputSize; i++)
                for (int j = 0; j < OutputSize; j++)
                {
                    writer.Write(mW[i, j]);
                    writer.Write(vW[i, j]);
                }

            for (int j = 0; j < OutputSize; j++)
            {
                writer.Write(mB[j]);
                writer.Write(vB[j]);
            }
        }

        public static DenseLayer Load(BinaryReader reader)
        {
            int inputSize = reader.ReadInt32();
            int outputSize = reader.ReadInt32();
            Activation act = (Activation)reader.ReadInt32();

            var layer = new DenseLayer(inputSize, outputSize, act);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    layer.W[i, j] = reader.ReadSingle();

            for (int j = 0; j < outputSize; j++)
                layer.B[j] = reader.ReadSingle();

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                {
                    layer.mW[i, j] = reader.ReadSingle();
                    layer.vW[i, j] = reader.ReadSingle();
                }

            for (int j = 0; j < outputSize; j++)
            {
                layer.mB[j] = reader.ReadSingle();
                layer.vB[j] = reader.ReadSingle();
            }

            return layer;
        }
    }

    public class Sequential
    {
        public List<DenseLayer> Layers { get; } = new();

        public float LearningRate { get; set; } = 0.001f;


        public Sequential Dense(int units, Activation act, int? inputSize = null)
        {
            if (inputSize.HasValue)
                Layers.Add(new DenseLayer(inputSize.Value, units, act));
            else if (Layers.Any())
                Layers.Add(new DenseLayer(Layers.Last().OutputSize, units, act));
            else
                throw new InvalidOperationException("inputSize must be specified for the first layer.");

            return this;
        }

        public float[] Forward(float[] x)
        {
            float[] currentOutput = x;
            foreach (var layer in Layers)
                currentOutput = layer.Forward(currentOutput);
            return currentOutput;
        }

        public float[] Backward(float[] grad, float lr, ref int t)
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
                grad = Layers[i].Backward(grad, lr, t);

            t++;
            return grad;
        }


        public float TrainBatch(float[][] X, float[][] Y, ref int t)
        {
            float totalMse = 0;

            for (int n = 0; n < X.Length; n++)
            {
                // 1. Forward pass to get the prediction.
                float[] pred = Forward(X[n]);

                // 2. Compute MSE loss for the sample and the initial gradient of the loss with respect to the prediction.
                float[] grad = new float[pred.Length];
                float sampleMse = 0;
                for (int i = 0; i < pred.Length; i++)
                {
                    float diff = pred[i] - Y[n][i];
                    grad[i] = 2 * diff;  // Gradient of MSE: d(diff^2)/dy_pred = 2*diff
                    sampleMse += diff * diff;
                }
                totalMse += sampleMse;

                // 3. Backward pass, propagating the gradient back through all layers.
                for (int i = Layers.Count - 1; i >= 0; i--)
                    grad = Layers[i].Backward(grad, LearningRate, t);

                t++;  // Increment Adam timestep for each sample processed.
            }

            return totalMse / X.Length;
        }
    }
}
