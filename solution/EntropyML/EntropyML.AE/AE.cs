
using EntropyML;

namespace ExampleData
{
    public class AutoEncoder
    {
        public int InputDim { get; }
        public int HiddenDim { get; }
        public int BottleneckDim { get; }

        private Sequential Encoder;
        private Sequential Decoder;
        private int _t = 1;

        public AutoEncoder(int inputDim, int hiddenDim, int bottleneckDim)
        {
            InputDim = inputDim;
            HiddenDim = hiddenDim;
            BottleneckDim = bottleneckDim;

            Build();
        }

        private void Build()
        {
            Encoder = new Sequential();
            Encoder.Dense(HiddenDim, Activation.Tanh, inputSize: InputDim);
            Encoder.Dense(BottleneckDim, Activation.Tanh);

            Decoder = new Sequential();
            Decoder.Dense(HiddenDim, Activation.Tanh, inputSize: BottleneckDim);
            Decoder.Dense(InputDim, Activation.Linear);
        }

        public float[] Encode(float[] x)
        {
            return Encoder.Forward(x);
        }

        public float[] Decode(float[] z)
        {
            return Decoder.Forward(z);
        }

        public float[] Forward(float[] x)
        {
            return Decode(Encode(x));
        }

        public float Fit(float[][] X, float lr = 0.001f, int epochs = 200, int batchSize = 32)
        {
            float loss = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var x in X)
                {
                    var y = Forward(x);
                    loss += MSE(y, x);

                    var grad = MSEGrad(y, x);
                    grad = Decoder.Backward(grad, lr, ref _t);
                    Encoder.Backward(grad, lr, ref _t);
                }
            }

            return loss / X.Length;
        }

        private float MSE(float[] y, float[] x)
        {
            float s = 0;
            for (int i = 0; i < x.Length; i++)
            {
                float d = y[i] - x[i];
                s += d * d;
            }
            return s / x.Length;
        }

        private float[] MSEGrad(float[] y, float[] x)
        {
            var grad = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
                grad[i] = 2f * (y[i] - x[i]) / x.Length;
            return grad;
        }
    }
}
