using System.Data;
using EntropyML;

namespace ExampleVAE
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ExampleVAE.TrainVAE();
            ExampleVAEv2.TrainVAE();
        }
    }
}
