namespace ExampleTVAE
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ExampleTVAE.TrainTVAE();
            ExampleTVAE.TestTVAE();
            Guardrails.Run();
        }
    }
}
