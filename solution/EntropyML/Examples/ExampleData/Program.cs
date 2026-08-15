using EntropyML;

namespace ExampleData
{
    internal class Program
    {
        static void Data01()
        {
            Console.WriteLine("\n** ExampleData.Data01() **");
            var data =      DataGen.Microstates(10, 3);
            ListStat(data);
        }

        static void Data02()
        {
            Console.WriteLine("\n** ExampleData.Data01() **");
            var data = DataGen.Microstates(1000, 3);
            ListStat(data);
        }

        static void Data03()
        {
            Console.WriteLine("\n** ExampleData.Data03() **");
            var data = DataGen.Microstates(10000, 10);
            ListStat(data);
        }

        static void ListStat(float[][] data)
        {
            Console.WriteLine($"Samples : {data.Length}");
            Console.WriteLine($"Dimensions : {data[0].Length}");
            int dim = data[0].Length;
            Console.WriteLine($"{"",4}  {"Min",8}  {"Max",8}  {"Mean",8}  {"Std",8}");
            for (int d = 0; d < dim; d++)
            {
                var col = data.Select(x => x[d]).ToArray();
                float min = col.Min();
                float max = col.Max();
                float mean = col.Average();
                float std = MathF.Sqrt(col.Average(v => (v - mean) * (v - mean)));
                Console.WriteLine($"Dim {d}  {min,8:F4}  {max,8:F4}  {mean,8:F4}  {std,8:F4}");
            }
        }
        static void Main(string[] args)
        {
            Data01();
            Data02();
            Data03();
        }
    }
}
