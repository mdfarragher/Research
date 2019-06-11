using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using PLplot;

namespace Spike
{
    /// <summary>
    /// The MeterData class contains one power consumption record.
    /// </summary>
    public class MeterData
    {
        [LoadColumn(0)] public string Name { get; set; }
        [LoadColumn(1)] public DateTime Time { get; set; }
        [LoadColumn(2)] public float Consumption { get; set; }
    }

    /// <summary>
    /// The SpikePrediction class contains one power spike prediction.
    /// </summary>
    public class SpikePrediction
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    public class Program
    {
        // filename for data set
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "power-export.csv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line parameters.</param>
        static void Main()
        {
            // create the machine learning context
            var context = new MLContext();

            // load the data file
            Console.WriteLine("Loading data...");
            var dataView = context.Data.LoadFromTextFile<MeterData>(path: dataPath, hasHeader: true, separatorChar: ',');

            // get an array of data points
            var values = context.Data.CreateEnumerable<MeterData>(dataView, reuseRowObject: false).ToArray();

            // plot the data
            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam("data.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                0, 90,                          // x-axis range
                0, 5000,                        // y-axis range
                AxesScale.Independent,          // scale x and y independently
                AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "Day",                          // x-axis label
                "Power consumption",            // y-axis label
                "Power consumption over time"); // plot title
            pl.line(
                (from x in Enumerable.Range(0, values.Count()) select (double)x).ToArray(),
                (from p in values select (double)p.Consumption).ToArray()
            );

            // build a training pipeline for detecting spikes
            var pipeline = context.Transforms.SsaSpikeEstimator(
                nameof(SpikePrediction.Prediction), 
                nameof(MeterData.Consumption), 
                confidence: 98, 
                pvalueHistoryLength: 30, 
                trainingWindowSize: 90, 
                seasonalityWindowSize: 30);

            // train the model
            Console.WriteLine("Detecting spikes...");
            var model = pipeline.Fit(dataView);

            // predict spikes in the data
            var transformed = model.Transform(dataView);
            var predictions = context.Data.CreateEnumerable<SpikePrediction>(transformed, reuseRowObject: false).ToArray();

            // find the spikes in the data
            var spikes = (from i in Enumerable.Range(0, predictions.Count()) 
                          where predictions[i].Prediction[0] == 1
                          select (Day: i, Consumption: values[i].Consumption));

            // plot the spikes
            pl.col0(2);     // blue color
            pl.schr(3, 3);  // scale characters
            pl.string2(
                (from s in spikes select (double)s.Day).ToArray(),
                (from s in spikes select (double)s.Consumption + 200).ToArray(),
                "↓");

            pl.eop();
        }
    }
}