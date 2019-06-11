using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

#pragma warning disable 649

namespace Mnist
{
    /// <summary>
    /// The Digit class represents one mnist digit.
    /// </summary>
    class Digit
    {
        [VectorType(785)] public float[] PixelValues;
    }

    /// <summary>
    /// The DigitPrediction class represents one digit prediction.
    /// </summary>
    class DigitPrediction
    {
        public float[] Score;
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filename for data set
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "handwritten_digits_large.csv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var context = new MLContext();

            // load data
            Console.WriteLine("Loading data....");
            var dataView = context.Data.LoadFromTextFile(
                path: dataPath,
                columns : new[] 
                {
                    new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                    new TextLoader.Column("Number", DataKind.Single, 0)
                },
                hasHeader : false,
                separatorChar : ',');

            // split data into a training and test set
            var partitions = context.MulticlassClassification.TrainTestSplit(dataView, testFraction: 0.2);

            // build a training pipeline
            // step 1: concatenate all feature columns
            var pipeline = context.Transforms.Concatenate(
                DefaultColumnNames.Features, 
                nameof(Digit.PixelValues))

                // step 2: cache data to speed up training                
                .AppendCacheCheckpoint(context)

                // step 3: train the model with SDCA
                .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                    labelColumnName: "Number", 
                    featureColumnName: DefaultColumnNames.Features));

            // train the model
            Console.WriteLine("Training model....");
            var model = pipeline.Fit(partitions.TrainSet);

            // use the model to make predictions on the test data
            Console.WriteLine("Evaluating model....");
            var predictions = model.Transform(partitions.TestSet);

            // evaluate the predictions
            var metrics = context.MulticlassClassification.Evaluate(
                data: predictions, 
                label: "Number", 
                score: DefaultColumnNames.Score);

            // show evaluation metrics
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.AccuracyMicro:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.AccuracyMacro:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();

            // grab three digits from the data: 2, 7, and 9
            var digits = context.Data.CreateEnumerable<Digit>(dataView, reuseRowObject: false).ToArray();
            var testDigits = new Digit[] { digits[5], digits[12], digits[20] };

            // create a prediction engine
            var engine = model.CreatePredictionEngine<Digit, DigitPrediction>(context);

            // predict each test digit
            for (var i=0; i < testDigits.Length; i++)
            {
                var prediction = engine.Predict(testDigits[i]);

                // show results
                Console.WriteLine($"Predicting test digit {i}...");
                for (var j=0; j < 10; j++)
                {
                    Console.WriteLine($"  {j}: {prediction.Score[j]:P2}");
                }
                Console.WriteLine();
            }
            Console.ReadKey();
        }
    }
}
