using System;
using System.IO;
using System.Net;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace Sentiment
{
    /// <summary>
    /// The SentimentIssue class represents a single sentiment record.
    /// </summary>
    public class SentimentIssue
    {
        [LoadColumn(0)] public bool Label { get; set; }
        [LoadColumn(2)] public string Text { get; set; }
    }

    /// <summary>
    /// The SentimentPrediction class represents a single sentiment prediction.
    /// </summary>
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    static class Program
    {
        // filenames for data set
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "wikiDetoxAnnotated40kRows.tsv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var mlContext = new MLContext();

            // load the data file
            Console.WriteLine("Loading data...");
            var data = mlContext.Data.LoadFromTextFile<SentimentIssue>(dataPath, hasHeader: true);

            // split the data into 80% training and 20% testing partitions
            var partitions = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            // build a machine learning pipeline
            // step 1: featurize the text
            var pipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: DefaultColumnNames.Features, 
                inputColumnName: nameof(SentimentIssue.Text))

                // step 2: add a fast tree learner
                .Append(mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: DefaultColumnNames.Label, 
                    featureColumnName: DefaultColumnNames.Features));

            // train the model
            Console.WriteLine("Training model...");
            var model = pipeline.Fit(partitions.TrainSet);

            // validate the model
            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(
                data:predictions, 
                label: DefaultColumnNames.Label, 
                score: DefaultColumnNames.Score);

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"  Auc:               {metrics.Auc:P2}");
            Console.WriteLine($"  Auprc:             {metrics.Auprc:P2}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.##}");
            Console.WriteLine();

            // create a prediction engine to make a single prediction
            Console.WriteLine("Making a prediction...");
            var issue = new SentimentIssue { Text = "With all due respect, you are a moron" };
            var engine = model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(mlContext);

            // make a single prediction
            var prediction = engine.Predict(issue);

            // report results
            Console.WriteLine($"  Text:        {issue.Text}");
            Console.WriteLine($"  Prediction:  {prediction.Prediction}");
            Console.WriteLine($"  Probability: {prediction.Probability:P2}");
            Console.WriteLine($"  Score:       {prediction.Score}");

            Console.ReadKey();
        }
    }
}