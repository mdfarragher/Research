using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms.Normalizers;

namespace GitHub
{
    /// <summary>
    /// The GitHubIssue class represents one single GitHub issue.
    /// </summary>
    public class GitHubIssue
    {
        [LoadColumn(0)]
        public string ID { get; set; }

        [LoadColumn(1)]
        public string Area { get; set; }

        [LoadColumn(2)]
        public string Title { get; set; }
        
        [LoadColumn(3)]
        public string Description { get; set; }
    }

    /// <summary>
    /// The IssuePrediction class represents one single prediction.
    /// </summary>
    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // set up paths to data files
        private static string trainDataPath => Path.Combine(Directory.GetCurrentDirectory(), "issues_train.tsv");
        private static string testDataPath => Path.Combine(Directory.GetCurrentDirectory(),  "issues_test.tsv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments</param>
        static void Main(string[] args)
        {
            // set up a machine learning context
            var context = new MLContext(seed: 0);

            // load training data
            Console.Write("Loading training data....");
            var dataLoader = context.Data.CreateTextLoader<GitHubIssue>(hasHeader: true);
            var trainingData = dataLoader.Read(trainDataPath);
            Console.WriteLine("done");

            // set up a learning pipeline.
            // start by converting the 'area' label to a numeric value
            var pipeline = context.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Area", 
                outputColumnName: "Label")

                // now 'featurize' both text columns (meaning: convert to a numeric feature)
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))

                // concatenate both featurized columns into a single feature for training
                .Append(context.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))

                // cache training data to speed up learning
                .AppendCacheCheckpoint(context)

                // use the stochastic dual coordinate ascent learner
                .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                    "Label", 
                    "Features"))

                // convert learned labels back to their original text form
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // train the model
            Console.Write("Training model....");
            var model = pipeline.Fit(trainingData);
            Console.WriteLine("done");

            // load the test data
            Console.Write("Loading test data....");
            var testData = dataLoader.Read(testDataPath);
            Console.WriteLine("done");

            // use the model to make predictions on the test data
            Console.Write("Evaluating model....");
            var predictions = model.Transform(testData);

            // evaluate the predictions
            var metrics = context.MulticlassClassification.Evaluate(predictions);
            Console.WriteLine("done");

            // show evaluation metrics
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.AccuracyMicro:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.AccuracyMacro:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();

            // set up a single issue to test the model
            GitHubIssue issue = new GitHubIssue() {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            // predict the area of the test issue
            var predictor = model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(context);
            var prediction = predictor.Predict(issue);

            // show the result
            Console.WriteLine($"Single Prediction:");
            Console.WriteLine($"    Result: {prediction.Area}");
        }
    }
}
