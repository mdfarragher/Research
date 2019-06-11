using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using Microsoft.Data.DataView;

namespace Recommender
{
    /// <summary>
    /// The MovieRating class holds a single movie rating.
    /// </summary>
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;

        [LoadColumn(1)]
        public float movieId;

        [LoadColumn(2)]
        public float Label;
    }

    /// <summary>
    /// The MovieRatingPrediction class holds a single movie prediction.
    /// </summary>
    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filenames for training and test data
        private static string trainingDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-ratings-train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-ratings-test.csv");

        /// <summary>
        /// The program entry point.
        /// </summary>
        /// <param name="args">The command line arguments</param>
        static void Main(string[] args)
        {
            // set up a new machine learning context
            var mlContext = new MLContext();

            // load training and test data
            var trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            var testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            // prepare matrix factorization options
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded", 
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            // set up a training pipeline
            // step 1: map userId and movieId to keys
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "userId",
                    outputColumnName: "userIdEncoded")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "movieId",
                    outputColumnName: "movieIdEncoded")

                // step 2: find recommendations using matrix factorization
                .Append(mlContext.Recommendation().Trainers.MatrixFactorization(options)));

            // train the model
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainingDataView);  
            Console.WriteLine();

            // evaluate the model performance 
            Console.WriteLine("Evaluating the model...");
            var predictions = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, label: DefaultColumnNames.Label, score: DefaultColumnNames.Score);
            Console.WriteLine($"  RMSE: {metrics.Rms:#.##}");
            Console.WriteLine($"  L1:   {metrics.L1:#.##}");
            Console.WriteLine($"  L2:   {metrics.L2:#.##}");
            Console.WriteLine();

            // check if a given user likes 'GoldenEye'
            Console.WriteLine("Calculating the score for user 6 liking the movie 'GoldenEye'...");
            var predictionEngine = model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(mlContext);
            var prediction = predictionEngine.Predict(
                new MovieRating()
                {
                    userId = 6,
                    movieId = 10  // GoldenEye
                }
            );
            Console.WriteLine($"  Score: {prediction.Score}");
            Console.WriteLine();

            // find the top 5 movies for a given user
            Console.WriteLine("Calculating the top 5 movies for user 6...");
            var top5 =  (from m in Movies.All
                         let p = predictionEngine.Predict(
                            new MovieRating()
                            {
                                userId = 6,
                                movieId = m.ID
                            })
                         orderby p.Score descending
                         select (MovieId: m.ID, Score: p.Score)).Take(5);
            foreach (var t in top5)
                Console.WriteLine($"  Score:{t.Score}\tMovie: {Movies.Get(t.MovieId)?.Title}");

            Console.ReadLine();
        }
    }
}
