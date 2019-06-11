using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;

// CS0649 compiler warning is disabled because some fields are only 
// assigned to dynamically by ML.NET at runtime
#pragma warning disable CS0649

namespace MyApp
{
    /// <summary>
    /// The application class.
    /// </summary>
    class Program
    {
        /// <summary>
        /// A data transfer class that holds a single iris flower.
        /// </summary>
        public class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        /// <summary>
        /// A prediction class that holds a single model prediction.
        /// </summary>
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args"The command line arguments></param>
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // read the iris flower data from a text file
            var trainingData = mlContext.Data.ReadFromTextFile<IrisData>(
                path: "iris-data.txt", 
                hasHeader: false, 
                separatorChar: ',');

            // set up a learning pipeline
            // step 1: convert the label text to numeric key
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")

                // step 2: concatenate input features into a single column
                .Append(mlContext.Transforms.Concatenate(
                    "Features", 
                    "SepalLength", 
                    "SepalWidth", 
                    "PetalLength", 
                    "PetalWidth"))

                // step 3: cache the training data to improve performance
                .AppendCacheCheckpoint(mlContext)

                // step 4: use the stochastic dual coordinate ascent learning algorithm
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                    labelColumn: "Label", 
                    featureColumn: "Features"))

                // step 5: convert the label key back to a string value
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // train the model on the data file
            Console.WriteLine("Start training model....");
            var model = pipeline.Fit(trainingData);
            Console.WriteLine("Model training complete!");

            // predict a single flower based on input data
            Console.WriteLine("Predicting a sample flower....");
            var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");

            Console.WriteLine("Press any key to exit....");
            Console.ReadLine();
        }
    }
}