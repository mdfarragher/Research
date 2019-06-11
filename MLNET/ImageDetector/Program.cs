using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageDetector
{
    /// <summary>
    /// The application class
    /// </summary>
    class Program
    {
        /// <summary>
        /// A data class that hold one image data record
        /// </summary>
        public class ImageNetData
        {
            [LoadColumn(0)] public string ImagePath;
            [LoadColumn(1)] public string Label;

            /// <summary>
            /// Load the contents of a TSV file as an object sequence representing images and labels
            /// </summary>
            /// <param name="file">The name of the TSV file</param>
            /// <returns>A sequence of objects representing the contents of the file</returns>
            public static IEnumerable<ImageNetData> ReadFromCsv(string file)
            {
                return File.ReadAllLines(file)
                    .Select(x => x.Split('\t'))
                    .Select(x => new ImageNetData 
                    { 
                        ImagePath = x[0], 
                        Label = x[1] 
                    });
            }
        }

        /// <summary>
        /// A prediction class that holds only a model prediction.
        /// </summary>
        public class ImageNetPrediction
        {
            [ColumnName("softmax2")]
            public float[] PredictedLabels;
        }

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args">The command line arguments></param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var mlContext = new MLContext();

            // load the TSV file with image names and corresponding labels
            var data = mlContext.Data.LoadFromTextFile<ImageNetData>("images/tags.tsv", hasHeader: true);

            // set up a learning pipeline
            var pipeline = mlContext.Transforms
            
                // step 1: load the images
                .LoadImages(
                    outputColumnName: "input", 
                    imageFolder: "images", 
                    inputColumnName: nameof(ImageNetData.ImagePath))

                // step 2: resize the images to 224x224
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "input", 
                    imageWidth: 224, 
                    imageHeight: 224, 
                    inputColumnName: "input"))

                // step 3: extract pixels in a format the TF model can understand
                // these interleave and offset values are identical to the images the model was trained on
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input", 
                    interleavePixelColors: true, 
                    offsetImage: 117))

                // step 4: load the TensorFlow model
                .Append(mlContext.Model.LoadTensorFlowModel("models/tensorflow_inception_graph.pb")

                // step 5: score the images using the TF model
                .ScoreTensorFlowModel(
                    outputColumnNames: new[] { "softmax2" },
                    inputColumnNames: new[] { "input" }, 
                    addBatchDimensionInput:true));
                        
            // train the model on the data file
            Console.WriteLine("Start training model....");
            var model = pipeline.Fit(data);
            Console.WriteLine("Model training complete!");

            // create a prediction engine
            var engine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

            // load all imagenet labels
            var labels = File.ReadAllLines("models/imagenet_comp_graph_label_strings.txt");

            // predict what is in each image
            Console.WriteLine("Predicting image contents....");
            var images = ImageNetData.ReadFromCsv("images/tags.tsv");
            foreach (var image in images)
            {
                Console.Write($"  [{image.ImagePath}]: ");
                var prediction = engine.Predict(image).PredictedLabels;

                // find the best prediction
                var i = 0;
                var best = (from p in prediction 
                            select new { Index = i++, Prediction = p }).OrderByDescending(p => p.Prediction).First();
                var predictedLabel = labels[best.Index];

                // show the corresponding label
                Console.WriteLine($"{predictedLabel} {(predictedLabel != image.Label ? "***WRONG***" : "")}");
            }
        }
    }
}
