using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Heart
{
    /// <summary>
    /// The HeartData record holds one single heart data record.
    /// </summary>
    public class HeartData 
    {
        [LoadColumn(0)]
        public float Age { get; set; }

        [LoadColumn(1)]
        public float Sex { get; set; }

        [LoadColumn(2)]
        public float Cp { get; set; }

        [LoadColumn(3)]
        public float TrestBps { get; set; }

        [LoadColumn(4)]
        public float Chol { get; set; }

        [LoadColumn(5)]
        public float Fbs { get; set; }

        [LoadColumn(6)]
        public float RestEcg { get; set; }

        [LoadColumn(7)]
        public float Thalac { get; set; }

        [LoadColumn(8)]
        public float Exang { get; set; }

        [LoadColumn(9)]
        public float OldPeak { get; set; }

        [LoadColumn(10)]
        public float Slope { get; set; }

        [LoadColumn(11)]
        public float Ca { get; set; }

        [LoadColumn(12)]
        public float Thal { get; set; }
        
        [LoadColumn(13)]
        public bool Label { get; set; }
    }

    /// <summary>
    /// The HeartPrediction class contains a single heart data prediction.
    /// </summary>
    public class HeartPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction;

        public float Probability;

        public float Score;
    }

    /// <summary>
    /// The application class.
    /// </summary>
    public class Program
    {
        // filenames for training and test data
        private static string trainingDataPath = Path.Combine(Environment.CurrentDirectory, "HeartTraining.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "HeartTest.csv");

        /// <summary>
        /// The main applicaton entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        public static void Main(string[] args)
        {
            // set up a machine learning context
            var mlContext = new MLContext();

            // load training and test data
            Console.WriteLine("Loading data...");
            var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(trainingDataPath, hasHeader: true, separatorChar: ';');
            var testDataView = mlContext.Data.LoadFromTextFile<HeartData>(testDataPath, hasHeader: true, separatorChar: ';');

            // set up a training pipeline
            // step 1: concatenate all feature columns
            var pipeline = mlContext.Transforms.Concatenate(
                "Features", 
                "Age", 
                "Sex", 
                "Cp", 
                "TrestBps",
                "Chol", 
                "Fbs", 
                "RestEcg", 
                "Thalac", 
                "Exang", 
                "OldPeak", 
                "Slope", 
                "Ca", 
                "Thal")

                // step 2: set up a fast tree learner
                .Append(mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: DefaultColumnNames.Label, 
                    featureColumnName: DefaultColumnNames.Features));

            // train the model
            Console.WriteLine("Training model...");
            var trainedModel = pipeline.Fit(trainingDataView);

            // make predictions for the test data set
            Console.WriteLine("Evaluating model...");
            var predictions = trainedModel.Transform(testDataView);

            // compare the predictions with the ground truth
            var metrics = mlContext.BinaryClassification.Evaluate(
                data: predictions, 
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

            // set up a prediction engine
            Console.WriteLine("Making a prediction for a sample patient...");
            var predictionEngine = trainedModel.CreatePredictionEngine<HeartData, HeartPrediction>(mlContext);

            // create a sample patient
            var heartData = new HeartData()
            { 
                Age = 36.0f,
                Sex = 1.0f,
                Cp = 4.0f,
                TrestBps = 145.0f,
                Chol = 210.0f,
                Fbs = 0.0f,
                RestEcg = 2.0f,
                Thalac = 148.0f,
                Exang = 1.0f,
                OldPeak = 1.9f,
                Slope = 2.0f,
                Ca = 1.0f,
                Thal = 7.0f,
            };

            // make the prediction
            var prediction = predictionEngine.Predict(heartData);

            // report the results
            Console.WriteLine($"  Age: {heartData.Age} ");
            Console.WriteLine($"  Sex: {heartData.Sex} ");
            Console.WriteLine($"  Cp: {heartData.Cp} ");
            Console.WriteLine($"  TrestBps: {heartData.TrestBps} ");
            Console.WriteLine($"  Chol: {heartData.Chol} ");
            Console.WriteLine($"  Fbs: {heartData.Fbs} ");
            Console.WriteLine($"  RestEcg: {heartData.RestEcg} ");
            Console.WriteLine($"  Thalac: {heartData.Thalac} ");
            Console.WriteLine($"  Exang: {heartData.Exang} ");
            Console.WriteLine($"  OldPeak: {heartData.OldPeak} ");
            Console.WriteLine($"  Slope: {heartData.Slope} ");
            Console.WriteLine($"  Ca: {heartData.Ca} ");
            Console.WriteLine($"  Thal: {heartData.Thal} ");
            Console.WriteLine();
            Console.WriteLine($"Prediction: {(prediction.Prediction ? "A disease could be present" : "Not present disease" )} ");
            Console.WriteLine($"Probability: {prediction.Probability} ");

            Console.ReadLine();
        }

    }
}
