using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;
using System.Linq;

namespace ProductRecommender
{
    /// <summary>
    /// The ProductInfo class represents one single product from the dataset.
    /// </summary>
    public class ProductInfo
    {
        [LoadColumn(0)] public float ProductID { get; set; }
        [LoadColumn(1)] public float CombinedProductID { get; set; }
    }

    /// <summary>
    /// The ProductPrediction class represents a prediction made by the model.
    /// </summary>
    public class ProductPrediction
    {
        public float Score { get; set; }
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filename for dataset
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "amazon0302.txt");

        /// <summary>
        /// The main entry point of the program.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var context = new MLContext();

            // load the dataset in memory
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<ProductInfo>(
                dataPath, 
                hasHeader: true, 
                separatorChar: '\t');

            // split the data into 80% training and 20% testing partitions
            var partitions = context.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            // prepare matrix factorization options
            var options = new MatrixFactorizationTrainer.Options()
            {
                MatrixColumnIndexColumnName = "ProductIDEncoded",
                MatrixRowIndexColumnName = "CombinedProductIDEncoded",
                LabelColumnName = "CombinedProductID",
                LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
                Alpha = 0.01,
                Lambda = 0.025,
            };

            // set up a training pipeline
            // step 1: map ProductID and CombinedProductID to keys
            var pipeline = context.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "ProductID",
                    outputColumnName: "ProductIDEncoded")
                .Append(context.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "CombinedProductID",
                    outputColumnName: "CombinedProductIDEncoded"))

                // step 2: find recommendations using matrix factorization
                .Append(context.Recommendation().Trainers.MatrixFactorization(options));
            
            // train the model
            Console.WriteLine("Training the model...");
            ITransformer model = pipeline.Fit(partitions.TrainSet);
            Console.WriteLine();

            // evaluate the model performance 
            Console.WriteLine("Evaluating the model...");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = context.Regression.Evaluate(predictions, label: "CombinedProductID", score: DefaultColumnNames.Score);
            Console.WriteLine($"  RMSE: {metrics.Rms:#.##}");
            Console.WriteLine($"  L1:   {metrics.L1:#.##}");
            Console.WriteLine($"  L2:   {metrics.L2:#.##}");
            Console.WriteLine();

            // check how well products 3 and 63 go together
            Console.WriteLine("Predicting if two products combine...");
            var engine = model.CreatePredictionEngine<ProductInfo, ProductPrediction>(context);
            var prediction = engine.Predict(
                new ProductInfo()
                {
                    ProductID = 3,
                    CombinedProductID = 63
                });
            Console.WriteLine($"  Score of products 3 and 63 combined: {prediction.Score}");
            Console.WriteLine();

            // find the top 5 combined products for product 6
            Console.WriteLine("Calculating the top 5 products for product 3...");
            var top5 =  (from m in Enumerable.Range(1, 262111)
                         let p = engine.Predict(
                            new ProductInfo()
                            {
                                ProductID = 3,
                                CombinedProductID = (uint)m
                            })
                         orderby p.Score descending
                         select (ProductID: m, Score: p.Score)).Take(5);
            foreach (var t in top5)
                Console.WriteLine($"  Score:{t.Score}\tProduct: {t.ProductID}");

            Console.ReadLine();

        }
    }
}
