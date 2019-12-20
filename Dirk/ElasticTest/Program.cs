using System.IO;
using System;
using Nest;
using Elasticsearch.Net;
using System.Linq;

namespace ElasticTest
{
    /// <summary>
    /// The country class represents data to be stored in Elastic.
    /// </summary>
    public class Country
    {
        public string Name { get; set; }
    }

    class Program
    {
        // use this file with sample data
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "sampledata.txt");

        static void Main(string[] args)
        {
            // load the sample data - a list of countries scraped from Reuters articles
            var lines = File.ReadAllLines(dataPath);
            var countries = from l in lines select new Country() { Name = l };
            Console.WriteLine($"Found {countries.Count()} documents ready for indexing...");

            // ask for the admin password, so I can safely store this code in github
            Console.Write("Please provide password for Elastic cluster: ");
            var password = Console.ReadLine();

            // connect to elasticsearch using basic auth
            var credentials = new BasicAuthenticationCredentials("elastic", password);
            var cloudId = "Test:d2VzdGV1cm9wZS5henVyZS5lbGFzdGljLWNsb3VkLmNvbTo5MjQzJDI3MTE2MzJhMDBjZTRmNmZhNDE1MWU2ZmU2NDYxYTU3JDYwMGRhOTBhNDUwMzRmNmE4MDA5NDg2OWIxZDZhYmJi";
            var settings = new ConnectionSettings(cloudId, credentials);
            var client = new ElasticClient(settings);

            // index the country objects using bulk indexing
            var response = client.Bulk(b => b.Index("countries").IndexMany(countries));

            // report any errors
            if (response.Errors)
                Console.WriteLine(response.ServerError.ToString());
            else
                Console.WriteLine($"Indexed {countries.Count()} new documents!");
        }
    }
}
