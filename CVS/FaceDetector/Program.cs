using System;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;

namespace FaceDetector
{
    /// <summary>
    /// The main program class
    /// </summary>
    class Program
    {
        // file paths
        private const string inputFilePath = "./input.jpg";

        /// <summary>
        /// The main program entry point
        /// </summary>
        /// <param name="args">The command line arguments</param>
        static void Main(string[] args)
        {
            // set up Dlib facedetector
            using (var fd = Dlib.GetFrontalFaceDetector())
            {
                // load input image
                var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

                // find all faces in the image
                var faces = fd.Operator(img);
                foreach (var face in faces)
                {
                    // draw a rectangle for each face
                    Dlib.DrawRectangle(img, face, color: new RgbPixel(0, 255, 255), thickness: 4);
                }

                // export the modified image
                Dlib.SaveJpeg(img, "output.jpg");
            }
        }
    }
}
