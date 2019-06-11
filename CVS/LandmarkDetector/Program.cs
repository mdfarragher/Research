using System;
using DlibDotNet;
using DlibDotNet.Extensions;
using Dlib = DlibDotNet.Dlib;

namespace LandmarkDetector
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
            // set up Dlib facedetectors and shapedetectors
            using (var fd = Dlib.GetFrontalFaceDetector())
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                // load input image
                var img = Dlib.LoadImage<RgbPixel>(inputFilePath);

                // find all faces in the image
                var faces = fd.Operator(img);
                foreach (var face in faces)
                {
                    // find the landmark points for this face
                    var shape = sp.Detect(img, face);

                    // draw the landmark points on the image
                    for (var i = 0; i < shape.Parts; i++)
                    {
                        var point = shape.GetPart((uint)i);
                        var rect = new Rectangle(point);
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                    }
                }

                // export the modified image
                Dlib.SaveJpeg(img, "output.jpg");
            }
        }
    }
}
