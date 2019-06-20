using System;
using System.Linq;
using DlibDotNet;
using Dlib = DlibDotNet.Dlib;
using OpenCvSharp;

namespace HeadTracking
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

                    // build the 3d face model
                    var model = Utility.GetFaceModel();

                    // get the landmark point we need
                    var landmarks = new MatOfPoint2d(1, 6,
                        (from i in new int[] { 30, 8, 36, 45, 48, 54 }
                         let pt = shape.GetPart((uint)i)
                         select new OpenCvSharp.Point2d(pt.X, pt.Y)).ToArray());

                    // build the camera matrix
                    var cameraMatrix = Utility.GetCameraMatrix((int)img.Rect.Width, (int)img.Rect.Height);

                    // build the coefficient matrix
                    var coeffs = new MatOfDouble(4, 1);
                    coeffs.SetTo(0);

                    // find head rotation and translation
                    Mat rotation = new MatOfDouble();
                    Mat translation = new MatOfDouble();
                    Cv2.SolvePnP(model, landmarks, cameraMatrix, coeffs, rotation, translation);

                    // find euler angles
                    var euler = Utility.GetEulerMatrix(rotation);

                    // calculate head rotation in degrees
                    var yaw = 180 * euler.At<double>(0, 2) / Math.PI;
                    var pitch = 180 * euler.At<double>(0, 1) / Math.PI;
                    var roll = 180 * euler.At<double>(0, 0) / Math.PI;

                    // looking straight ahead wraps at -180/180, so make the range smooth
                    pitch = Math.Sign(pitch) * 180 - pitch;

                    // calculate if the driver is facing forward
                    // the left/right angle must be in the -25..25 range
                    // the up/down angle must be in the -10..10 range
                    var facingForward = 
                        yaw >= -25 && yaw <= 25
                        && pitch >= -10 && pitch <= 10;

                    // create a new model point in front of the nose, and project it into 2d
                    var poseModel = new MatOfPoint3d(1, 1, new Point3d(0, 0, 1000));
                    var poseProjection = new MatOfPoint2d();
                    Cv2.ProjectPoints(poseModel, rotation, translation, cameraMatrix, coeffs, poseProjection);

                    // draw the key landmark points in yellow on the image
                    foreach (var i in new int[] { 30, 8, 36, 45, 48, 54 })
                    {
                        var point = shape.GetPart((uint)i);
                        var rect = new Rectangle(point);
                        Dlib.DrawRectangle(img, rect, color: new RgbPixel(255, 255, 0), thickness: 4);
                    }

                    // draw a line from the tip of the nose pointing in the direction of head pose
                    var landmark = landmarks.At<Point2d>(0);  
                    var p = poseProjection.At<Point2d>(0);
                    Dlib.DrawLine(
                        img, 
                        new DlibDotNet.Point((int)landmark.X, (int)landmark.Y), 
                        new DlibDotNet.Point((int)p.X, (int)p.Y),
                        color: new RgbPixel(0, 255, 255));

                    // draw a box around the face if it's facing forward
                    if (facingForward)
                        Dlib.DrawRectangle(img, face, color: new RgbPixel(0, 255, 255), thickness: 4);
                }

                // export the modified image
                Dlib.SaveJpeg(img, "output.jpg");
            }
        }
    }
}
