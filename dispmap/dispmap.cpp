#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/SymEigen.h>
#include <TooN/LU.h>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <thread>
#include <cstdio>
#include <iostream>


using namespace cv;
using namespace std;
using namespace TooN;

enum DispAlgorithm { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };

DispAlgorithm alg = STEREO_BM;
int SADWindowSize = 10;

Mat disp;

size_t numPoints = 640*480;
GLuint vboId, iboId;
GLfloat* vertices = new GLfloat[numPoints*3];
bool vboUpdated = false;
int numPointsUsed = 0;
SE3<> groundPlane1;
SE3<> groundPlane2;
mutex vbo_mutex;

int moving = 0;
int beginx, beginy;
double yaw = 0, pitch = 0;

StereoBM bm(StereoBM::PREFILTER_NORMALIZED_RESPONSE);
StereoSGBM sgbm;
StereoVar var;

int g_numberOfDisparities;
int g_uniquenessRatio = 10;
int g_speckleWindowSize = 100;
int g_speckleRange = 32;
int g_disp12MaxDiff = 1;

void
draw_plane(const SE3<>& plane, float size)
{
  TooN::Vector<3> pt1 = plane * makeVector(-size,  size, 0);
  TooN::Vector<3> pt2 = plane * makeVector( size,  size, 0);
  TooN::Vector<3> pt3 = plane * makeVector( size, -size, 0);
  TooN::Vector<3> pt4 = plane * makeVector(-size, -size, 0);

  glLineWidth(5);
  glBegin(GL_LINE_STRIP);
    glVertex3f(pt1[0], pt1[1], pt1[2]);
    glVertex3f(pt2[0], pt2[1], pt2[2]);
    glVertex3f(pt3[0], pt3[1], pt3[2]);
    glVertex3f(pt4[0], pt4[1], pt4[2]);
    glVertex3f(pt1[0], pt1[1], pt1[2]);
  glEnd();
}

void
display()
{
  {
    std::lock_guard<std::mutex> lock(vbo_mutex);
    if (vboUpdated) {
      glBindBuffer(GL_ARRAY_BUFFER, vboId);         // for vertex coordinates
      glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*3*numPoints, vertices);
      vboUpdated = false;
    }
  }

  glLoadIdentity();

  glTranslatef(0, 0, -200);
  glRotatef(yaw, 0, 1, 0);
  glRotatef(pitch, 1, 0, 0);
  glTranslatef(0, 0, 100);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glColor3f(1,0,1);
  glPointSize(1);

  // bind VBOs for vertex array and index array
  glBindBuffer(GL_ARRAY_BUFFER, vboId);         // for vertex coordinates
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboId); // for indices

  // do same as vertex array except pointer
  glEnableClientState(GL_VERTEX_ARRAY);             // activate vertex coords array
  glVertexPointer(3, GL_FLOAT, sizeof(GLfloat)*3, 0);               // last param is offset, not ptr

  glDrawElements(GL_POINTS, numPointsUsed, GL_UNSIGNED_INT, 0);

  glDisableClientState(GL_VERTEX_ARRAY);            // deactivate vertex array

  // bind with 0, so, switch back to normal pointer operation
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glColor3f(0, 1, 0);
  draw_plane(groundPlane1, 50.0f);
  glColor3f(1, 0, 0);
  draw_plane(groundPlane2, 50.0f);

  glutSwapBuffers();
  glutPostRedisplay();

  usleep(10000);
}

void
reshape(int w, int h)
{
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(80.0, 640.0/480.0, 1, 2000);
  glMatrixMode(GL_MODELVIEW);
}

void
mouse(int button, int state, int x, int y)
{
  if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
    moving = 1;
    beginx = x;
    beginy = y;
  }
  if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
    moving = 0;
  }
}

void
motion(int x, int y)
{
  if (moving) {
    yaw += (x - beginx) * 0.1;
    pitch += (y - beginy) * 0.1;
    beginx = x;
    beginy = y;
  }
}

void
init_glut(int argc, char** argv)
{
  glutInit(&argc, argv);
  glutInitWindowPosition(500, 500);
  glutInitWindowSize(640, 480);
  glutCreateWindow("points");
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  GLenum err = glewInit();
  if (GLEW_OK != err)
  {
    /* Problem: glewInit failed, something is seriously wrong. */
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }

  glGenBuffers(1, &vboId);
  glBindBuffer(GL_ARRAY_BUFFER, vboId);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*3*numPoints, NULL, GL_DYNAMIC_DRAW);
//  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(GLfloat)*3*numPoints, geometry);

  GLuint* indices = new GLuint[numPoints];
  for (size_t i = 0; i < numPoints; ++i) {
    indices[i] = i;
  }

  glGenBuffers(1, &iboId);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboId);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numPoints, NULL, GL_STATIC_DRAW);
  glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(GLuint)*numPoints, indices);

  delete[] indices;

  glutReshapeFunc(reshape);
  glutDisplayFunc(display);

  glutMainLoop();

  glDeleteBuffers(1, &vboId);
  glDeleteBuffers(1, &iboId);
}

void
update_visualization(const vector<TooN::Vector<3>>& pointCloud, const SE3<> &plane1, const SE3<> &plane2)
{
  std::lock_guard<std::mutex> lock(vbo_mutex);

  numPointsUsed = 0;

  size_t index = 0;
  for (auto it = pointCloud.begin(); it != pointCloud.end(); ++it) {
    vertices[index++] = (*it)[0];
    vertices[index++] = (*it)[1];
    vertices[index++] = (*it)[2];
    ++numPointsUsed;
  }

  groundPlane1 = plane1.inverse();
  groundPlane2 = plane2.inverse();

  vboUpdated = true;
}

void
mouse_handler(int event, int x, int y, int flags, void* param)
{
  switch (event) {
  case CV_EVENT_LBUTTONDOWN:
    cout << disp.at<float>(y, x) << endl;
    break;
  default:
    break;
  }
}

SE3<> AlignerFromPointAndUp(const TooN::Vector<3>& point,
                            const TooN::Vector<3>& normal)
{
  Matrix<3> m3Rot = Identity;
  m3Rot[2] = normal;
  m3Rot[0] = m3Rot[0] - (normal * (m3Rot[0] * normal));
  normalize(m3Rot[0]);
  m3Rot[1] = m3Rot[2] ^ m3Rot[0];

  SE3<> se3;
  se3.get_rotation() = m3Rot;
  TooN::Vector<3> v3RMean = se3 * point;
  se3.get_translation() = -v3RMean;

  return se3;
}

TooN::Vector<4> Se3ToPlane(const SE3<>& se3)
{
  TooN::Vector<3> normal = se3.get_rotation().get_matrix()[2];
  double d = -normal * se3.inverse().get_translation();
  return makeVector(normal[0], normal[1], normal[2], d);
}

TooN::SE3<> PlaneToSe3(const TooN::Vector<4>& plane)
{
  TooN::Vector<3> normal = plane.slice<0,3>();
  normalize(normal);
  TooN::Vector<3> point = -plane[3] * normal;
  return AlignerFromPointAndUp(point, normal);
}

void
ExtractPoints(const Mat &_3dImage, vector<TooN::Vector<3> >& points)
{
  const double max_z = 2000.0;

  points.clear();

  for (int i = 0; i < _3dImage.rows; ++i) {
    for (int j = 0; j < _3dImage.cols; ++j) {
      const Vec3f &pt = _3dImage.at<Vec3f>(i, j);
      if (fabs(pt[2]) < max_z) {
        points.push_back(makeVector(-pt[0], pt[1], pt[2]));
      }
    }
  }
}

bool FindPlaneAligner(const vector<TooN::Vector<3> >& points,
                      bool bFlipNormal, double inlierThreshold,
                      SE3<>& planeAligner)
{
  size_t nPoints = points.size();
  if(nPoints < 10) {
    cerr << "FindPlaneAligner needs more point to calculate plane" << endl;
    return false;
  }

  int nRansacs = 500;
  TooN::Vector<3> v3BestMean;
  TooN::Vector<3> v3BestNormal;
  double dBestDistSquared = numeric_limits<double>::max();

  for (int i = 0; i < nRansacs; ++i) {
    int nA = rand()%nPoints;
    int nB = nA;
    int nC = nA;
    while(nB == nA)
      nB = rand()%nPoints;
    while(nC == nA || nC==nB)
      nC = rand()%nPoints;

    TooN::Vector<3> v3Mean = (1.0/3.0) * (points[nA] +
                                          points[nB] +
                                          points[nC]);

    TooN::Vector<3> v3CA = points[nC]  - points[nA];
    TooN::Vector<3> v3BA = points[nB]  - points[nA];
    TooN::Vector<3> v3Normal = v3CA ^ v3BA;
    if ((v3Normal * v3Normal) == 0) {
      continue;
    }
    normalize(v3Normal);

    double dSumError = 0.0;
    for (size_t i = 0; i < nPoints; ++i) {
      TooN::Vector<3> v3Diff = points[i] - v3Mean;
      double dDistSq = v3Diff * v3Diff;
      if (dDistSq == 0.0) {
        continue;
      }
      double dNormDist = fabs(v3Diff * v3Normal);
      if(dNormDist > inlierThreshold)
        dNormDist = inlierThreshold;
      dSumError += dNormDist;
    }

    if (dSumError < dBestDistSquared) {
      dBestDistSquared = dSumError;
      v3BestMean = v3Mean;
      v3BestNormal = v3Normal;
    }
  }

  // Done the ransacs, now collect the supposed inlier set
  vector<TooN::Vector<3> > vv3Inliers;
  for (size_t i = 0; i < nPoints; ++i) {
    TooN::Vector<3> v3Diff = points[i] - v3BestMean;
    double dDistSq = v3Diff * v3Diff;
    if (dDistSq == 0.0)
      continue;
    double dNormDist = fabs(v3Diff * v3BestNormal);
    if (dNormDist < inlierThreshold)
      vv3Inliers.push_back(points[i]);
  }

  // With these inliers, calculate mean and cov
  TooN::Vector<3> v3MeanOfInliers = Zeros;
  for (size_t i = 0; i < vv3Inliers.size(); ++i) {
    v3MeanOfInliers += vv3Inliers[i];
  }

  v3MeanOfInliers *= (1.0 / vv3Inliers.size());

  Matrix<3> m3Cov = Zeros;
  for (size_t i = 0; i < vv3Inliers.size(); ++i) {
    TooN::Vector<3> v3Diff = vv3Inliers[i] - v3MeanOfInliers;
    m3Cov += v3Diff.as_col() * v3Diff.as_row();
  }

  // Find the principal component with the minimal variance: this is the plane normal
  SymEigen<3> sym(m3Cov);
  TooN::Vector<3> v3Normal = sym.get_evectors()[0];

  if (bFlipNormal) {
    // Use the version of the normal which points towards the cam center
    if(v3Normal[2] > 0) {
      v3Normal *= -1.0;
    }
  } else {
    // Use the positive Z
    if(v3Normal[2] < 0) {
      v3Normal *= -1.0;
    }
  }

  planeAligner = AlignerFromPointAndUp(v3MeanOfInliers, v3Normal);
  return true;
}

void
GenerateDisparityMap(const Mat &img1, const Mat &img2, DispAlgorithm alg, int numberOfDisparities,
                     const Rect &roi1, const Rect &roi2, const Mat &Q, Mat &disp)
{
  if (alg == STEREO_BM) {
    Mat img1bw, img2bw;

    cvtColor(img1, img1bw, CV_RGB2GRAY);
    cvtColor(img2, img2bw, CV_RGB2GRAY);

    bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = SADWindowSize >= 5 ? SADWindowSize | 1 : 9;
    bm.state->minDisparity = 0;
    bm.state->numberOfDisparities = numberOfDisparities;
    bm.state->textureThreshold = 10;
    bm.state->uniquenessRatio = g_uniquenessRatio;
    bm.state->speckleWindowSize = g_speckleWindowSize;
    bm.state->speckleRange = g_speckleRange;
    bm.state->disp12MaxDiff = 1;

    bm.state->trySmallerWindows = 1;

    bm(img1bw, img2bw, disp, CV_32F);
  } else if (alg == STEREO_VAR) {
    var.levels = 3;                       // ignored with USE_AUTO_PARAMS
    var.pyrScale = 0.5;                   // ignored with USE_AUTO_PARAMS
    var.nIt = 25;
    var.minDisp = -numberOfDisparities;
    var.maxDisp = 0;
    var.poly_n = 3;
    var.poly_sigma = 0.0;
    var.fi = 15.0f;
    var.lambda = 0.03f;
    var.penalization = var.PENALIZATION_TICHONOV; // ignored with USE_AUTO_PARAMS
    var.cycle = var.CYCLE_V;              // ignored with USE_AUTO_PARAMS
    var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;

    var(img1, img2, disp);
  } else if (alg == STEREO_SGBM || alg == STEREO_HH) {
    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;

    int cn = img1.channels();

    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = g_uniquenessRatio;
    sgbm.speckleWindowSize = g_speckleWindowSize;
    sgbm.speckleRange = g_speckleRange;
    sgbm.disp12MaxDiff = -1; //sg_disp12MaxDiff;
    sgbm.fullDP = alg == STEREO_HH;

    sgbm(img1, img2, disp);
  }
}


template <int StateSize, int ObsSize, int CtrlSize>
class KalmanFilter {
  public:
    KalmanFilter()
    {
    }

    KalmanFilter(const Matrix<StateSize>& A, const Matrix<StateSize, CtrlSize>& B,
                 const Matrix<StateSize>& R, const Matrix<ObsSize, StateSize>& C,
                 const Matrix<StateSize>& Q)
      : A_(A), B_(B), R_(R), C_(C), Q_(Q)
    {
    }

    void init(const TooN::Vector<StateSize>& mu, const Matrix<StateSize>& sigma)
    {
      mu_ = mu;
      sigma_ = sigma;
    }

    void predict(const TooN::Vector<CtrlSize>& ctrl)
    {
      muBar_ = A_ * mu_ + B_ * ctrl;
      sigmaBar_ = A_ * sigma_ * A_.T() + R_;
    }

    void observe(const TooN::Vector<ObsSize>& obs)
    {
      TooN::LU<StateSize> lu(C_ * sigmaBar_ * C_.T() + Q_);
      Matrix<> K = sigmaBar_ * C_.T() * lu.get_inverse();
      mu_ = muBar_ + K * (obs - C_ * muBar_);
      sigma_ = (Identity - K * C_) * sigmaBar_;
    }

    const TooN::Vector<StateSize>& mu() const { return mu_; }

  private:
    Matrix<StateSize> A_;
    Matrix<StateSize, CtrlSize> B_;
    Matrix<StateSize> R_;
    Matrix<ObsSize, StateSize> C_;
    Matrix<StateSize> Q_;

    TooN::Vector<StateSize> mu_;
    TooN::Vector<StateSize> muBar_;
    Matrix<StateSize> sigma_;
    Matrix<StateSize> sigmaBar_;
};

class KalmanPlaneEstimator {
  public:
    KalmanPlaneEstimator()
      : first_(true)
    {
      Matrix<4> R = Zeros;
      R.diagonal_slice() = makeVector(0.4, 0.4, 0.4, 8.0); // Prediction noise
      Matrix<4> Q = Zeros;
      Q.diagonal_slice() = makeVector(0.2, 0.2, 0.2, 4.0); // Observation noise
      filter_ = KalmanFilter<4, 4, 1>(Identity, Zeros, R, Identity, Q);
    }

    void update(const SE3<>& planeAligner)
    {
      if (first_) {
        filter_.init(Se3ToPlane(planeAligner), Identity);
        first_ = false;
      } else {
        filter_.predict(Zeros);
        filter_.observe(Se3ToPlane(planeAligner));
      }
    }

    SE3<> getPlane() const {
      return PlaneToSe3(filter_.mu());
    }

  private:
    bool first_;
    KalmanFilter<4, 4, 1> filter_;
};

void opencv_thread(int cam1, int cam2)
{
  VideoCapture capture1;
  capture1.open(cam1);

  if (!capture1.isOpened())
    printf("Cannot initialize camera 0\n");

  capture1.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
  capture1.set( CV_CAP_PROP_FRAME_HEIGHT, 480 );

  VideoCapture capture2;
  capture2.open(cam2);

  if (!capture2.isOpened())
      printf("Cannot initialize camera 1\n");

  capture2.set( CV_CAP_PROP_FRAME_WIDTH, 640 );
  capture2.set( CV_CAP_PROP_FRAME_HEIGHT, 480 );

  namedWindow("left", 1);
  namedWindow("right", 1);
  namedWindow("disp", 1);

  cvSetMouseCallback("disp", mouse_handler);

  Mat img1;
  Mat img2;
  capture1 >> img1;
  capture2 >> img2;

  Size img_size = img1.size();

  int numberOfDisparities = ((img_size.width/8) + 15) & -16;

  createTrackbar("numberOfDisparities", "disp", &numberOfDisparities, 1000);
  createTrackbar("uniquenessRatio", "disp", &g_uniquenessRatio, 100);
  createTrackbar("speckleWindowSize", "disp", &g_speckleWindowSize, 500);
  createTrackbar("speckleRange", "disp", &g_speckleRange, 100);
  createTrackbar("disp12MaxDiff", "disp", &g_disp12MaxDiff, 20);
  createTrackbar("SADWindowSize", "disp", &SADWindowSize, 50);


  // reading intrinsic parameters
  FileStorage fs("intrinsics.yml", CV_STORAGE_READ);
  if(!fs.isOpened())
  {
    printf("Failed to intrinsic.yml");
    return;
  }

  Mat M1, D1, M2, D2;
  fs["M1"] >> M1;
  fs["D1"] >> D1;
  fs["M2"] >> M2;
  fs["D2"] >> D2;

  fs.open("extrinsics.yml", CV_STORAGE_READ);
  if(!fs.isOpened())
  {
    printf("Failed to open extrinsinc.yml");
    return;
  }

  Mat R, T, R1, P1, R2, P2;
  fs["R"] >> R;
  fs["T"] >> T;

  Rect roi1, roi2;
  Mat Q;

  stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q,
                CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

  cout << Q << endl;

  Mat map11, map12, map21, map22;
  initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
  initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

  Mat img1r, img2r;

  KalmanPlaneEstimator planeEstimator;

  while (true) {

    for (int i = 0; i < 3; ++i) {
      capture1 >> img1;
      capture2 >> img2;
    }

    if (img1.empty() || img2.empty()) {
        break;
    }

    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);

    img1 = img1r;
    img2 = img2r;

    //int64 s = getTickCount();
    GenerateDisparityMap(img1, img2, alg, numberOfDisparities & -16, roi1, roi2, Q, disp);
    //cout << "Elapsed: " << (double)(getTickCount() - s)*1000.0/getTickFrequency() << endl;

    assert(disp.type() == CV_32F);

    Mat xyz;
    reprojectImageTo3D(disp, xyz, Q);

    // Convert the Mat into a list of points (with some filtering)
    vector<TooN::Vector<3> > pointCloud;
    ExtractPoints(xyz, pointCloud);

    SE3<> plane;
    if (FindPlaneAligner(pointCloud, false, 1.0, plane)) {
      // Normalize plane
      plane = PlaneToSe3(Se3ToPlane(plane));
      // Send it through the Kalman filter
      planeEstimator.update(plane);
      SE3<> filteredPlane = planeEstimator.getPlane();
      // Distance is the Z component of the translation (in cm)
      cout << "Distance: " << filteredPlane.get_translation()[2] << endl;
      //
      update_visualization(pointCloud, plane, filteredPlane);
    }

    Mat disp8;
    if (alg != STEREO_VAR) {
      disp.convertTo(disp8, CV_8U, 255.0/numberOfDisparities);
    } else {
      disp.convertTo(disp8, CV_8U);
    }

    imshow("left", img1);
    imshow("right", img2);
    imshow("disp", disp8);

    int key = waitKey(10);
    if (key == 27) {
      break;
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) return 1;

  int cam1 = atoi(argv[1]);
  int cam2 = atoi(argv[2]);

  thread cv_thread(bind(opencv_thread, cam1, cam2));

  init_glut(argc, argv);

  cv_thread.join();

  return 0;
}
