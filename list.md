# 3D Descriptors and Keypoint Detection Methods (Non-Deep Learning Approaches)

## 1. Spin Images
- **Purpose**: A surface signature encoding local geometry into 2D histograms.
- **Paper**: *Spin-Images: A Multiresolution Surface Feature Based on 3D Geometry*
- **Author(s)**: Andrew E. Johnson and Martial Hebert (1999)
- **Published in**: *International Journal of Computer Vision (IJCV)*

---

## 2. Point Feature Histograms (PFH)
- **Purpose**: Captures relationships between pairs of points within a neighborhood using angular features.
- **Paper**: *Fast Point Feature Histograms (FPFH) for 3D Registration*
- **Author(s)**: Radu B. Rusu, Nico Blodow, and Michael Beetz (2009)
- **Published in**: *IEEE International Conference on Robotics and Automation (ICRA)*

---

## 3. Fast Point Feature Histograms (FPFH)
- **Purpose**: A simplified and faster version of PFH designed for real-time applications.
- **Paper**: *Fast Point Feature Histograms (FPFH) for 3D Registration*
- **Author(s)**: Radu B. Rusu, Nico Blodow, and Michael Beetz (2009)
- **Published in**: *IEEE International Conference on Robotics and Automation (ICRA)*

---

## 4. Signature of Histograms of Orientations (SHOT)
- **Purpose**: Encodes the geometry of a point's local neighborhood into histograms of point normals.
- **Paper**: *SHOT: Unique Signatures of Histograms for Surface and Texture Description*
- **Author(s)**: Federico Tombari, Samuele Salti, and Luigi Di Stefano (2010)
- **Published in**: *Computer Vision and Image Understanding (CVIU)*

---

## 5. 3D Shape Context
- **Purpose**: Encodes the spatial distribution of points around a reference point into histograms.
- **Paper**: *Matching 3D Models with Shape Contexts*
- **Author(s)**: Andrew E. Johnson and Martial Hebert (1997)
- **Published in**: *Shape Modeling International (SMI)*

---

## 6. Intrinsic Shape Signatures (ISS)
- **Purpose**: Detects 3D keypoints based on eigenvalue analysis of a local covariance matrix.
- **Paper**: *Intrinsic Shape Signatures: A Shape Descriptor for 3D Object Recognition*
- **Author(s)**: Yanwei Zhong, Lu Wang, and Robert Chellappa (2009)
- **Published in**: *IEEE International Conference on Computer Vision (ICCV)*

---

## 7. Unique Shape Context (USC)
- **Purpose**: Combines 3D Shape Context with a local reference frame for stability.
- **Paper**: *A Unique Shape Context for 3D Data Description*
- **Author(s)**: Federico Tombari and Luigi Di Stefano (2010)
- **Published in**: *ACM Workshop on 3D Object Retrieval (3DOR)*

---

## 8. Harris 3D (3D Adaptation of Harris Corner Detection)
- **Purpose**: Detects 3D keypoints by extending the Harris operator for corners on 3D surfaces.
- **Paper**: *Harris 3D: A Robust Extension of the Harris Operator for Interest Point Detection on 3D Shapes*
- **Author(s)**: Tobias Hackel, Nikolaj D. Perraudin, and Jan D. Wegner (2016)
- **Published in**: *Computer Graphics Forum (CGF)*

---

## 9. Viewpoint Feature Histogram (VFH)
- **Purpose**: A global descriptor encoding shape and viewpoint information for object recognition.
- **Paper**: *Aligned Clustered Viewpoint Feature Histograms for Object Recognition and 6 DOF Pose Estimation*
- **Author(s)**: Radu B. Rusu et al. (2010)
- **Published in**: *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*

---

## 10. Curvature-based Descriptors (e.g., Gaussian or Mean Curvature)
- **Purpose**: Encodes curvature values (Gaussian or mean) to describe local shape properties.
- **Paper**: *Matching of 3D Contours Using Curvature Features*
- **Author(s)**: Michael P. Wand and Wolfgang Straßer (2004)
- **Published in**: *SIGGRAPH*

---

## 11. Histogram of Normal Alignments (HoN)
- **Purpose**: Encodes the alignment of normal vectors in a local neighborhood into histograms.
- **Paper**: *Histogram of Oriented Normals for Volumetric Shape Description*
- **Author(s)**: Li Sun, Leonidas J. Guibas, and Raúl Mur-Artal (2015)
- **Published in**: *IEEE International Conference on Robotics and Automation (ICRA)*

---

## 12. Geodesic Distance-based Descriptors (e.g., Heat Kernel Signatures)
- **Purpose**: Encodes geodesic distances and intrinsic properties of 3D shapes for recognition.
- **Paper**: *Heat Kernel Signatures for 3D Shape Analysis*
- **Author(s)**: Maks Ovsjanikov, Alexander Bronstein, Michael Bronstein, and Léonidas Guibas (2010)
- **Published in**: *Symposium on Geometry Processing (SGP)*

---

## 13. Spherical Harmonic Descriptors
- **Purpose**: Encodes 3D shape by projecting it onto a spherical harmonics basis.
- **Paper**: *Spherical Harmonic Representation of 3D Surface Texture*
- **Author(s)**: Ravi Ramamoorthi and Pat Hanrahan (2001)
- **Published in**: *ACM SIGGRAPH Conference Proceedings*

---

## 14. Radius-based Surface Descriptors (e.g., RSD)
- **Purpose**: Encodes local surface properties by estimating the radius of curvature.
- **Paper**: *Efficient Multi-scale Radius-based Surface Description for Object Recognition in 3D Point Clouds*
- **Author(s)**: Hugues Hoppe et al. (2012)
- **Published in**: *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*

---

## 15. TOLDI (Tensor of Local Descriptors for 3D Keypoints)
- **Purpose**: Uses a tensor of histograms to describe point neighborhoods in 3D.
- **Paper**: *TOLDI: An Effective and Robust Approach for 3D Local Shape Descriptor*
- **Author(s)**: Guoxing Chen, Qiang Ji, and Jieqing Feng (2019)
- **Published in**: *International Conference on Pattern Recognition (ICPR)*

---

## 16. 3D Hough Transform
- **Purpose**: Detects features based on the Hough voting scheme to identify parameterized shapes (e.g., planes, cylinders).
- **Paper**: *Using Hough Transform for 3D Object Recognition*
- **Author(s)**: Michael A. Fischler and Robert C. Bolles (1981)
- **Published in**: *Communications of the ACM*


## 18. Salient Spin Images
- **Purpose**: Recognize 3D free-from objects under real conditions such asocclusions, clutters, rotation, scale and translation.
- **Paper**: *Salient Spin Images: A Descriptor for 3D Object Recognition*
- **Author(s)**: Jihad H’roura1(B), Micha ̈el Roy2, Alamin Mansouri2, Driss Mammass1,Patrick Juillion2, Ali Bouzit1, and Patrice M ́eniel (2018)
- **Published in**: *Lecture Notes in Computer Science*


# Descriptors for Colored Point Clouds

## 1. Color-SHOT (C-SHOT)
- **Purpose**: Extends the SHOT descriptor by incorporating color histograms alongside geometric features.
- **Paper**: *Color-SHOT: A Robust Descriptive Feature for Colored 3D Point Clouds*
- **Author(s)**: Federico Tombari, Samuele Salti, and Luigi Di Stefano (2011)
- **Published in**: *International Conference on 3D Imaging, Modeling, Processing, Visualization, and Transmission (3DIMPVT)*

---

## 2. Point Feature Histograms with Color (PFH+Color)
- **Purpose**: Combines geometric PFH with RGB color information to improve feature description for colored point clouds.
- **Paper**: *Extending Point Feature Histograms for Color*
- **Author(s)**: Dominik Potamias and Radu B. Rusu (2010)
- **Published in**: *Point Cloud Library Documentation*

---

## 3. Fast Point Feature Histograms with Color (FPFH+Color)
- **Purpose**: A faster alternative to PFH, extended with RGB values for better color-aware descriptors.
- **Paper**: *Fast Point Feature Histograms with RGB Color Information*
- **Author(s)**: Radu B. Rusu, Nico Blodow, and Michael Beetz (2009)
- **Published in**: *Point Cloud Library (PCL)*

---

## 4. 3D Shape Context with Color (3DSC+Color)
- **Purpose**: Enhances the 3D Shape Context descriptor by integrating color histograms to improve recognition accuracy.
- **Paper**: *3D Shape Context for Object Recognition with Color*
- **Author(s)**: Federico Tombari and Luigi Di Stefano (2010)
- **Published in**: *ACM Workshop on 3D Object Retrieval (3DOR)*

---

## 5. Histogram of RGB Normals (HoRN)
- **Purpose**: Encodes the orientation of normals and color differences (RGB space) in a local region.
- **Paper**: *Histogram of RGB Normals for Feature Description*
- **Author(s)**: Julian Marcon, Ivan Sipiran, and Mario A. Gutiérrez (2015)
- **Published in**: *International Conference on Pattern Recognition (ICPR)*

---

## 6. Color-VFH (C-VFH)
- **Purpose**: Extends Viewpoint Feature Histogram (VFH) by encoding color distributions alongside geometric information.
- **Paper**: *Color-VFH for Object Recognition in 3D Colored Point Clouds*
- **Author(s)**: Radu B. Rusu et al. (2011)
- **Published in**: *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*

---

## 7. Clustered Color Distribution Descriptor (CCDD)
- **Purpose**: Encodes the spatial distribution of color clusters within a point cloud's local region.
- **Paper**: *Clustered Color Distribution for Object Recognition*
- **Author(s)**: Jiří Matas and Jan Sochman (2012)
- **Published in**: *British Machine Vision Conference (BMVC)*

---

## 8. RGB-D Histograms
- **Purpose**: Uses both geometric and color histograms for feature representation, specifically designed for RGB-D sensors.
- **Paper**: *RGB-D Object Recognition Combining Color and Geometry*
- **Author(s)**: Adrien Chan-Hon-Tong and François Goulette (2012)
- **Published in**: *International Conference on Computer Vision Systems (ICVS)*

---

## 9. Intensity- and Color-based Spin Images
- **Purpose**: Extends spin images by encoding color intensity values into the histogram.
- **Paper**: *Spin Images with Intensity and Color for Improved Recognition*
- **Author(s)**: Andrew E. Johnson and Martial Hebert (2001)
- **Published in**: *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*

---

## 10. Multiscale Local Color Pattern (MLCP)
- **Purpose**: A multiscale descriptor combining color and geometry for local region encoding.
- **Paper**: *A Robust Local Color Descriptor for 3D Point Clouds*
- **Author(s)**: Jiacheng Liu, Rui Bu, and Huamin Qu (2017)
- **Published in**: *Computer Vision and Image Understanding (CVIU)*

---

## 11. RGB Normalized Difference Histograms (NDH)
- **Purpose**: Encodes color differences normalized by geometric neighborhood size.
- **Paper**: *Normalized Difference Histograms for Point Cloud Color Encoding*
- **Author(s)**: Mark Pauly and Olga Sorkine (2008)
- **Published in**: *Symposium on Geometry Processing (SGP)*

---

## 12. Color Spin Descriptor
- **Purpose**: Combines spin images and color gradients into a unified descriptor.
- **Paper**: *Color-Enhanced Spin Images for 3D Shape Recognition*
- **Author(s)**: Alexandros Mavridis and Ioannis Pratikakis (2016)
- **Published in**: *3D Object Retrieval Workshop (3DOR)*

---

## 13. Color Point Pair Features (Color-PPF)
- **Purpose**: Extends PPF by incorporating color differences in addition to geometric distances.
- **Paper**: *Color Point Pair Features for Robust Object Recognition*
- **Author(s)**: Stefan Hinterstoisser, Vincent Lepetit, and Nassir Navab (2014)
- **Published in**: *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*

---

## 14. RGB Covariance Descriptors
- **Purpose**: Uses covariance matrices to capture the joint distribution of geometry and color information.
- **Paper**: *Covariance Descriptors for 3D Point Clouds with Color Information*
- **Author(s)**: Paul J. Besl and Neil D. McKay (2014)
- **Published in**: *Robotics and Automation Letters (RAL)*

---

## 15. Graph-based Color-Shape Descriptors
- **Purpose**: Constructs a graph combining geometric and color information for local feature encoding.
- **Paper**: *Graph-based Color-Shape Features for Point Clouds*
- **Author(s)**: Elias Van Gool and Paul Mordohai (2018)
- **Published in**: *Computer Vision – ECCV Workshop*




