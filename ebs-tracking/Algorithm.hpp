// File        Algorithm.hpp
// Summary     Create the joint detector/tracker using DBSCAN and Kalman nearest neighbors.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <vector>
# include <string>
# include <cmath>

// Other imports
# include "base-classes.hpp"
# include "dbscan-detector.hpp"
# include "meanshift-detector.hpp"
# include "Kalman-NN.hpp"
# include "gmphd.hpp"


class DBSCAN_KNN : public Algo {
     /*
     This class implements an entire detection and tracking pipeline.
     It uses DBSCAN for the detection stage and Kalman Nearest Neighbors
     for the tracking stage.
     */
     private:
     DbscanDetector det;
     KalmanNN track;
     KModel kmodel;
     std::string name {"DBSCAN-KNN"};

     public:
     DBSCAN_KNN(Eigen::MatrixXd params, KModel k_model) 
          : det(params(0,0), params(0,1), params(0,2)), track(k_model, 15, 10.0), kmodel {k_model} {
          /*
          Constructor.

          Args:
               params: An Eigen matrix holding all the simple parameters. Each row will
                       correspond to a different point on the LROC curve.
                       params(0,0): eps
                       params(0,1): num_pts
                       params(0,2): characteristicTime
               k_model: A struct defining the transition and noise matrices for the tracker.

          Ret:
               None
          */
     }

     ~DBSCAN_KNN() {}


     void operator() (double * events, int N) {
          /*
          Feed the detector and tracker a set of events and have it update the tracks,

          Args:
               events: A pointer to memory containing the events. The data should have the format
                       [txyp],[txyp],[txyp],...; where t, x, y, and p are the time, x-,y-position
                       and event polarity.
               N: Number of events.

          Ret:
               None. This function only affects the state of the algorithm class.
          */
          Eigen::MatrixXd detections {det.processSensorData(events, N)};
          track.update(detections);
          return;
     }


     Eigen::MatrixXd currentTracks() {
          /*
          A wrapper for the Tracker function.
          
          Args:
               None

          Ret:
               An Eigen matrix containing the current object locations.
          */
          return track.currentTracks();
     }


     void predict() {
          /*
          A wrapper for the tracker function 'predict'.

          Args:
               None

          Ret:
               None
          */
          track.predict();
          return;
     }

     
     std::string algoName() {
          /*
          Get a string descriptor for the class.

          Args:
               None

          Ret:
               A string describing the class composition.
          */
          return name;
     }
};


class MS_KNN : public Algo {
     /*
     This class implements an entire detection and tracking pipeline.
     It uses Mean Shifts for the detection stage and Kalman Nearest Neighbors
     for the tracking stage.
     */ 
     private:
     MeanShiftDetector det;
     KalmanNN track;
     KModel kmodel;
     std::string name {"MS-KNN"};

     public:
     MS_KNN(Eigen::MatrixXd params, KModel k_model) 
          : det(params(0, 0), params(0,1), params(0,2), params(0,3)), track(k_model, 15, 10.0), kmodel {k_model} {
          /*
          Constructor.

          Args:
               params: An Eigen matrix holding all the simple parameters. Each row will
                       correspond to a different point on the LROC curve.
                       params(0,0): eps
                       params(0,1): prune
                       params(0,2): iterations
                       params(0,3): characteristicTime
               k_model: A struct defining the transition and noise matrices for the tracker.

          Ret:
               None
          */
     }

     ~MS_KNN() {}


     void operator() (double * events, int N) {
          /*
          Feed the detector and tracker a set of events and have it update the tracks,

          Args:
               events: A pointer to memory containing the events. The data should have the format
                       [txyp],[txyp],[txyp],...; where t, x, y, and p are the time, x-,y-position
                       and event polarity.
               N: Number of events.

          Ret:
               None. This function only affects the state of the algorithm class.
          */ 
          Eigen::MatrixXd detections {det.processSensorData(events, N)};
          track.update(detections);
     }


     Eigen::MatrixXd currentTracks() {
          /*
          A wrapper for the Tracker function.
          
          Args:
               None

          Ret:
               An Eigen matrix containing the current object locations.
          */ 
          return track.currentTracks();
     }


     void predict() {
          /*
          A wrapper for the tracker function 'predict'.

          Args:
               None

          Ret:
               None
          */
          track.predict();
     }

     
     std::string algoName() {
          /*
          Get a string descriptor for the class.

          Args:
               None

          Ret:
               A string describing the class composition.
          */
          return name;
     }
};



class DBSCAN_GMPHD : public Algo {
     /*
     This class implements a detection and tracking algorithm
     using DBSCAN clustering for detection and a Gaussian
     Mixture Probability Hypothesis Density filter for tracking.
     */
     private:
     DbscanDetector det;
     gmphdFilter track; 
     std::string name {"DBSCAN-GMPHD"};

     public:
     DBSCAN_GMPHD(Eigen::MatrixXd params, phdModel model) 
          : det(params(0,0), params(0,1), params(0,2)), track(model) {
          /*
          Constructor.

          Args:
               params: An Eigen matrix holding all the simple parameters. Each row will
                       correspond to a different point on the LROC curve.
                       params(0,0): eps
                       params(0,1): num_pts
                       params(0,2): characteristicTime
               model: A phdModel object containing the parameters necessary to initialize
                      the GMPHD tracker. 
          Ret:
               None
          */

     }

     ~DBSCAN_GMPHD() {}


     void operator() (double * events, int N) {
          /*
          Feed the detector and tracker a set of events and have it update the tracks,

          Args:
               events: A pointer to memory containing the events. The data should have the format
                       [txyp],[txyp],[txyp],...; where t, x, y, and p are the time, x-,y-position
                       and event polarity.
               N: Number of events.

          Ret:
               None. This function only affects the state of the algorithm class.
          */ 
          Eigen::MatrixXd detections {det.processSensorData(events, N)};
          track.update(detections);
          return;
     }


     Eigen::MatrixXd currentTracks() {
          /*
          A wrapper for the tracker function of the same name.

          Args:
               None

          Ret:
               A Eigen matrix containing the positions of currently tracked 
               objects (as rows).
          */
          return track.currentTracks();
     }


     void predict() {
          /*
          A wrapper for the tracker function 'predict'.

          Args:
               None

          Ret:
               None
          */
          track.predict();
          return;
     }


     std::string algoName() {
          /*
          Get a string descriptor for the class.

          Args:
               None
          
          Ret:
               A string describing the class composition.
          */
          return name;
     }
};


class ECM_SCPHD {// : public Algo {
     /*
     This class implements detection and tracking using my forward model for the
     event camera and a sequential monte-carlo based cardinality PHD filter with
     adaptive birth intensity.
     */
     private:
     
     std::string name {"ECM-SCPHD"};

     public:
     ECM_SCPHD() {
          /*
          Constructor.

          Args:
               ...

          Ret:
               None
          */

     }

     ~ECM_SCPHD() {}


//     Eigen::MatrixXd currentTracks() {
          /*

          */
//     }


     void predict() {
          /*
           A wrapper for the tracker function 'predict'.

          Args:
               None

          Ret:
               None
          */
          return;
     }


     std::string algoName() const {
          /*
          Get a string descriptor for the class.

          Args:
               None
          
          Ret:
               A string describing the class composition.
          */
          return name;
     }
};
