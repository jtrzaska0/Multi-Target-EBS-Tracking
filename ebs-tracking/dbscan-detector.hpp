// File        dbscan-detector.hpp
// Summary     DBSCAN-based detector for event data.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <vector>
# include <set>
# include <map>
# include <algorithm>
# include <iterator>

// Other imports
# include "base-classes.hpp"
# include <mlpack.hpp>
# include <mlpack/core.hpp>
# include <mlpack/methods/dbscan.hpp>
# include <armadillo>

class DbscanDetector : public Detector {
     private:
     double tau;
     mlpack::DBSCAN<> dbscan;

     public:
     DbscanDetector(double eps, int num_pts, double characteristicTime) 
          : dbscan(eps, num_pts), tau {characteristicTime} {
          /*
          Constructor.

          Args:
               radius: Search radius for forming the clusters.
               pts: The number of points required to declare a core point.
               characterisitcTime: Time to scale events to.

          Ret:
               None
          */
     }

     ~DbscanDetector() {}


     Eigen::MatrixXd processSensorData(double * events, int N) override {
          /*
          Read in sensor data and cluster the events.

          Args:
               events: An array containing the event data for the most recent
                       integration period. The array should be a set of packed events
                       {[t1, x1, y1, p1], [t2, x2, y2, p2], ..., [tN, xN, yN, pN]}.
                       All the data is assumed to be double-precision.
               N: The number of events reported.

          Ret:
               Centroids of possibly interesting objects.
          */
          // Scale timestamps.
          processTimestamps(events, N);

          // Cluster events. Aramdillo stores each event as a column. Hence the number of
          // rows in Events is equal to the dimension of an event vector, and the number of
          // columns equals the number of events.
          arma::mat Events(events, 4, N);
          arma::mat Centroids;
          size_t n {dbscan.Cluster(Events, Centroids)}; 

          // Convert Armadillo matrix to Eigen matrix. The centroids in Centroids are stored as columns.
          Eigen::MatrixXd centroids {Eigen::MatrixXd::Zero(Centroids.n_cols, Centroids.n_rows)};
          for (int i {0}; i < Centroids.n_cols; ++i)
               for (int j {0}; j < Centroids.n_rows; ++j)
                    centroids(i, j) = Centroids(j, i);

          return centroids(Eigen::all, Eigen::seq(1,2));
     }


     private:
     void processTimestamps(double * events, int N) {
          /*
          Convert the timestamps into units of a pre-determined characteristic time.

          Args:
               events: Event data.
               N: The number of events reported.

          Ret:
               The event data with scaled timestamps.
          */
          for (int i {0}; i < N; ++i)
               events[4 * i] = events[4 * i] / tau;

          return;
     }

};
