// File        meanshift-detector.hpp
// Summary     MeanShift-based detector for event data.
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
# include <mlpack/methods/mean_shift.hpp>
# include <armadillo>


class MeanShiftDetector : public Detector {
     /*
     This class defines a detector based on the Mean Shift clustering algorithm.
     */
     private:
     double tau;
     int pruneTh;
     mlpack::MeanShift<> ms;

     public:
     MeanShiftDetector(double eps, int prune, int iterations, double characteristicTime) 
          : ms(eps, iterations), tau {characteristicTime}, pruneTh {prune} {
          /*
          Constructor.

          Args:
               radius: Kernel width.
               prune: Minimum number of points a cluster must have.
               iterations: Maximum number of iterations to cluster.
               characterisitcTime: Time to scale events to.

          Ret:
               None
          */
     }

     ~MeanShiftDetector() {}


     Eigen::MatrixXd processSensorData(double * events, int N) override {
          /*
          Read in sensor data and cluster the events.

          Args:
               events: An array containing the event data for the most recent
                       integration period. The array should be a set of packed events
                       {[x1, y1, t1, p1], [x2, y2, t2, p2], ..., [xN, yN, tN, pN]}.
                       All the data is assumed to be double-precision.
               N: The number of events reported.

          Ret:
               Centroids of possibly interesting objects.
          */
          // Scale timestamps.
          processTimestamps(events, N);

          // Cluster events. Aramdillo stores each event as a column. Hence the number of
          // rows of Events is equal to the dimension of an event vector, and the number of
          // columns equals the number of events.
          arma::mat Events(events, 4, N);
          arma::Row<size_t> Assignments;
          arma::mat Centroids;
          ms.Cluster(Events, Assignments, Centroids); 

          // Prune small clusters.
          size_t nClusters {Centroids.n_cols};
          std::vector<size_t> a(nClusters);
          for (int i {0}; i < Assignments.n_cols; ++i)
               a[Assignments(i)] += 1;

          std::vector<size_t> clIdx;
          for (int i {0}; i < nClusters; ++i)
               if (a[i] > pruneTh)
                    clIdx.push_back(i);

          // Convert Armadillo matrix to Eigen matrix. The centroids in Centroids are stored as columns.
          Eigen::MatrixXd centroids {Eigen::MatrixXd::Zero(clIdx.size(), Centroids.n_rows)};
          int k {0};
          for (auto i : clIdx) {
               for (int j {0}; j < Centroids.n_rows; ++j)
                    centroids(k, j) = Centroids(j, i);
               ++k;
          }

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
