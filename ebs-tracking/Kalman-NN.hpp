// File        Kalman-NN.hpp
// Summary     Target tracking using a bank of Kalman filters nearest-neighbor assocation.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <vector>
# include <map>
# include <set>
# include <algorithm>
# include <functional>

// Other imports
# include "kalman.hpp"

Eigen::MatrixXd cdist(Eigen::MatrixXd A, Eigen::MatrixXd B) {
     /*
     This function is based off the SciPy method of the 
     same name.

     Args:
          A: First matrix. Vectors stored as rows.
          B. Second matrix. Vectors stored as rows.

     Ret:
          Eigen matrix whose elements are the euclidean 2-norm between
          each of the vector
     */
     Eigen::MatrixXd D {Eigen::MatrixXd::Zero(A.rows(), B.rows())};

     for (int i {0}; i < A.rows(); ++i)
          for (int j {0}; j < B.rows(); ++j) {
               Eigen::ArrayXXd arr { (A.row(i) - B.row(j)).array() }; 
               D(i, j) = (arr * arr).sum();
          }

     return D;
}


struct KModel {
     // Size of time step.
     double dt;
     // System matrices.
     Eigen::MatrixXd P;
     Eigen::MatrixXd F;
     Eigen::MatrixXd Q;
     Eigen::MatrixXd H;
     Eigen::MatrixXd R;
};


class KalmanNN : public Tracker {
     /*
     This class implements a target tracker using a set of Kalman filters
     and the nearest neighbor algorithm to associate measurements to tracks.
     The basis for this implemetation was the code found at the webpage
     https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv.
     */
     private:
     // Define the system parameters.
     double dt;
     Eigen::MatrixXd P;
     Eigen::MatrixXd F;
     Eigen::MatrixXd Q;
     Eigen::MatrixXd H;
     Eigen::MatrixXd R;
     // Keep a list of target IDs.
     unsigned long int nextID {0};
     // Ordered dictionary of target information.
     std::map<int, Eigen::MatrixXd> targets;
     // List of filters for each targets.
     std::map<int, kalmanFilter> filters;
     // Number of frames each target has been missing.
     std::map<int, int> numFramesLost;
     // Set thresholds.
     int maxNumFramesLost;
     double distThreshold;

     public:
     KalmanNN(KModel model, int mNumFrLost,  double distTh) : nextID {0} {
          /*
          Constructor.

          Args:
               KModel: Struct containing relevant system matrices and other
                       parameters.
               mNumFrLost: Max number of frames missing.
               distTh: Maximum distance a detection may be from an already registered object
                       to be considered as from that object.

          Ret:
               None
          */
          // Load system parameters.
          dt = model.dt;
           P = model.P;
           F = model.F;
           Q = model.Q;
           H = model.H;
           R = model.R;
          
          // Set thresholds.
          maxNumFramesLost = mNumFrLost;
          distThreshold = distTh;
     }

     ~KalmanNN() {}


     void update(Eigen::MatrixXd meas) {
          /*
          Use the most recent set of measurements to update the target states.

          Args:
               meas: Each row is a single datum (vector).

          Ret:
               None. This function purely updates the state of the tracker.
          */
          // Check if 'meas' is empty.
          if (meas.rows() == 0) {
               // Deregister objects if they are lost.
               for (std::map<int, int>::iterator targ = numFramesLost.begin(); targ != numFramesLost.end(); ++targ) {
                    int ID {targ->first};
                    ++numFramesLost[ID];
                    if (numFramesLost[ID] > maxNumFramesLost)
                         deRegister(ID);
                    else {
                         // Evolve the targets in time. Use the predicted state as the measurement.
                         filters[ID].autoUpdate();
                         targets[ID] = (filters[ID].getCurrState())({0,1}, 0);
                    }
               }

               return;
          }

          // Track new objects if none are currently tracked.
          if (targets.size() == 0) {
               for (int i {0}; i < meas.rows(); ++i)
                    Register(meas(i, Eigen::all).reshaped(1,2));
          } else {
               // Make a vector of active target IDs and of target centroids.
               std::vector<int> targIDs(filters.size());
               Eigen::MatrixXd targCentroids {Eigen::MatrixXd::Zero(filters.size(), 2)};
               int k = 0;
               for (std::map<int, Eigen::MatrixXd>::iterator iter = targets.begin(); iter != targets.end(); ++iter) {
                    targIDs[k] = iter->first;
                    targCentroids(k, Eigen::all) = (iter->second).reshaped(1, 2);
                    ++k;
               }

               // Evaluate pairwise distances for each existing target and detection.
               Eigen::MatrixXd D {cdist(targCentroids, meas)};
               // Find the minimum element in each row and sort the row indices by their min values.
               std::map<double, int> rmap;
               for (int i {0}; i < D.rows(); ++i)
                    rmap[D.row(i).minCoeff()] = i;

               // Keep track of which rows and columns have been associated.
               std::set<int> activeRows;
               std::set<int> activeCols;
               
               for (int l {0}; l < D.rows(); ++l)
                    activeRows.insert(l);
               for (int l {0}; l < D.cols(); ++l)
                    activeCols.insert(l);

               if (D.rows() >= D.cols()) {
                    // For each row index, check if the nearest detection is within threshold.
                    // If so, update that row. Otherwise, do nothing and update the filter using
                    // predicted location.
                    for (auto iter {rmap.begin()}; iter != rmap.end(); ++iter) {
                         // Get row index
                         int ridx {iter->second};
                         // Break from the loop if all the columns have been exhausted.
                         if (activeCols.size() == 0)
                              break;

                         // Find the nearest detection.
                         std::ptrdiff_t I, J;
                         std::vector<int> ac(activeCols.size());
                         std::copy(activeCols.begin(), activeCols.end(), ac.begin());

                         D(ridx, ac).minCoeff(&I, &J);
                         int cidx {ac[J]};

                         // Update current target.
                         int ID {targIDs[ridx]};
                         if (D(ridx, cidx) < distThreshold * distThreshold) {
                              numFramesLost[ID] = 0;
                              filters[ID].update(meas(cidx, Eigen::all).reshaped(2, 1));
                              targets[ID] = (filters[ID].getCurrState())({0,1}, 0);
                         } else {
                              ++numFramesLost[ID];
                              // Deregister missing targets.
                              if (numFramesLost[ID] > maxNumFramesLost) {
                                   deRegister(ID);
                              } else {
                                   filters[ID].autoUpdate();
                                   targets[ID] = (filters[ID].getCurrState())({0,1}, 0);
                              }
                         }

                         // Eliminate used rows and columns.
                         std::set<int> resR;
                         std::set<int> resC;
                         std::set<int> R {ridx};
                         std::set<int> C {cidx};

                         std::set_difference(activeRows.begin(), activeRows.end(), R.begin(), R.end(), std::inserter(resR, resR.end()));
                         activeRows = resR;
                         std::set_difference(activeCols.begin(), activeCols.end(), C.begin(), C.end(), std::inserter(resC, resC.end()));
                         activeCols = resC;
                    }
                   
                    // Update the remaining targets and their predicted states. 
                    for (auto& ridx : activeRows) {
                         int ID {targIDs[ridx]};
                         ++numFramesLost[ID];
                         if (numFramesLost[ID] > maxNumFramesLost) {
                              deRegister(ID);
                         } else {
                              filters[ID].autoUpdate();
                              targets[ID] = (filters[ID].getCurrState())({0,1}, 0);
                         }
                    }

               } else {
                    // There are more detections than there are active tracks, i.e., D.cols() > D.rows().
                    // For each row index, check if the nearest detection is within threshold. If so,
                    // update that row. Otherwise, check miss status and update the filter using the predicted state.
                    for (auto iter = rmap.begin(); iter != rmap.end(); ++iter) {
                         // Get row index
                         int ridx {iter->second}; 
                         // Break from the loop if all the rows have been exhausted.
                         if (activeRows.size() == 0)
                              break;

                         // Find the nearest detection.
                         std::ptrdiff_t I, J;
                         std::vector<int> ac(activeCols.size());
                         std::copy(activeCols.begin(), activeCols.end(), ac.begin());

                         D(ridx, ac).minCoeff(&I, &J);
                         int cidx {ac[J]};

                         // Update current target.
                         int ID {targIDs[ridx]};
                         if (D(ridx, cidx) < distThreshold * distThreshold) {
                              numFramesLost[ID] = 0;
                              filters[ID].update(meas(cidx, Eigen::all).reshaped(2,1));
                              targets[ID] = (filters[ID].getCurrState())({0,1}, 0);
                         } else {
                              ++numFramesLost[ID];
                              // Deregister if missing.
                              if (numFramesLost[ID] > maxNumFramesLost) {
                                   deRegister(ID);
                              } else {
                                   filters[ID].autoUpdate();
                                   targets[ID] = (filters[ID].getCurrState())({0,1}, 0);
                              }
                         }

                         // Eliminate used rows and columns.
                         std::set<int> resR;
                         std::set<int> resC;
                         std::set<int> R {ridx};
                         std::set<int> C {cidx};

                         std::set_difference(activeRows.begin(), activeRows.end(), R.begin(), R.end(), std::inserter(resR, resR.end()));
                         activeRows = resR;
                         std::set_difference(activeCols.begin(), activeCols.end(), C.begin(), C.end(), std::inserter(resC, resC.end()));
                         activeCols = resC;
                    }

                    // Use the remaining detections to initialize new tracks.
                    for (auto& cidx : activeCols) {
                         Register(meas(cidx, Eigen::all).reshaped(1, 2));
                    }
               }
          }

          return;
     }


     void predict() {
          /*
          Predict the target locations for the next time step.

          Args:
               None

          Ret:
               None
          */
          for (std::map<int, int>::iterator iter = numFramesLost.begin(); iter != numFramesLost.end(); ++iter)
               filters[iter->first].predict();

          return;
     }


     Eigen::MatrixXd currentTracks() override {
          /*
          Read out the current target positions.

          Args:
               None

          Ret:
               List of current objects tracked.
          */
          Eigen::MatrixXd targs = Eigen::MatrixXd::Zero(targets.size(), 2);

          int i = 0;
          for (std::map<int, Eigen::MatrixXd>::iterator iter = targets.begin(); iter != targets.end(); ++iter) {
               targs(i, Eigen::all) = (iter->second).reshaped(1, 2);
               ++i;
          }

          return targs;
     }


     private:
     void Register(Eigen::MatrixXd centroid) {
          /*
          Add a new target to the list of tracked objects.

          Args:
               centroid: Location of new target.

          Ret:
               None
          */
          // Update the target, filter, and numFrameLost maps. Then increment
          // the ID counter.
          targets[nextID] = centroid;
          Eigen::MatrixXd x {{centroid(0,0)}, {centroid(0,1)}, {0}, {0}};
          kalmanFilter K(x, P, F, H, Q, R);
          filters[nextID] = K;
          numFramesLost[nextID] = 0;
          ++nextID;
          return;
     }


     void deRegister(int ID) {
          /*
          Remove a target from the tracking list.

          Args:
               objKey: Dictionary key for object being removed.

          Ret:
               None
          */
          numFramesLost.erase(ID);
          targets.erase(ID);
          filters.erase(ID);
          return;
     }
};
