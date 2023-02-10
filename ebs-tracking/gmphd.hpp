// File        gmphd.hpp
// Summary     Implementation of the Gaussian Mixture Probability Hypothesis Density filter.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <vector>
# include <string>
# include <set>
# include <algorithm>
# include <cmath>

// Other imports
# include <../eigen3/Eigen/Dense>
# include "base-classes.hpp"


class Gaussian {
     /*
     Gaussian component object. This class encapsulates the properties
     of the mixture component including
          (1) mean,
          (2) covariance,
          (3) weight (amplitude).
     */
     public:
     double w; // Weight
     Eigen::MatrixXd m; // Mean
     Eigen::MatrixXd P; // Covariance

     public:
     Gaussian(double weight, Eigen::MatrixXd mean, Eigen::MatrixXd cov) {
          /*
          Constructor.
           
          Args:
               weight (double): Scalar quantity descirbing the Gaussian's amplitude.
               mean (Matrix): Column vector locating the Gaussian's mean.
               cov (Matrix): Gaussian's covariance matrix.
           
          Ret:
               None
          */
          if (weight < 0) {
               std::cout << "Error! Weight must be non-negative.\n";
               throw;
          }

          if (mean.cols() != 1) {
               std::cout << "Error! Mean vectors must be single-column.\n";
               throw;
          } 

          if (cov.rows() != cov.cols()) {
               std::cout << "Error! Covariance matrix must be square.\n";
               throw;
          }

          w = weight;
          m = mean;
          P = cov;
     }

     ~Gaussian() {}


     double eval(Eigen::MatrixXd x) const {
          /*
          Evaluate the Gaussian at position x.
           
          Args:
               x (Eigen::MatrixXd): Position matrix.
           
          Ret:
               Scalar result of Gaussian.
          */
          double s = ( (x-m) * P * (x-m) )(0,0);
          return w * exp(-0.5 * s * s) / sqrt(2 * M_PI * P.determinant());
     }
};


struct phdModel {
     // Size of time step.
     double dt;
     // System matrices.
     Eigen::MatrixXd F;
     Eigen::MatrixXd Q;
     Eigen::MatrixXd H;
     Eigen::MatrixXd R;
     // Pruning parameters
     double T;
     double U;
     double Jmax;
     // Spawning paramters
     std::vector<Eigen::MatrixXd> F_spawn;
     std::vector<double> w_spawn;
     std::vector<Eigen::MatrixXd> d_spawn;
     std::vector<Eigen::MatrixXd> Q_spawn;
     // Survival and detection probabilities.
     double pS;
     double pD;
     // Clutter function.
     double clutterIntensity;
     // Initial mixutre.
     std::vector<Gaussian> initMix;
};


class gmphdFilter : public Tracker {
     /*
     This class implements the Gaussian Mixture Probability Hypothesis Density filter.
     For more information, and a complete derivation of the filter, see the paper by
     Vo and Ma: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1710358&tag=1
     */
     private:
     // Transition and measurement matrices.
     // x(t+1) = F * x(t) + n(t); n(t) * n(t)' = Q.
     // z(t) = H * x(t) + v(t); v(t) * v(t)' = R.
     Eigen::MatrixXd F;
     Eigen::MatrixXd Q;
     Eigen::MatrixXd H;
     Eigen::MatrixXd R;
     // Pruning and merging parameters.
     // See the paper by Vo and Ma for more information.
     double T;
     double U;
     double J;
     // Spawning parameters.
     std::vector<double> w_spawn;
     std::vector<Eigen::MatrixXd> F_spawn;
     std::vector<Eigen::MatrixXd> d_spawn;
     std::vector<Eigen::MatrixXd> Q_spawn;
     // Current intensity mixture.
     std::vector<Gaussian> mix;
     // Survival and detection probabilities.
     double pS;
     double pD;
     // Clutter function.
     double clutterIntensity;

     public:
     gmphdFilter(phdModel model) {
          /*
          Constructor. Set the motion and measurement model.
          
          Args:
               model (phdModel): Contains aall relevant target and 
                              measurement system information. See the phdModel
                              class for more details.
                    
          Ret:
               None
          */
          // Set the system parameters.
          F = model.F;
          Q = model.Q;
          H = model.H;
          R = model.R;
          // Set the merging and pruning parameters.
          T = model.T;
          U = model.U;
          J = model.Jmax;
          // Set spawning parameters.
          w_spawn = model.w_spawn;
          F_spawn = model.F_spawn;
          d_spawn = model.d_spawn;
          Q_spawn = model.Q_spawn;
          // Set initial Gaussian mixture.
          mix = model.initMix;
          // Set survival and detection probabilities.
          pS = model.pS;
          pD = model.pD;
          // Set the clutter rate.
          clutterIntensity = model.clutterIntensity;
     }

     ~gmphdFilter() {}


     int update(Eigen::MatrixXd Z) {
          /*
          Use the latest measurements to update the filter tracks.

          Args:
               Z (Matrix): Measurement set (NxM matrix). Each measurement is stored
                           as a row in the matrix. The matrix hold N measurements.

          Ret:
               None
          */
          // Name some alias variables.
          std::vector<Eigen::MatrixXd> eta;
          std::vector<Eigen::MatrixXd> K;
          std::vector<Eigen::MatrixXd> S;
          std::vector<Eigen::MatrixXd> P;

          long unsigned int Jc {static_cast<unsigned long>(mix.size())};
          std::vector<Gaussian> newMix;

          // 1. Calculate the update components.
          for (long unsigned int j {0}; j < Jc; ++j) {
               eta.push_back(H * mix[j].m);
               S.push_back(R + H * mix[j].P * H.transpose());
               K.push_back(mix[j].P * H.transpose() * S[j].inverse());
               P.push_back(mix[j].P - K[j] * H * mix[j].P);
          }

          // 2. Update.
          // Update the starting mixture.
          for (long unsigned int j {0}; j < Jc; ++j) {
               double w {(1 - pD) * mix[j].w};
               Eigen::MatrixXd m {mix[j].m};
               Eigen::MatrixXd p {mix[j].P};
               Gaussian q(w, m, p);
               newMix.push_back(q);
          }

          // Use the measurements to update the remaining components.
          int l {0};
          for (int i {0}; i < Z.rows(); ++i) {
               Eigen::MatrixXd z {Z.row(i).reshaped(Z.cols(), 1)};
               l = l + 1;
               double sw {0};
               for (long unsigned int j {0}; j < Jc; ++j) {
                    double s {( (z-eta[j]).transpose() * S[j].inverse() * (z-eta[j]) )(0,0)};
                    double v {exp(-0.5 * s * s) / sqrt(2 * M_PI) * sqrt(S[j].determinant())};
                    double w {pD * mix[j].w * v};
                    Eigen::MatrixXd m {mix[j].m + K[j] * (z - eta[j])};
                    Eigen::MatrixXd p {P[j]};
                    Gaussian q(w, m, p);
                    newMix.push_back(q);
                    sw = sw + w;
               }

               for (long unsigned int j {0}; j < Jc; ++j) {
                    newMix[l * Jc + j].w = (newMix[l * Jc + j]).w / (clutterIntensity + sw);
               }
          }

          // 3. Assign new mixture to tracks.
          mix = newMix;

          // 4. Remove weak (w < T) Gaussian components.
          prune();

          // Status indicator '0'. Function executed successfully.
          return 0;
     }


     int predict(std::vector<Gaussian> birthMix = std::vector<Gaussian>()) {
          /*
          Propagate the intensity in time and predict the Gausian mixture at the
          next time step.

          Args:
               birthMix: Vector of Gaussian components.

          Ret:
               None.
          */
          // Alias the new mixture.
          std::vector<Gaussian> newMix;
          
          int Jb = birthMix.size();
          int Jc = mix.size();
          int Js = w_spawn.size();

          // 1. Run prediction for birth targets first.
          for (int j{0}; j< Jb; ++j) {
               double w = birthMix[j].w;
               Eigen::MatrixXd m = birthMix[j].m;
               Eigen::MatrixXd p = birthMix[j].P;
               Gaussian q(w, m, p);
               newMix.push_back(q);
          }

          for (int j {0}; j < Js; ++j) {
               for (int l {0}; l < Jc; ++l) {
                    double w = mix[l].w * w_spawn[j];
                    Eigen::MatrixXd m = d_spawn[j] + F_spawn[j] * mix[l].m;
                    Eigen::MatrixXd p = Q_spawn[j] + F_spawn[j] * mix[l].P * F_spawn[j].transpose();
                    Gaussian q(w, m, p);
                    newMix.push_back(q);
               }
          }

          // 2. Run prediction for existing targets.
          for (int j{0}; j< Jc; ++j) {
               double w = pS * mix[j].w;
               Eigen::MatrixXd m = F * mix[j].m;
               Eigen::MatrixXd p = Q + F * mix[j].P * F.transpose();
               Gaussian q(w, m, p);
               newMix.push_back(q);
          }

          // Assign tracking mixture to newly predicted mixture.
          mix = newMix;             

          // Return status indiciator '0'. Function executed successfully.
          return 0;
     }


     private:
     int prune() {
          /*
          Remove any Gaussian components that do not meet the threshold (T)
          on its weight. Merge any remaining components that are close together.

          Args:
               None

          Ret:
               None
          */
          // Check how many Gaussian are present.
          int numGauss = mix.size();
          if (numGauss == 0)
               return 0;

          // 1. Move all components above threshold to a separate set.
          std::set<int> I;
          Eigen::MatrixXd weights = Eigen::MatrixXd::Ones(1, numGauss);
          for (int n {0}; n < numGauss; ++n) {
               weights(0, n) = mix[n].w;
               if (mix[n].w >= T)
                    I.insert(n);
          }

          // Storage for the pruned mixture.
          std::vector<Gaussian> prunedMix;

          // 2. Iteratively merge components. 
          int l = 0;
          while (true) {
               // Check for out-of-bounds indices.
               l = l + 1;
               if ((I.size() == 0) || l > J)
                    break;

               // Collect the significant weights.
               Eigen::MatrixXi A = Eigen::MatrixXi::Zero(1, I.size());
               Eigen::MatrixXd W = Eigen::MatrixXd::Zero(1, I.size());
               int k = 0;
               for (auto& i : I) {
                    A(1, k) = i;
                    W(1, k) = weights(1, i);
                    ++k;
               }

               std::ptrdiff_t r, c;
               A.maxCoeff(&r, &c);
               int idx = A(0, (int)c);

               // Find all indices within range of the current max component.
               std::set<int> L;
               for (auto& i : I) {
                    Eigen::MatrixXd mDiff = mix[i].m - mix[idx].m;
                    double dist = (mDiff * mix[i].P.inverse() * mDiff)(0,0);
                    if (fabs(dist) <= U)
                         L.insert(i);
               }     

               //// Combine components with indices in L.
               // Calculate the component weight.
               double w = 0;
               for (auto& i : L) {
                    w = w + mix[i].w;
               }

               // Calculate the new component mean.
               Eigen::MatrixXd m = Eigen::MatrixXd::Zero(mix[idx].m.rows(), mix[idx].m.cols());
               for (auto& i : L) {
                    m = m + mix[i].w * mix[i].m;
               }
               m = m / w;

               // Calculate the new component covariance matrix.
               Eigen::MatrixXd p = Eigen::MatrixXd::Zero(mix[idx].P.rows(), mix[idx].P.cols());
               for (auto& i : L) {
                    Eigen::MatrixXd mDiff = m - mix[i].m;
                    p = p + mix[i].w * (mix[i].P + mDiff * mDiff.transpose());
               }
               p = p / w;

               // Update the pruned mixutre.
               Gaussian q(w, m, p);
               prunedMix.push_back(q);

               // Remove the indices in L from I.
               std::set<int> sdiff;
               std::set_difference(I.begin(), I.end(), L.begin(), L.end(), std::inserter(sdiff, sdiff.end()));
               I = sdiff;
          }

          mix = prunedMix;
         
          return 0; 
     }


     public:
     Eigen::MatrixXd currentTracks() {
          /*
          Return a copy of the current Gaussian mixture to the caller.

          Args:
               None

          Ret:
               Mean parameters for current tracks (as a vector of Eigen matrices).
          */

          // Report any component with a weight greater than a threshold.
          long unsigned int j {static_cast<unsigned long>(mix.size())};
          std::vector<long unsigned int> jList;
          for (long unsigned int j {0}; j < J; ++j)
               if (mix[j].w >= 0.4)
                    jList.push_back(j);

          Eigen::MatrixXd X {Eigen::MatrixXd::Zero(jList.size(), 2)};
          long unsigned int k {0};
          for (auto j : jList) {
               X.row(k) = (mix[j].m)(Eigen::seq(0,1), 0).reshaped(1,2);
               ++k;
          }

          return X;
     }

};
