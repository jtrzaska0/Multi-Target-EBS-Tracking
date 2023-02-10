// File        kalman.hpp
// Summary     Header file for Kalman filter class.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <vector>

// Other imports
# include <../eigen3/Eigen/Dense>
# include "base-classes.hpp"


class kalmanFilter {
     /*
     Implementation of the Kalman filter.
     */
     private:
     Eigen::MatrixXd xt;
     Eigen::MatrixXd xt1;
     Eigen::MatrixXd yt;
     Eigen::MatrixXd Pt;
     Eigen::MatrixXd Pt1;
     Eigen::MatrixXd T;
     Eigen::MatrixXd M;
     Eigen::MatrixXd Q;
     Eigen::MatrixXd R;
     Eigen::MatrixXd S;
     Eigen::MatrixXd K;

     public:
     // The Kalman-NN class refuses to compile without a default constructor.
     kalmanFilter() {}

     kalmanFilter(Eigen::MatrixXd x0, Eigen::MatrixXd P0, Eigen::MatrixXd T0, 
                  Eigen::MatrixXd M0, Eigen::MatrixXd Q0, Eigen::MatrixXd R0) {
          /*
          Constructor.

          Args:

               x0: Initial state estimate, [x(t0)].
               P0: Initial estimate of error covariance, [P(t)].
               T0: System transition matrix, [x(t+1) = T(t)*x(t) + u(t)].
               M0: System measurement matrix, [y(t) = M(t) * x(t) + w(t)].
               Q0: Process noise covariance, [Q(t) = E{u(t)*u(t)'}].
               R0: Measurement noise covariance, [R(t) = E{w(t)*w(t)'}].

               Note that all inputs must be Eigen matrices with compatible dimensions.

          Ret:
               None.
          */
          // Check that all the matrices are compatible and set memory.
          if (x0.rows() != T0.cols()) {
               std::cout << "Error! State vector and transition matrix have incompatible dimensions.\n";
               exit(EXIT_SUCCESS);
          }

          // State vectors.
          xt  = x0;
          xt1 = x0;
          yt  = Eigen::MatrixXd::Zero(M0.rows(), 1);
          // Error covariance matrices.
          Pt  = P0;
          Pt1 = P0;
          // System Matrices
          T = T0;
          M = M0;
          Q = Q0;
          R = R0;
          S = Eigen::MatrixXd::Zero(M0.rows(), M0.rows());
          K = Eigen::MatrixXd::Zero(P0.rows(), M0.rows());
          // Evaluate x(t+1|t).
          predict();
     }

     ~kalmanFilter() {}

     void update(Eigen::MatrixXd y) {
          // Use the new measurement, y, to estimate x(t+1|t+1) 
          // and P(t+1|t+1).
          xt = xt1 + K * (y - yt);
          Pt1 = Pt - K * (M * Pt);
          return;
     }


     void predict() {
          // Use x(t|t) and P(t|t) to calculate x(t+1|t) and P(t+1|t).
          xt1 = T * xt;
          Pt1 = T * Pt * T.transpose() + Q;
          // Estimate the next value of y: y(t+1|t).
          yt = M * xt1;
          // Compate S and K for use in the update step.
          S = M * Pt1 * M.transpose() + R;
          K = Pt1 * M.transpose() * S.inverse();
          return;
     }

     
     void autoUpdate() {
          /*
          Use x(t+1|t) as y(t+1) to estimate x(t+1|t+1) and P(t+1|t+1).

          Args:
               None

          Ret:
               None
          */
          xt = xt1 + K * (xt1(Eigen::seq(0,1), 0) - yt);
          Pt1 = Pt - K * (M * Pt);
          return;
     }
     
     void printInfo(int ID) const {
          /*
          For debugging. Prints parameters from the class to console.

          Args:
               ID: Integer label for the track.

          Ret:
               None
          */
          if (ID == 2) {
               std::cout << "xt1.cols = " << xt1.cols() << "   xt1.rows = " << xt1.rows() << "\n";
               std::cout << "yt.cols  = " << yt.cols() << "   yt.rows  = " << yt.rows() << "\n";
               std::cout << "xt.cols  = " << xt.cols() << "   xt.rows  = " << xt.rows() << "\n";
               std::cout << std::endl;
          }
     }

     Eigen::MatrixXd getCurrState() const {
          return xt;
     }

     Eigen::MatrixXd getPredState() const {
          return xt1;
     }

};
