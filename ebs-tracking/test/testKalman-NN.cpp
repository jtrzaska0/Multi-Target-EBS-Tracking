# include <iostream>
# include "../Kalman-NN.hpp"

using Eigen::MatrixXd;


int main() {
     std::cout << "Testing class instantiation.\n";

     KModel model;
     model.dt=5;
     model.P = MatrixXd::Zero(2,2);
     model.F = MatrixXd::Zero(2,2);
     model.Q = MatrixXd::Zero(2,2);
     model.R = MatrixXd::Zero(2,2);
     model.H = MatrixXd::Zero(2,2);

     KalmanNN(model, 15, 2.);

     std::cout << "Instantiation successful!" << std::endl;
     return 0;
}
