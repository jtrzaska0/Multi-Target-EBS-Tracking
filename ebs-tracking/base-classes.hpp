// File        base-classes.hpp
// Summary     Abstract base classes for Detector and Tracker objects.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <string>

// Other imports
# include <../eigen3/Eigen/Dense>

class Detector {
     public:
     Detector() {}
     virtual ~Detector() {};
     virtual Eigen::MatrixXd processSensorData(double * eventStream, int N) = 0;
};


class Tracker {
     public:
     Tracker() {};
     virtual ~Tracker() {};
     virtual Eigen::MatrixXd currentTracks() = 0;
};


class Algo {
     public: 
     virtual void operator() (double * events, int N) = 0;
     virtual void predict() = 0;
     virtual std::string algoName() = 0;
     virtual Eigen::MatrixXd currentTracks() = 0;
};

template <typename U>
concept AlgoImpl = std::is_base_of<Algo, U>::value;
