// File        testBase.cpp
// Summary     Test that the base classes are implemented properly.
// Author      Jacob Trzaska

// Standard imports
# include <iostream>

// Other imports
# include "../base-classes.hpp"


class testerD: public Detector {
     private:
          double x = 5;

     public:
          testerD() {}
          ~testerD() {}
          Eigen::MatrixXd processSensorData(Eigen::MatrixXd eventStream) override {
               return Eigen::MatrixXd::Zero(4,4);
          }
};


class testerT: public Tracker {
     private:
          double x = 5;

     public:
          testerT() {}
          ~testerT() {}
          Eigen::MatrixXd currentTracks() override {
               return Eigen::MatrixXd::Ones(4,4);
          }
};





int main(int argc, char ** argv) {

     testerD td;
     testerT tt;

     std::cout << td.processSensorData(Eigen::MatrixXd::Zero(2,2)) << std::endl;
     std::cout << tt.currentTracks() << std::endl;

     return 0;
}
