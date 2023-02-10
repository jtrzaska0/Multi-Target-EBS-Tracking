// File        dbgmphdLROC.cpp
// Summary     Generate LROC curves for the DBSCAN with GMPHD filter algorithm.
// Author      Jacob Trzaska

// Standard imports
# include <iostream>
# include <vector>
# include <string>
# include <cmath>

// Other imports
# include "../base-classes.hpp"
# include "../dbscan-detector.hpp"
# include "../gmphd.hpp"
# include "LROC.hpp"
# include "../Algorithm.hpp"


void prefixFiles(std::vector<std::pair<std::string, std::string>>& files, std::string prefix) {
     /*
     Throw the necessary file prefix onto the strings.

     Args:
          files: Set of filenames to be modified.
          prefix: The prefix to be added to the files.

     Ret:
          None
     */
     std::string temp;
     for (int i {0}; i < files.size(); ++i) {
          temp = prefix + files[i].first;
          files[i].first = temp;
          temp = prefix + files[i].second;
          files[i].second = temp;
     }
}


template <AlgoImpl T, typename U> 
void bgLROC(int choice, std::vector<std::pair<std::string, std::string>> files, std::string algoname, 
          std::map<std::string, Eigen::MatrixXd> dataframes, U extra) {
     /*
     This function takes an algorithm and chooses the appropriate event and ground truth files.

     Args:
          choice: An integer less than 4 describing which files to select.
          files: A collection of pairs of filenames, the first is the event data and the second is ground truth.
          algoname: The name of the algorithm object.
          dataframes: A set of matrices that contain simple parameters for initializing the algorithm.
          extra: Other parameters that are necessary to instantiate the algorithm.

     Ret:
          None.
     */
     // Set possible strings.
     std::string uniform1_3 {"../../../jacob_data/10vel_neg1motion_sbr1_3/"};
     std::string uniform2 {"../../../jacob_data/10vel_neg1motion_sbr2/"};
     std::string urban1_3 {"../../../jacob_data/10vel_neg1motion_sbr1_3_urban/"};
     std::string urban2 {"../../../jacob_data/10vel_neg1motion_sbr2_urban/"};

     // BG info string
     std::string bgInfo;

     switch (choice) {
     case 0: prefixFiles(files, uniform1_3); bgInfo = "Uniform1_3"; break;
     case 1: prefixFiles(files, uniform2); bgInfo = "Uniform2"; break;
     case 2: prefixFiles(files, urban1_3); bgInfo = "Urban1_3"; break;
     case 3: prefixFiles(files, urban2); bgInfo = "Urban2"; break;
     default: return;
     }

     // Ready inputs for LROC.run().
     double time {8};
     std::string outputTitle {"/home/jtrzaska/work/event-sensor/cpp_versions/lroc/out/" + bgInfo + "-" + algoname + ".dat"};
     std::vector<double> radii {12};//{5, 8, 10, 12, 15, 18};
 
     // Run LROC analysis.
     LROC<T, U> L(radii, files, dataframes);
     L.run(time, outputTitle, extra);
     return;
}


int main() {
     // Define the system matrices
     double DT {8};
     double p3 {pow(DT, 3) / 3};
     double p2 {pow(DT, 2) / 2};
     Eigen::MatrixXd P {{25, 0, 0, 0}, {0, 25, 0, 0}, {0, 0, 25, 0}, {0, 0, 0, 25}};
     Eigen::MatrixXd F {{1, 0, DT, 0}, {0, 1, 0, DT}, {0, 0, 1, 0}, {0, 0, 0, 1}};
     Eigen::MatrixXd Q {{p3, 0, p2, 0}, {0, p3, 0, p2}, {p2, 0, DT, 0}, {0, p2, 0, DT}};
     Eigen::MatrixXd H {{1, 0, 0, 0}, {0, 1, 0, 0}};
     Eigen::MatrixXd R {{7, 0}, {0, 7}};

     // Define the model.
     std::vector<Eigen::MatrixXd> Fs, ds, Qs;
     std::vector<double> ws;
     Eigen::MatrixXd mean {{211}, {62}, {0}, {0}};
     Eigen::MatrixXd gcov {{25, 0, 0, 0}, {0, 25, 0, 0}, {0, 0, 25, 0}, {0, 0, 0, 25}};
     Gaussian gauss0(1.0, mean, gcov);
     std::vector<Gaussian> mix {gauss0};
     phdModel k_model {.dt = DT, .F = F, .Q = Q, .H = H, .R = R, .T = 1e-4, .U = 12, .Jmax = 45, .F_spawn = Fs, .w_spawn = ws,
                         .d_spawn = ds, .Q_spawn = Qs, .pS = 0.98, .pD = 0.75, .clutterIntensity = 2e-7, .initMix = mix};

     // Create the file names.
     std::string relPathE {"runLinear_point0_run"};
     std::string relPathG {"targ_event_pos"};
     std::string wt {"_withTarg.csv"};
     std::string nt {"_noTarg.csv"};

     std::vector<std::pair<std::string, std::string>> filenames;
     for (int i {2}; i < 10; ++i) {
          if (i % 2 == 0) {
               std::string ev {relPathE + std::to_string(i) + wt};
               std::string gt {relPathG + std::to_string(i) + ".csv"};
               filenames.push_back(std::make_pair(ev, gt));
          } else {
               std::string ev {relPathE + std::to_string(i) + nt};
               std::string gt {relPathG + std::to_string(i) + ".csv"};
               filenames.push_back(std::make_pair(ev, gt));
          }
     }

     // Set up the parameters for the curves.
     int N {21};
     Eigen::MatrixXd pts {{2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}};
     Eigen::MatrixXd tau {Eigen::MatrixXd::Constant(N, 1, 1.2)};

     // Set up the dataframes themselves.
     std::map<std::string, Eigen::MatrixXd> dataframes;
     dataframes["eps=6"] = Eigen::MatrixXd::Zero(N, 3);
     dataframes["eps=8"] = Eigen::MatrixXd::Zero(N, 3);
     dataframes["eps=10"] = Eigen::MatrixXd::Zero(N, 3);

     dataframes["eps=6"](Eigen::all, 2) = tau;
     dataframes["eps=8"](Eigen::all, 2) = tau;
     dataframes["eps=10"](Eigen::all, 2) = tau;

     for (int i {0}; i < N; ++i) {
          dataframes["eps=6"](i, 1) = pts(i, 0);
          dataframes["eps=8"](i, 1) = pts(i, 0);
          dataframes["eps=10"](i, 1) = pts(i, 0);
     }

     dataframes["eps=6"](Eigen::all, 0) = Eigen::MatrixXd::Constant(N, 1, 6);
     dataframes["eps=8"](Eigen::all, 0) = Eigen::MatrixXd::Constant(N, 1, 8);
     dataframes["eps=10"](Eigen::all, 0) = Eigen::MatrixXd::Constant(N, 1, 10);

     // Run LROC
     bgLROC<DBSCAN_GMPHD, phdModel>(1, filenames, "DBSCAN_GMPHD", dataframes, k_model);

     return 0;
}
