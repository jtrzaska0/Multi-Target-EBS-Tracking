// File        LROC.hpp
// Summary     Implements a class for handling Localization ROC (LROC) analysis.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <fstream>
# include <vector>
# include <string>
# include <map>
# include <algorithm>
# include <iterator>
# include <array>

// Other imports
# include "../base-classes.hpp"
# include <armadillo>
# include "../reader.hpp"

template <AlgoImpl T, typename U>
class LROC {
     /*
     This class implements Localization Receiver Operating Characteristic (LROC) analysis.
     */
     private:
     std::vector<double> rads;
     EventData eventdata;
     std::map<std::string, Eigen::MatrixXd> dataframes;

     public:
     LROC(std::vector<double> radii, std::vector<std::pair<std::string, std::string>> eventFilenames,
          std::map<std::string, Eigen::MatrixXd> parameters) {
          /*
          Constructor.

          Args:
               radii: Maximum radius a track can be from ground truth label to count as a localization.
               eventFilenames: A vector containing a pair of two strings. The first string should provide a
                               filename for the events and the second the filename for the ground truth.
               parameters: Contains the parameters needed to initialize the T class.

          Ret:
               None
          */
          rads = radii;
          dataframes = parameters;

          for (auto fn : eventFilenames)
               eventdata.readEventsWithGndTruth(fn.first, fn.second);
     }

     ~LROC() {}


     void run(double dt, std::string fname, U extra, std::string info="") {
          /*
          Run LROC analysis. All LROC curves are written as blocks in a file.

          Args:
               dt: Integration time.
               fname: Name of the output file.
               info: Any info wanted displayed at the top of the file.

          Ret:
               None
          */
          using df = std::map<std::string, std::vector<double>>;
          std::ofstream outputFile {fname};
          if (!outputFile.good()) {
               std::cout << "Failed to create output file.\n";
               throw;
          }

          // If the info string is empty do not bother writing anything.
          if (info != "")
               outputFile << "# " << info << "\n\n";

          // Choose a localization threshold
          for (auto r : rads) {
               // Choose a dataframe of parameters
               for (std::map<std::string, Eigen::MatrixXd>::iterator iter = dataframes.begin(); iter != dataframes.end(); ++iter) {
                    // Get the name of the parameter set.
                    std::string pSetName {iter->first};
                    // Label the data block in the output file.
                    outputFile << "# " << pSetName << ": Radius = " << r << "\n";
                    // Loop over all the points and build the LROC curve.
                    for (int i {0}; i < (iter->second).rows(); ++i) {
                         Eigen::MatrixXd p {(iter->second).row(i)};
                         //std::cout << p << "\n";
                         std::array<double, 5> counts {0, 0, 0, 0, 0};
                         // Now loop over all the datasets and run localization tests.
                         for (int k {0}; k < eventdata.getNumSets(); ++k) {
                              auto [events, gt] = eventdata[k];
                              // Build algorithm.
                              T algo(p, extra);
                              auto [lt, tp, tn, fp, fn] = stream(algo, events, gt, dt, r);
                              // Update counts.
                              counts[0] += lt;
                              counts[1] += tp;
                              counts[2] += tn;
                              counts[3] += fp;
                              counts[4] += fn;
                         }

                         // Calculate the False Positive Rate and the Localized True Positive Rate.
                         double FPR {counts[3] / (counts[3] + counts[2])};
                         double LTP {counts[0] / (counts[1] + counts[4])};
                         // Write the result to file.
                         outputFile << FPR << " " << LTP << "\n";
                    }

                    outputFile << "\n\n";
               }
          }

          return;
     }


     private:
     std::array<double, 5> stream(T algo, std::vector<double> events, std::vector<double>& gt, double dt, double rad) {
          /*
          This function plays the event stream to test the performance of the detection/tracking algorithm.
     
          Args:
               dt: Integration time.
               algo: the class being used to detect and track.
               events: The set of events.
               gt: The ground truth labels for the events.
               rad: Localization threshold.

          Ret:
               {lt, tp, tn, fp, fn}
          */
          // This structure contains {lt, tp, tn, fp, fn}.
          std::array<double, 5> counts {0, 0, 0, 0, 0};

          // The detector takes a pointer to events. 
          double * mem {events.data()};

          // Starting time.
          double t0 {events[0]};

          // Keep sizes of the vectors in variables.
          int labelsUsed {0};
          int nEvents {(int)events.size() / 4};
          int nLabels {(int)gt.size() / 4};

          // Only position is being used.
          Eigen::Vector2d gtVector;

          while (true) {
               // Read all events in one integration time. 
               double t1 {t0 + dt};
               int N {0};
               for (; N < (int) (events.data() + events.size() - mem) / 4; ++N)
                    if (mem[4 * N] >= t1)
                         break;
               t0 = t1;

               // Feed events to the detector/tracker.
               algo(mem, N);
               Eigen::MatrixXd targets {algo.currentTracks()};

               // Test localization for LROC.
               if (labelsUsed < nLabels) {
                    // Check if we are temporally close to a label. If so, 
                    // run the localization test. Continue otherwise.
                    double sep_t {fabs(gt[4 * labelsUsed] - t0)};
                    if (sep_t <= dt) {
                         // Fill the ground truth matrix (only position needed).
                         for (int j {0}; j < 2; ++j)
                              gtVector(j) = gt[4 * labelsUsed + j + 1];
                         double card = gt[4 * labelsUsed + 3];
                         testLocalization(counts, targets, gtVector, card, rad);
                         ++labelsUsed;
                    }
               }

               // Break once all events have been used.
               if (t0 > events[4 * (nEvents - 1)])
                    break;

               // Evolve tracks in time.
               algo.predict();

               // Update eventIdx
               mem += 4*N;
          }

          return counts;
     }


     void testLocalization(std::array<double, 5>& counts, Eigen::MatrixXd targets, Eigen::Vector2d gt, double card, double rad) {
          /*
          Given the target set and ground truth label, determine whether the tracking was successful.

          Args:
               counts: The array keeping tally of the true/false counts.
               targets: An Eigen matrix holding all available target information. Each row is a different object.
               gt: The ground truth label as an Eigen Vector.
               card: The number of targets actually present.
               rad: Localization threshold.

          Ret:
               None.
          */
          if ( (targets.rows() == 0) && (card == 0)) {
               // True negative.
               counts[2] += 1;
          } else if ( (targets.rows() != 0) && (card == 0)) {
               // False positive.
               counts[3] += 1;
          } else if ( (targets.rows() == 0) && (card != 0)) {
               // False negative.
               counts[4] += 1;
          } else {
               // True positive.
               counts[1] += 1;

               for (int i {0}; i < targets.rows(); ++i) {
                    Eigen::Vector2d z {targets(i, Eigen::seq(0,1))};
                    Eigen::Vector2d s {z - gt};
                    double sep_d {s.dot(s)};
                    if (sep_d <= rad*rad) {
                         // Localized true positive.
                         counts[0] += 1;
                         break;
                    }
               }
          }

          return;
     }
};
