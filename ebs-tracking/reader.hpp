// File        reader.hpp
// Summary     Read event data from csv files and store as an object.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <fstream>
# include <vector>
# include <string>
# include <algorithm>

// Other imports
# include "base-classes.hpp"


// Start with defining tools for reading csv files.
class EventData {
     /*
     This class is an abstraction for event sets. For reading data, the class
     expects a header layout {time, x, y, polarity}, in that order. No headers
     should be explicity stated in the file though; just provide the values.
     */
     private:
     std::vector<std::vector<double>> eventSets {};
     std::vector<std::vector<double>> gndTruth {};

     public:
     EventData() {}

     ~EventData() {}

     private:
     std::vector<double> readEventCSV(std::string fname) {
          /*
          Open an event file and read its contents.

          Args:
               fname: Filename.

          Ret:
               std::vector of values contiguous in memory.
          */

          // Find and open the file.
          std::ifstream stream {fname};
          if (!stream.good()) {
               std::cout << "Given file: " << fname << ".\n";
               throw;
          }
          // Make the vector. Headers are known a priori.
          std::vector<double> csvmap;

          // csv's have format <data><comma><space><data>...<\newline>.
          // have a loop that reads this exact format on each line and
          // pushed the results to a set of vectors.
          while (!stream.eof()) {
               char tmpc;
               double tmpt;
               double tmpx;
               double tmpy;
               double tmpp;
               stream >> tmpt >> tmpc
                      >> tmpx >> tmpc
                      >> tmpy >> tmpc
                      >> tmpp;
               csvmap.push_back(tmpt);
               csvmap.push_back(tmpx);
               csvmap.push_back(tmpy);
               csvmap.push_back(tmpp);
               if (std::isinf(tmpt) || std::isinf(tmpx) || std::isinf(tmpy) || std::isinf(tmpp))
                    std::cout << "Infinite values in file: " << fname << "\n";
          }         

          return csvmap;
     }
     

     std::vector<double> readGndTruthCSV(std::string fname) {
          /*
          Open a ground truth file and read its contents.

          Args:
               fname: Filename.

          Ret:
               A vector of double. Each group of four is a different ground truth label. {[||||], [||||], ...}
          */

          // Find and open the file.'
          std::ifstream stream {fname};
          if (!stream.good()) {
               std::cout << "Given file: " << fname << ".\n";
               throw;
          }

          // Make the vector. Headers are known a priori.
          std::vector<double> csvmap;

          // csv's have format <data><comma><space><data>...<\newline>.
          // have a loop that reads this exact format on each line and
          // pushed the results to a set of vectors.
          while (!stream.eof()) {
               char tmpc;
               double tmpt;
               double tmpx;
               double tmpy;
               double tmpn;
               stream >> tmpt >> tmpc
                      >> tmpx >> tmpc
                      >> tmpy >> tmpc
                      >> tmpn;
               csvmap.push_back(tmpt);
               csvmap.push_back(tmpx);
               csvmap.push_back(tmpy);
               csvmap.push_back(tmpn);
               // Make sure the file contains only finite values.
               if (std::isinf(tmpt) || std::isinf(tmpx) || std::isinf(tmpy) || std::isinf(tmpn))
                    std::cout << "Infinite values in file: " << fname << "\n";
          }

          return csvmap;
     }

    
     public:
     void readEventsWithGndTruth(std::string ename, std::string gname) {
          /*
          Read a set of events and the associated ground truth labels.
          */ 
          eventSets.push_back(readEventCSV(ename));
          if (gname == "") {
               std::cout << "Error invalid ground truth filename.\n";
               throw;
          } else
               gndTruth.push_back(readGndTruthCSV(gname));
     }


     void readEventsNoGndTruth(std::string ename) {
          /*
          Read a set of events.
          */ 
          eventSets.push_back(readEventCSV(ename));
          std::vector<double> gt;
          gndTruth.push_back(gt);
     }


     int getNumSets() const {
          /*
          How many pairs of events and ground truths are there.

          Args:
               None

          Ret:
               The number of pairs of events and ground truths.
          */
          return eventSets.size();
     }

     
     std::pair<std::vector<double>, std::vector<double>> operator[](int idx) {
          /*
          Read a set of events with ground truth.

          Args:
               idx: An integer indicating the set to be read.

          Ret:
               A pair of vectors. The first holds the event data, the second holds
               ground truth labels.
          */
          if (idx >= eventSets.size())
               throw;
          else
               return std::make_pair(eventSets[idx], gndTruth[idx]);
     }
};

