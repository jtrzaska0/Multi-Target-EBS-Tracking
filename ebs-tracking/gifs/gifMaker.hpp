// File        gifMaker.hpp
// Summary     Build a GIF of the event stream using data stored in a csv file.
// Author      Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <fstream>
# include <vector>
# include <string>
# include <chrono>

// Other imports
# include "../base-classes.hpp"
# include "../Algorithm.hpp"
# include "../reader.hpp"


class GifMaker {
     /*
     Take a file of event data (in CSV format) and output a GIF.
     */
     private:
     double Tau;
     EventData eventdata;
     bool stat;

     public:
     GifMaker(std::string eventfile, double time) : Tau {time} {
          /*
          Constructor.

          Args:
               eventfile: Strings specifying location of the event file.
               time: Integration time for each GIF frame (in ms).

          Ret:
               None.
          */

          // Get event data
          eventdata.readEventsNoGndTruth(eventfile);
     }

     ~GifMaker() {}


     void run(std::string fname) {
          /*
          Run the event stream and produce a GIF of the events.
 
          Args:
               fname: Filename to write event frames to.

          Ret:
               None
          */

          // Get events and ground truth.
          auto [events, gt] = eventdata[0];

          // The detector takes a pointer to events. 
          double * mem {events.data()};

          // Starting time.
          double t0 {events[0]};

          // Keep sizes of the vectors in variables.
          int nEvents {(int) events.size() / 4};

          // Open a file to stream the GIF data to.
          std::ofstream outFile {fname};

          while (true) {
               // Read all events in one integration time. 
               double t1 {t0 + Tau};
               int N {0};
               for (; N < (int) (events.data() + events.size() - mem) / 4; ++N)
                    if (mem[4 * N] >= t1)
                         break;

               // Advance starting time.
               t0 = t1;

               // Store frame as a block in outFile.
               if (N == 0)
                    outFile << -10 << " " << -10 << "\n";
               else {
                    for (int i {0}; i < N; ++i) {
                         outFile << mem[4 * i + 1] // X-Position 
                                 << " " 
                                 << mem[4 * i + 2] // Y-Position
                                 << " " 
                                 << mem[4 * i + 3] // Polarity
                                 << "\n";
                    }
               }

               outFile << "\n\n";

               // Break once all events have been used.
               if (t0 > events[4 * (nEvents - 1)])
                    break;

               // Update eventIdx
               mem += 4*N;
          }
     
          outFile.close();

          return;
     }

     
     void run(AlgoImpl auto T, std::string efname, std::string tfname) {
          /*
          Run the event stream and produce a GIF of the events. This version
          of the function also tracks object positions.

          Args:
               T: the class being used to detect and track.
               efname: Filename to write event frames to.
               tfname: Filename to write locations to.

          Ret:
               None
          */
          auto [events, gt] = eventdata[0];

          // The detector takes a pointer to events. 
          double * mem {events.data()};

          // Starting time.
          double t0 {events[0]};

          // Keep sizes of the vectors in variables.
          int nEvents {(int) events.size() / 4};

          // Open a file to stream the GIF to.
          std::ofstream eFile {efname};
          std::ofstream tFile {tfname};

          while (true) {
               // Read all events in one integration time. 
               double t1 {t0 + Tau};
               int N {0};
               for (; N < nEvents + (int) (events.data() - mem) / 4; ++N)
                    if (mem[4 * N] >= t1)
                         break;
               t0 = t1;

               // Feed events to the detector/tracker.
               auto start {std::chrono::high_resolution_clock::now()};
               T(mem, N);
               Eigen::MatrixXd targets {T.currentTracks()};
               auto end {std::chrono::high_resolution_clock::now()};
               auto duration = duration_cast<std::chrono::microseconds>(end - start);
               std::cout << duration.count() << " us" << "\n\n";
               //std::cout << targets << "\n\n";

               // Store frame as a block outFile.
               if (N == 0)
                    eFile << -10 << " " << -10 << "\n";
               else {
                    for (int i {0}; i < N; ++i) {
                         eFile << mem[4 * i + 1] // X-Position 
                               << " " 
                               << mem[4 * i + 2] // Y-Position
                               << " " 
                               << mem[4 * i + 3] // Polarity
                               << "\n";
                    }
               }

               eFile << "\n\n";

               // Store the objects as their own file. 
               if (targets.rows() == 0)
                    tFile << -10 << " " << -10 << "\n";
               else {
                    for (int i {0}; i < targets.rows(); ++i) {
                         tFile << targets(i, 0)
                               << " "
                               << targets(i, 1)
                               << "\n";
                    }
               }

               tFile << "\n\n";

               // Break once all events have been used.
               if (t0 > events[4 * (nEvents - 1)])
                    break;

               // Evolve tracks in time.
               T.predict();

               // Update eventIdx
               mem += 4*N;
          }

          // Close files.
          eFile.close();
          tFile.close();

          return;
     }
};
