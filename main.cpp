#include <fstream>
#include <semaphore>

#include "ebs-tracking/Algorithm.hpp"
#include "ebs-tracking/reader.hpp"
#include "ebs-tracking/gifs/gifMaker.hpp"

std::counting_semaphore<1> prepareTracker(0);

// Read csv and process in "real time"
// dt: integration time in ms
// delay: time behind actual in ms
void tracker (std::string eData, double dt, DBSCAN_KNN T, Eigen::MatrixXd&targets, int&slice, int&delay, bool&active) {
    printf("Initializing tracker...\n");
    // Process tracking data.
    EventData eventdata;
    eventdata.readEventsNoGndTruth(eData);
    auto [events, gt] = eventdata[0];

    // The detector takes a pointer to events.
    double * mem {events.data()};

    // Starting time.
    double t0 {events[0]};

    // Keep sizes of the vectors in variables.
    int nEvents {(int) events.size() / 4};

    printf("Tracker prepared.\n");
    prepareTracker.release();

    printf("Starting tracker...\n");
    while (true) {
        auto start {std::chrono::high_resolution_clock::now()};
        // Read all events in one integration time.
        double t1 {t0 + dt};
        int N {0};
        for (; N < (int) (events.data() + events.size() - mem) / 4; ++N)
            if (mem[4 * N] >= t1)
                break;

        // Advance starting time.
        t0 = t1;

        // Feed events to the detector/tracker.
        T(mem, N);
        targets = T.currentTracks();

        // Break once all events have been used.
        if (t0 > events[4 * (nEvents - 1)])
            break;

        // Evolve tracks in time.
        T.predict();

        // Update eventIdx
        mem += 4*N;
        slice += 1;

        auto end {std::chrono::high_resolution_clock::now()};
        auto duration = duration_cast<std::chrono::microseconds>(end - start);
        if ((double)duration.count() > dt*1000){
            delay += (int)duration.count() - (int)dt*1000;
        }
        else {
            int wait = (int)dt*1000 - (int)duration.count();
            if (delay < wait) {
                std::this_thread::sleep_for(std::chrono::microseconds(wait-delay));
                delay = 0;
            }
            else {
                delay -= wait;
            }
        }
    }

    active = false;
    printf("Reached end of file. Total delay (ms): %d\n", delay);
}

int main(int argc, char* argv[]) {
    /*
     Simulate live data tracking using a CSV file containing event data.
     A separate Matlab script can be used to generate GIFs of the result.

     Args:
          argv[1]: File containing the event data.
          argv[2]: Integration time in milliseconds.
          argv[3]: Camera loop time in milliseconds. If 0, semaphores are used to capture each event.
          argv[4]: Path for events file.
          argv[5]: Path for locations file.

     Ret:
          0
     */

    if (argc != 6) {
        printf("Invalid number of arguments.\n");
        return 0;
    }

    // Path to event data
    std::string eData {argv[1]};
    // Integration time in ms
    double integrationtime = {std::stod(argv[2])};
    // Loop time in ms
    double looptime = {std::stod(argv[3])};


    // Path for events file
    std::string efname {argv[4]};
    // Path for locations file
    std::string tfname {argv[5]};


    /**Create an Algorithm object here.**/
    // Matrix initializer
    // DBSCAN
    Eigen::MatrixXd invals {Eigen::MatrixXd::Zero(1, 4)};
/*     invals(0, 0) = 8;
     invals(0, 1) = 8;
     invals(0, 2) = 1.2;
*/
    // Mean Shift
    invals(0,0) = 5.2;
    invals(0,1) = 9;
    invals(0,2) = 74;
    invals(0,3) = 1.2;
    // Model initializer
    double DT = integrationtime;
    double p3 = pow(DT, 3) / 3;
    double p2 = pow(DT, 2) / 2;

    Eigen::MatrixXd P {{16, 0, 0, 0}, {0, 16, 0, 0}, {0, 0, 9, 0}, {0, 0, 0, 9}};
    Eigen::MatrixXd F {{1, 0, DT, 0}, {0, 1, 0, DT}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    Eigen::MatrixXd Q {{p3, 0, p2, 0}, {0, p3, 0, p2}, {p2, 0, DT, 0}, {0, p2, 0, DT}};
    Eigen::MatrixXd H {{1, 0, 0, 0}, {0, 1, 0, 0}};
    Eigen::MatrixXd R {{7, 0}, {0, 7}};

    // Define the model.
    KModel k_model {.dt = DT, .P = P, .F = F, .Q = Q, .H = H, .R = R};
    // Algo initializer
    DBSCAN_KNN algo(invals, k_model);

    // Write the events to a file before the simulation
    GifMaker gm(eData, integrationtime);
    gm.run(efname);

    // Open a file for the tracking data
    std::ofstream tFile {tfname};

    bool active = false;
    int slice = 1;
    int delay = 0;
    Eigen::MatrixXd targets {};
    std::thread tracking_thread(tracker, eData, integrationtime, algo, std::ref(targets), std::ref(slice), std::ref(delay), std::ref(active));
    prepareTracker.acquire();
    active = true;
    int fixedslice;

    printf("Starting reading loop...\n");
    while (active) {
        auto start {std::chrono::high_resolution_clock::now()};

        fixedslice = slice;
        // Store the objects as their own file.
        if (targets.rows() == 0)
            tFile << -10 << " " << -10 << " " << fixedslice << "\n";
        else {
            for (int i {0}; i < targets.rows(); ++i) {
                tFile << targets(i, 0)
                      << " "
                      << targets(i, 1)
                      << " "
                      << fixedslice
                      << "\n";
            }
        }

        tFile << "\n\n";

        auto end {std::chrono::high_resolution_clock::now()};
        auto duration = duration_cast<std::chrono::milliseconds>(end - start);
        int wait = (int)looptime - (int)duration.count();
        if (wait > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(wait));
        }
        else {
            printf("WARNING: Camera loop exceeded expected time.\n");
        }
    }

    tracking_thread.join();
    printf("Slices: %hd\n", slice);

    return 0;
}