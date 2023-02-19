#include "ebs-tracking/Algorithm.hpp"
#include "threads.h"

int main(int argc, char* argv[]) {
    /*

     Args:
          argv[1]: Device type: "xplorer" or "davis"
          argv[2]: Integration time in milliseconds.
          argv[3]: Number of event packets to aggregate.
          argv[4]: Tracking: 0 for disabled, 1 for enabled
          argv[5]: Stage: 0 for disabled, 1 for enabled
          argv[6]: Stage command calculation method: "median"
          argv[7]: Magnification

     Ret:
          0
     */

    if (argc != 8) {
        printf("Invalid number of arguments.\n");
        return 1;
    }

    std::string device_type = {std::string(argv[1])};
    double integrationtime = {std::stod(argv[2])};
    int num_packets = {std::stoi(argv[3])};
    bool enable_tracking = {std::stoi(argv[4])!=0};
    bool enable_stage= {std::stoi(argv[5])!=0};
    std::string position_method = {std::string(argv[6])};
    double mag = {std::stod(argv[7])};

    bool active = true;
    std::thread stage_thread(drive_stage, position_method, enable_stage, std::ref(active));
    prepareStage.acquire();
    launch_threads(device_type, integrationtime, num_packets, enable_tracking, position_method, mag, std::ref(active));
    stage_thread.join();

    return 0;
}