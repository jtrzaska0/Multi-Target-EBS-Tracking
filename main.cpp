#include <fstream>
#include <nlohmann/json.hpp>
#include "Event-Sensor-Detection-and-Tracking/Algorithm.hpp"
#include "threads.h"

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    /*

    Args:
        argv[1]: Absolute path to config JSON file

    config.json:
        DEVICE_TYPE: "xplorer" or "davis"
        INTEGRATION_TIME_MS: Integration time in milliseconds.
        PACKET_NUMBER: Number of event packets to aggregate.
        ENABLE_TRACKING: true or false
        ENABLE_STAGE: true or false
        STAGE_METHOD: Stage command calculation method: "median", "dbscan", or "median-history"
        EPSILON: epsilon for mlpack clustering
        MAGNIFICATION: Magnification
        ENABLE_LOGGING: true or false
        EVENT_FILEPATH: File for event CSV. Do not include the ".csv" extension

    Ret:
        0
    */

    if (argc != 2) {
        printf("Invalid number of arguments.\n");
        return 1;
    }

    std::string config_file = {std::string(argv[1])};
    std::ifstream f(config_file);
    json settings = json::parse(f);
    json params = settings["PROGRAM_PARAMETERS"];
    json noise_params = settings["NOISE_FILTER"];

    std::string device_type = params.value("DEVICE_TYPE", "xplorer");
    double integrationtime = params.value("INTEGRATION_TIME_MS", 2);
    int num_packets = params.value("PACKET_NUMBER", 1);
    bool enable_tracking = params.value("ENABLE_TRACKING", false);
    bool enable_stage = params.value("ENABLE_STAGE", false);
    std::string position_method = params.value("STAGE_METHOD", "median-history");
    double eps = params.value("EPSILON", 15);
    double mag = params.value("MAGNIFICATION", 0.05);
    bool enable_event_log = params.value("ENABLE_LOGGING", false);
    std::string event_file = params.value("EVENT_FILEPATH", "recording");

    bool active = true;
    std::thread stage_thread(drive_stage, position_method, eps, enable_stage, device_type, integrationtime, num_packets, mag, noise_params, std::ref(active));
    prepareStage.acquire();
    launch_threads(device_type, integrationtime, num_packets, enable_tracking, position_method, eps, enable_event_log, event_file, mag, noise_params, false, false, std::ref(active));
    stage_thread.join();

    return 0;
}