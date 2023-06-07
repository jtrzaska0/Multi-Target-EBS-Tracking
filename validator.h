#pragma once
#include <cmath>
#include <chrono>

class Validator {
public:
    Validator(double position_thres, double time_thres):
            start(std::chrono::high_resolution_clock::now()) {
        coarse_pan = 0;
        coarse_tilt = 0;
        fine_pan = 0;
        fine_tilt = 0;
        pos_thres = position_thres;
        this->time_thres = time_thres;
        time_since_ebs_detection = 0.0;
    };

    void new_ebs_detection(int pan, int tilt) {
        coarse_pan = pan;
        coarse_tilt = tilt;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        time_since_ebs_detection = (double)duration.count();
        start = std::chrono::high_resolution_clock::now();
    }

    void new_camera_detection(int pan, int tilt) {
        fine_pan = pan;
        fine_tilt = tilt;
    }

    [[nodiscard]] bool verify() const {
        double pan_error = (double)abs(fine_pan - coarse_pan) / coarse_pan;
        double tilt_error = (double)abs(fine_tilt - coarse_tilt) / coarse_tilt;
        if (pan_error < pos_thres && tilt_error < pos_thres && time_since_ebs_detection < time_thres)
            return true;
        return false;
    }

private:
    int coarse_pan;
    int coarse_tilt;
    int fine_pan;
    int fine_tilt;
    double pos_thres;
    double time_thres;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    double time_since_ebs_detection;
};