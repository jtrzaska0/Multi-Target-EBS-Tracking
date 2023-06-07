#pragma once
#include <cmath>
#include <chrono>

class Validator {
public:
    explicit Validator(double position_thres) {
        coarse_pan = 0;
        coarse_tilt = 0;
        fine_pan = 0;
        fine_tilt = 0;
        pos_thres = position_thres;
        is_valid = false;
    };

    void new_ebs_detection(int pan, int tilt) {
        coarse_pan = pan;
        coarse_tilt = tilt;
        is_valid = verify();
    }

    void new_camera_detection(int pan, int tilt) {
        fine_pan = pan;
        fine_tilt = tilt;
    }

    void start_validator() {
        is_valid = true;
    }

    [[nodiscard]] bool get_status() const {
        return is_valid;
    }

private:
    int coarse_pan;
    int coarse_tilt;
    int fine_pan;
    int fine_tilt;
    double pos_thres;
    bool is_valid;

    [[nodiscard]] bool verify() const {
        double pan_error = (double)abs(fine_pan - coarse_pan) / coarse_pan;
        double tilt_error = (double)abs(fine_tilt - coarse_tilt) / coarse_tilt;
        if (pan_error < pos_thres && tilt_error < pos_thres)
            return true;
        return false;
    }
};