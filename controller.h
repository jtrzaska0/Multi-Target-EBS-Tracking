// File         controller.h
// Summary      Class for controlling the FLIR stage.
// Author       Trevor Schlackt - Modified by Jacob Trzaska
# pragma once

// Standard imports
# include <iostream>
# include <cmath>
# include <thread>
# include <chrono>
# include <string>

// Local imports
# include "pointing.h"
extern "C" {
# include "ptu-sdk/examples/estrap.h"
}

// Namespacing
// None.


class PIDController {
    /*
    Implements a proportional-integral-derivative (PID) controller.
    */

    private:
    double kp;          // Proportional gain
    double ki;          // Integral gain
    double kd;          // Derivative gain
    double integral;    // Integral term
    int prev_error;     // Previous error
    int upper_bound;    // Upper motor position limit
    int lower_bound;    // Lower motor position limit
    bool accumulate;    // Toggle integral accumulation using limits

    public:
    PIDController(double kp, double ki, double kd, int upper_bound, int lower_bound) {
        this->kp = kp;
        this->ki = ki;
        this->kd = kd;
        this->upper_bound = upper_bound;
        this->lower_bound = lower_bound;
        integral = 0;
        prev_error = 0;
        accumulate = true;
    }

    int calculate(int setpoint, int measured_value, double dt) {
        /*
        Calculate the new set.

        Args:
            setpoint:       Where to set.
            measured_value: Measurement.
            dt:             Timestep for the derivative.

        Ret:
            New value.

        Notes:
            None.
        */

        int error = setpoint - measured_value;

        if (accumulate)
            integral += error * dt;

        double derivative = (error - prev_error) / dt;

        int output = (int)((kp * (double)error) + (ki * integral) + (kd * derivative));
        if (output < lower_bound) {
            accumulate = false;
            output = lower_bound;
        }

        if (output > upper_bound) {
            accumulate = false;
            output = upper_bound;
        }

        if (!accumulate && abs(output) < abs(lower_bound) && abs(output) < abs(upper_bound))
            accumulate = true;

        prev_error = error;

        return output;
    }

    void update_gains(double new_kp, double new_ki, double new_kd) {
        kp = new_kp;
        ki = new_ki;
        kd = new_kd;
    }

    void reset() {
        integral = 0.0;
        prev_error = 0;
    }
};



class StageController {
    /*
    Class for moving the FLIR stage.
    */

    private:
    double kp_coarse;
    double ki_coarse;
    double kd_coarse;
    double kp_fine;
    double ki_fine;
    double kd_fine;
    PIDController pan_ctrl;
    PIDController tilt_ctrl;
    struct cerial *cer;
    std::mutex update_mtx;
    std::thread ctrl_thread;
    int pan_setpoint;
    int tilt_setpoint;
    int current_pan;
    int current_tilt;
    uint16_t status;
    bool active;
    bool fine_active;
    std::ofstream stageFile;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    bool enable_logging;
    bool pid;
    int pan_offset;
    int tilt_offset;
    double fine_overshoot_time;
    double coarse_overshoot_time;
    double overshoot_thres;
    int update_time;
    int update_thres;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_update;
    bool verbose;


    public:
    StageController(double kp_coarse, double ki_coarse, double kd_coarse, double kp_fine, double ki_fine,
                    double kd_fine, int pan_max, int pan_min, int tilt_max, int tilt_min,
                    std::chrono::time_point<std::chrono::high_resolution_clock> start, const std::string& event_file,
                    bool enable_logging, struct cerial *cer, bool pid, double fine_time, double coarse_time,
                    double overshoot_thres, int update_time, int update_thres, bool verbose, int n):
            pan_ctrl(kp_coarse, ki_coarse, kd_coarse, pan_max, pan_min), active(true), pan_setpoint(0), tilt_setpoint(0),
            tilt_ctrl(kp_coarse, ki_coarse, kd_coarse, tilt_max, tilt_min), cer(cer), status(0), stageFile(event_file + "-stage" + std::to_string(n) + ".csv"),
            start(start), enable_logging(enable_logging), fine_active(false), current_pan(0), current_tilt(0), pid(pid),
            pan_offset(0), tilt_offset(0), last_update(std::chrono::high_resolution_clock::now()) {
        /*
        Constructor.

        Args:
            kp_coarse:      Proportional constant for coarse-track.
            ki_coarse:      Integral constant for coarse-track.
            kd_coarse:      Derivative constant for coarse-track.
            kp_fine:        Proportional constant for fine-track.
            ki_fine:        Integral constant for fine-track.
            kd_fine:        Derivative constant for fine-track.
            pan_max:        Max pan angle.
            pan_min:        Min pan angle.
            tilt_max:       Max tilt angle.
            tilt_min:       Min tilt angle.
            start:          Program starting time.
            event_file:     Logging file.
            enable_logging: Enable or disable logging.
            cer:            Pointer to FLIR stage control.
            pid:            
            fine_time:
            coarse_time:
            overshoot_thres:
            update_time:
            update_thres:
            verbose:
            n:              Integer indicating which stage is in use.

        Ret:
            None.

        Notes:
            None.
        */

        this->kp_coarse = kp_coarse;
        this->ki_coarse = ki_coarse;
        this->kd_coarse = kd_coarse;
        this->kp_fine = kp_fine;
        this->ki_fine = ki_fine;
        this->kd_fine = kd_fine;
        this->fine_overshoot_time = fine_time;
        this->coarse_overshoot_time = coarse_time;
        this->overshoot_thres = overshoot_thres;
        this->update_time = update_time;
        this->update_thres = update_thres;
        this->verbose = verbose;
        if (cer) {
            ctrl_thread = std::thread(&StageController::ctrl_loop, this);
        } else {
            active = false;
        }
    };

    ~StageController() {
        this->shutdown();
    }

    void shutdown() {
        if (active) {
            active = false;
            ctrl_thread.join();
        }
    }

    void update_setpoints(int pan_target, int tilt_target) {
        if (!fine_active) {
            update_mtx.lock();
            pan_setpoint = pan_target;
            tilt_setpoint = tilt_target;
            update_mtx.unlock();
        }
    };

    void increment_setpoints(int pan_inc, int tilt_inc) {
        if (fine_active) {
            update_mtx.lock();
            pan_setpoint = current_pan + pan_inc;
            tilt_setpoint = current_tilt + tilt_inc;
            update_mtx.unlock();
        }
    };

    void activate_fine() {
        update_mtx.lock();
        pan_ctrl.update_gains(kp_fine, ki_fine, kd_fine);
        pan_ctrl.reset();
        tilt_ctrl.update_gains(kp_fine, ki_fine, kd_fine);
        tilt_ctrl.reset();
        fine_active = true;
        update_mtx.unlock();
    }

    void deactivate_fine() {
        update_mtx.lock();
        pan_ctrl.update_gains(kp_coarse, ki_coarse, kd_coarse);
        pan_ctrl.reset();
        tilt_ctrl.update_gains(kp_coarse, ki_coarse, kd_coarse);
        tilt_ctrl.reset();
        fine_active = false;
        update_mtx.unlock();
    }


    bool get_tracker_status() const {
        return fine_active;
    }


    private:
    bool check_move(double pan_change, double tilt_change) {
        auto current = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current - last_update).count();

        if (elapsed > update_time*1000 && (pan_change > update_thres || tilt_change > update_thres))
            return true;

        return false;
    }


    void update_log(int pan, int tilt) {
        if (enable_logging) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start).count();
            auto stage_string = std::to_string(elapsed_time) + ",";
            stage_string += std::to_string(tilt) + ",";
            stage_string += std::to_string(pan) + ",";
            if (fine_active)
                stage_string += "Fine\n";
            else
                stage_string += "Coarse\n";

            stageFile << stage_string;
        }

        return;
    }


    void ctrl_loop() {
        auto start_time = std::chrono::high_resolution_clock::now();
        int pan_command;
        int tilt_command;

        while(active) {
            cpi_ptcmd(cer, &status, OP_PAN_CURRENT_POS_GET, &current_pan);
            cpi_ptcmd(cer, &status, OP_TILT_CURRENT_POS_GET, &current_tilt);

            if (key_is_pressed(XK_Up)) {
                tilt_offset += 1;
                printf("Tilt Offset: %d steps\n", tilt_offset);
            }

            if (key_is_pressed(XK_Down)) {
                tilt_offset -= 1;
                printf("Tilt Offset: %d steps\n", tilt_offset);
            }

            if (key_is_pressed(XK_Left)) {
                pan_offset -= 1;
                printf("Pan Offset: %d steps\n", pan_offset);
            }

            if (key_is_pressed(XK_Right)) {
                pan_offset += 1;
                printf("Pan Offset: %d steps\n", pan_offset);
            }

            auto stop_time = std::chrono::high_resolution_clock::now();
            auto command_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
            update_mtx.lock();
            double pan_velocity = (pan_setpoint - current_pan) / (double)command_time;
            double tilt_velocity = (tilt_setpoint - current_tilt) / (double)command_time;
            int pan_change = abs((pan_setpoint - current_pan));
            int tilt_change = abs((tilt_setpoint - current_tilt));
            int tilt_overshoot = 0;
            int pan_overshoot = 0;

            if (pan_change < overshoot_thres && tilt_change < overshoot_thres) {
                if (fine_active) {
                    tilt_overshoot = (int)(tilt_velocity * fine_overshoot_time * 1000);
                    pan_overshoot = (int)(pan_velocity * fine_overshoot_time * 1000);
                } else {
                    tilt_overshoot = (int)(tilt_velocity * coarse_overshoot_time);
                    pan_overshoot = (int)(pan_velocity * coarse_overshoot_time);
                }
            }

            if (pid) {
                if (!fine_active) {
                    pan_command = pan_ctrl.calculate(pan_setpoint + pan_overshoot + pan_offset, current_pan, (double) command_time);
                    tilt_command = tilt_ctrl.calculate(tilt_setpoint + tilt_overshoot + tilt_offset, current_tilt, (double) command_time);
                } else {
                    pan_command = pan_ctrl.calculate(pan_setpoint + pan_overshoot, current_pan, (double) command_time);
                    tilt_command = tilt_ctrl.calculate(tilt_setpoint + tilt_overshoot, current_tilt, (double) command_time);
                }
            } else {
                if (!fine_active) {
                    pan_command = pan_setpoint + pan_overshoot + pan_offset;
                    tilt_command = tilt_setpoint + tilt_overshoot + tilt_offset;
                } else {
                    pan_command = pan_setpoint + pan_overshoot;
                    tilt_command = tilt_setpoint + tilt_overshoot;
                }
            }

            update_mtx.unlock();

            bool move = check_move(pan_change, tilt_change);
            if (move) {
                cpi_ptcmd(cer, &status, OP_TILT_DESIRED_POS_SET, tilt_command);
                cpi_ptcmd(cer, &status, OP_PAN_DESIRED_POS_SET, pan_command);
                update_log(pan_command, tilt_command);
                last_update = std::chrono::high_resolution_clock::now();
            }

            if (verbose) {
                printf("Pan Change: %d steps\nTilt Change: %d steps\n", pan_change, tilt_change);
                printf("Pan Velocity: %0.3f steps/ms\nTilt Velocity: %0.3f steps/ms\n", pan_velocity, tilt_velocity);
                printf("Pan Overshoot: %d steps\nTilt Overshoot: %d steps\n", pan_overshoot, tilt_overshoot);
                printf("Pan Offset: %d steps\nTilt Offset: %d steps\n", pan_offset, tilt_offset);
                printf("Pan Command: %d steps\nTilt Command: %d steps\n", pan_command, tilt_command);
                printf("Stage MovedL %d\n\n", (int)move);
            }

            start_time = std::chrono::high_resolution_clock::now();
        }
    }
};
