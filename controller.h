#pragma once
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
extern "C" {
#include "ptu-sdk/examples/estrap.h"
}

class PIDController {
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

private:
    double kp;          // Proportional gain
    double ki;          // Integral gain
    double kd;          // Derivative gain
    double integral;    // Integral term
    int prev_error;     // Previous error
    int upper_bound;    // Upper motor position limit
    int lower_bound;    // Lower motor position limit
    bool accumulate;    // Toggle integral accumulation using limits
};

class StageController {
public:
    StageController(double kp, double ki, double kd, int pan_max, int pan_min, int tilt_max, int tilt_min, struct cerial *cer):
            pan_ctrl(kp, ki, kd, pan_max, pan_min), active(true), pan_setpoint(0), tilt_setpoint(0),
            tilt_ctrl(kp, ki, kd, tilt_max, tilt_min), cer(cer), status(0),
            ctrl_thread(std::thread(&StageController::ctrl_loop, this)) {};

    ~StageController() {
        if (active)
            this->shutdown();
    }

    void update_setpoints(int pan_target, int tilt_target) {
        update_mtx.lock();
        pan_setpoint = pan_target;
        tilt_setpoint = tilt_target;
        update_mtx.unlock();
    };

    void increment_setpoints(int pan_inc, int tilt_inc) {
        update_mtx.lock();
        pan_setpoint += pan_inc;
        tilt_setpoint += tilt_inc;
        update_mtx.unlock();
    };

    void shutdown() {
        active = false;
        ctrl_thread.join();
    }

private:
    PIDController pan_ctrl;
    PIDController tilt_ctrl;
    struct cerial *cer;
    std::mutex update_mtx;
    std::thread ctrl_thread;
    int pan_setpoint;
    int tilt_setpoint;
    uint16_t status;
    bool active;

    void ctrl_loop() {
        auto start_time = std::chrono::high_resolution_clock::now();
        while(active) {
            int pan_pos;
            int tilt_pos;
            cpi_ptcmd(cer, &status, OP_PAN_CURRENT_POS_GET, &pan_pos);
            cpi_ptcmd(cer, &status, OP_TILT_CURRENT_POS_GET, &tilt_pos);
            update_mtx.lock();
            auto stop_time = std::chrono::high_resolution_clock::now();
            auto command_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
            int pan_command = pan_ctrl.calculate(pan_setpoint, pan_pos, (double)command_time);
            int tilt_command = tilt_ctrl.calculate(tilt_setpoint, tilt_pos, (double)command_time);
            start_time = std::chrono::high_resolution_clock::now();
            update_mtx.unlock();
            cpi_ptcmd(cer, &status, OP_TILT_DESIRED_POS_SET, tilt_command);
            cpi_ptcmd(cer, &status, OP_PAN_DESIRED_POS_SET, pan_command);
        }
    }
};
