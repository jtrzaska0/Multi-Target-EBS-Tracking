#pragma once

#include <X11/Xlib.h>
#include <X11/keysym.h>

double get_hfov(double focal_len, double dist, int npx, double px_size) {
    /*
    Get the half field of view
    Args:
        focal_len: focal length in meters
        dist: distance to focal plan in meters
        npx: number of pixels along desired dimension
        px_size: pixel size in meters
    Ret:
        hfov: horizontal half field of view in radians
    */
    return atan((1 / dist + 1 / focal_len) * npx * px_size / 2);
}

bool key_is_pressed(KeySym ks) {
    Display *dpy = XOpenDisplay(":0");
    char keys_return[32];
    XQueryKeymap(dpy, keys_return);
    KeyCode kc2 = XKeysymToKeycode(dpy, ks);
    bool isPressed = !!(keys_return[kc2 >> 3] & (1 << (kc2 & 7)));
    XCloseDisplay(dpy);
    return isPressed;
}

double get_phi(double x, int nx, double hfov) {
    /*
    Get the azimuthal angle of an object in frame
    Args:
        x: x coordinate of object
        nx: number of pixels along x
        hfov: horizontal half field of view in radians
    Ret:
        phi: azimuthal angle of object in radians
    */
    return atan(2 * x * tan(hfov) / nx);
}

double get_theta(double y, int ny, double hfov) {
    /*
    Get the polar angle of an object in frame
    Args:
        y: y coordinate of object
        ny: number of pixels along y
        hfov: vertical half field of view in radians
    Ret:
        theta: polar angle of object in radians
    */
    return (M_PI / 2) - atan(2 * y * tan(hfov) / ny);
}

double get_phi_prime(double phi, double offset_x, double offset_y, double r_center) {
    /*
    Get the azimuthal angle of an object in frame with respect to the stage
    Args:
        phi: azimuthal angle in radians with respect to camera
        offset_x: distance along x-axis from camera to stage in meters
        offset_y: distance along y-axis from camera to stage in meters
        r_center: distance to object in meters when centered in camera
        error: systematic error of pan in radians
    Ret:
        phi_prime: azimuthal angle with respect to the stage
    */
    double num = r_center*tan(phi) - offset_y;
    double denom = r_center - offset_x;
    return atan(num/denom);
}

double get_theta_prime(double phi, double theta, double offset_x, double offset_y, double offset_z, double r_center, double arm) {
    /*
    Get the polar angle of an object in frame with respect to the stage
    Args:
        phi: azimuthal angle in radians with respect to camera
        theta: polar angle in radians with respect to camera
        offset_x: distance along x-axis from camera to stage in meters
        offset_y: distance along y-axis from camera to stage in meters
        offset_z: distance along z-axis from camera to stage in meters
        arm: length of stage arm in meters
        r_center: distance to object in meters when centered in camera
        error: systematic error of tilt in radians
    Ret:
        theta_prime: polar angle with respect to the stage
    */
    double num = r_center/tan(theta)/cos(phi) - offset_z;
    double denom_1 = pow(offset_x, 2) + pow(offset_y, 2) + pow(offset_z, 2);
    double denom_2 = pow(r_center, 2)/pow(sin(theta), 2)/pow(cos(phi), 2);
    double denom_3 = 2*r_center*(offset_x + offset_y*tan(phi) + offset_z/tan(theta)/cos(phi));
    double denom = sqrt(denom_1 + denom_2 - denom_3);
    return acos(num/denom) + asin(arm*sin(theta)*cos(phi)/denom);
}

int get_motor_position(int motor_begin, int motor_end, float ref_begin, float ref_end, double ref_target) {
    /*
    Get the target motor position from within a specified reference frame
    Args:
        motor_begin: motor position at reference frame start
        motor_end: motor position at reference frame end
        ref_begin: reference frame start
        ref_end: reference frame end
        ref_target: desired position in reference frame
    Ret:
        motor_position: desired position in reference frame expressed as a motor position, bounded by the calibration
    */
    float slope = (float) (motor_end - motor_begin) / (ref_end - ref_begin);
    int target = (int) (slope * (ref_target - ref_begin) + motor_begin);
    return std::max(std::min(motor_end, target), motor_begin);
}