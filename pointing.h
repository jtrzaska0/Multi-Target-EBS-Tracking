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

double get_phi_prime(double phi, double theta, double sep, double r, double error) {
    /*
    Get the azimuthal angle of an object in frame with respect to the stage
    Args:
        phi: azimuthal angle in radians with respect to camera
        theta: polar angle in radians with respect to camera
        sep: distance from camera to stage in meters
        r: distance to object in meters
        error: systematic error of pan in radians
    Ret:
        phi_prime: azimuthal angle with respect to the stage
    */
    return phi - (sep / r) * (cos(phi) / sin(theta)) + error;
}

double get_theta_prime(double phi, double theta, double sep, double r, double error) {
    /*
    Get the polar angle of an object in frame with respect to the stage
    Args:
        phi: azimuthal angle in radians with respect to camera
        theta: polar angle in radians with respect to camera
        sep: distance from camera to stage in meters
        r: distance to object in meters
        error: systematic error of tilt in radians
    Ret:
        theta_prime: polar angle with respect to the stage
    */
    return theta - (sep / r) * cos(theta) * sin(phi) + error;
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
    float slope = (float)(motor_end - motor_begin) / (ref_end - ref_begin);
    int target =  (int)(slope*(ref_target-ref_begin) + motor_begin);
    return std::max(std::min(motor_end, target), motor_begin);
}

void controller(struct cerial *cer, KeySym ks)
// Drive stage with manual controls. Use up/down arrow keys to adjust the speed
// ks is the key to stop this function. It can be anything except WASDQE or the
// arrow keys.
{
    printf("\n");
    printf("CONTROLS:\n");
    printf("---------\n");
    printf("Tilt Up: 'W'\n");
    printf("Tilt Down: 'S'\n");
    printf("Pan Left: 'A'\n");
    printf("Pan Right: 'D'\n");
    printf("Slide Left: Left Arrow\n");
    printf("Slide Right: Right Arrow\n");
    printf("Speed Up: Up Arrow\n");
    printf("Speed Down: Down Arrow\n");
    printf("Stop: %s\n", XKeysymToString(ks));

    // Get max speeds
    int pan_speed, tilt_speed;
    uint16_t status;
    if (cpi_ptcmd(cer, &status, OP_PAN_UPPER_SPEED_LIMIT_GET, &pan_speed) ||
        cpi_ptcmd(cer, &status, OP_TILT_UPPER_SPEED_LIMIT_GET, &tilt_speed))
        die("Basic unit queries failed.\n");

    bool running = true;
    int speed = 0;

    int prev_tilt_dir = 0;
    int prev_pan_dir = 0;

    while (running) {
        const bool w_pressed = key_is_pressed(XK_W);
        const bool s_pressed = key_is_pressed(XK_S);
        const bool a_pressed = key_is_pressed(XK_A);
        const bool d_pressed = key_is_pressed(XK_D);
        const bool up_pressed = key_is_pressed(XK_Up);
        const bool down_pressed = key_is_pressed(XK_Down);
        const float speed_p = (float)speed / 100;

        int tilt_dir = 0;
        if (w_pressed)
            tilt_dir = 1;
        if (s_pressed)
            tilt_dir = -1;

        if (prev_tilt_dir != tilt_dir) {
            cpi_ptcmd(cer, &status, OP_TILT_DESIRED_SPEED_SET, (int)((float)tilt_dir*(float)tilt_speed*speed_p));
            prev_tilt_dir = tilt_dir;
        }

        int pan_dir = 0;
        if (d_pressed)
            pan_dir = 1;
        if (a_pressed)
            pan_dir = -1;

        if (prev_pan_dir != pan_dir) {
            cpi_ptcmd(cer, &status, OP_PAN_DESIRED_SPEED_SET, (int)((float)pan_dir*(float)pan_speed*speed_p));
            prev_pan_dir = pan_dir;
        }

        int speed_dir = 0;
        if (up_pressed)
            speed_dir = 1;
        if (down_pressed)
            speed_dir = -1;

        speed = std::clamp(speed + 1 * speed_dir, 0, 100);
        if (speed_dir != 0)
            printf("Speed: %d\n", speed);

        if (key_is_pressed(ks)) {
            cpi_ptcmd(cer, &status, OP_PAN_DESIRED_SPEED_SET, pan_speed);
            cpi_ptcmd(cer, &status, OP_TILT_DESIRED_SPEED_SET, tilt_speed);
            cpi_ptcmd(cer, &status, OP_PAN_DESIRED_POS_SET, 0);
            cpi_ptcmd(cer, &status, OP_TILT_DESIRED_POS_SET, 0);
            running = false;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}