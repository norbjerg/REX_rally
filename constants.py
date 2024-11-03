import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from local_constants import LocalConstants
except ImportError:
    LocalConstants = None


onRobot = False


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("../../../../Arlo/python")


try:
    if isRunningOnArlo():
        import robot

        onRobot = True
    else:
        onRobot = False
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False


class Constants:
    class Robot:
        if "PICAM" in os.environ:
            INCLUDE = True
        else:
            INCLUDE = False
        MAX_VOLT = 12
        MAX_RPM = 100
        DIAMETER = 395  # mm +- 5 mm
        RADIUS = DIAMETER / 2
        CIRCUMFERENCE = 39.5 * 3.14  # cm
        SHAPE = (50, 50)  # Around 50 cm before the landmarks are visible in the camera
        WHEEL_DIAMETER = 155  # mm
        WHEEL_CIRCUMFERENCE = WHEEL_DIAMETER * 3.14  # cm
        QUARTER_TURN_64 = 0.725  # sleep
        # FORWARD_SPEED = 100 / 2.6  # cm/s 64 power
        FORWARD_SPEED = 100 / 5  # cm/s 55 power
        #ROTATIONAL_SPEED = 0.85  # np.deg2rad(360 / 7.3)  # rad/s
        ROTATIONAL_SPEED = 0.60
        DISTANCE_NOISE = 5  # cm
        ANGULAR_NOISE = 0.1  # rad
        CTRL_RANGE = [-20, 20]  # cm

    class Sensor:
        MAX_SPEED = 100
        HALF_SPEED = MAX_SPEED / 2
        MIN_SPEED = 30
        THRESHOLD = 100
        FINDING_SLEEP = 0.5
        FRONT = 0
        BACK = 1
        LEFT = 2
        RIGHT = 3

    class Obstacle:
        SHAPE = [145, 145]  # in mm
        SHAPE_MIN = 220  # in mm
        # SHAPE_MAX = 250  # in mm
        SHAPE_MAX = 100  # in mm
        SHAPE_RADIUS = SHAPE_MAX / 2
        SHAPE_RADIUS_CM = SHAPE_RADIUS / 10

    class PID:
        if "PICAM" in os.environ:
            CAMERA_MODEL = "picam"
        else:
            CAMERA_MODEL = "webcam"
        DOWNSCALE = 0
        SCREEN_RESOLUTION = (1640, 1232)
        # FOCALLENGTH_ARR = [1300, 640]
        FOCALLENGTH = 1300.0
        MARKER_HEIGHT = 145.0
        DISTORTION_COEFFICIENT = np.array([0, 0, 0, 0, 0], dtype=float)
        CAMERA_MATRIX = np.array(
            [
                [FOCALLENGTH, 0, SCREEN_RESOLUTION[0] / 2],
                [0, FOCALLENGTH, SCREEN_RESOLUTION[1] / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        PREVIEW_DOWNSCALE = 0
        PREVIEW_DIMENSIONS = (
            SCREEN_RESOLUTION[0] // (2**PREVIEW_DOWNSCALE),
            SCREEN_RESOLUTION[1] // (2**PREVIEW_DOWNSCALE),
        )
        ENABLE_PREVIEW = 0
        ENABLE_GUI = 1
        CAMERA_FPS = 24
        DRAW_PATH_BLOCKING = 1

    class PyPlot:
        valid_interactive_backends = [
            "GTK3Cairo",  # blocking
            "QtAgg",
            "QtCairo",
            "Qt5Agg",
            "Qt5Cairo",
        ]
        interactive_backend = "Qt5Agg"

    class World:
        landmarks = {
            1: (0.0, 0.0),
            2: (0.0, 300.0),
            3: (400.0, 0.0),
            4: (400.0, 300.0),
        }
        landmarkIDs = list(landmarks)
        goals = [np.array(pos) for id, pos in landmarks.items()]
        landmarkMin = (min([pos[0] for pos in goals]), min([pos[1] for pos in goals]))
        landmarkMax = (max([pos[0] for pos in goals]), max([pos[1] for pos in goals]))
        threshold_outside = 40

        goal_order = [1, 2, 3, 4, 1, -1]
        num_particles = 400
        running_on_arlo = "PICAM" in os.environ
        draw_particles = True


if LocalConstants is not None:
    for const_class in [
        Constants.Robot,
        Constants.Sensor,
        Constants.Obstacle,
        Constants.PID,
        Constants.PyPlot,
    ]:
        local_const_class = getattr(LocalConstants, const_class.__name__, None)
        if local_const_class is not None:
            for attr, value in const_class.__dict__.items():
                if not attr.startswith("__"):
                    local_value = getattr(local_const_class, attr, None)
                    if local_value is not None:
                        setattr(const_class, attr, local_value)
