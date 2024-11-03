import numpy as np


class LocalConstants:
    class Robot:
        INCLUDE = True
        DISTANCE_NOISE = 35
        ANGULAR_NOISE = np.deg2rad(10)

    class Sensor:
        pass

    class Obstacle:
        pass

    class PID:
        ENABLE_PREVIEW = 0
        ENABLE_GUI = 0
        DRAW_PATH_BLOCKING = 0

    class PyPlot:
        pass

    class World:
        running_on_arlo = True
        num_particles = 600 * 2 
