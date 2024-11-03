import importlib
import math
import os
import time

import numpy as np
import pytest

os.environ["ROBOT_TEST_MODE"] = "1"
# NOTE: you must import local module after this line, to ensure test variables are used
import command
from constants import Constants
from particle import ParticlesWrapper
from state import State


def test_self_localize1() -> None:
    state = State()
    landmarks = {
        1: (0.0, 0.0),
        2: (0.0, 300.0),
        3: (400.0, 0.0),
        4: (400.0, 300.0),
    }
    measurements = {1: (150.0, 0.0)}  # see marker 1 in between 1 and 2
    state.particles.resample(measurements)
    est_pose = state.particles.estimate_pose()
    assert math.isclose(est_pose.getX(), 0, abs_tol=30)
    assert math.isclose(est_pose.getY(), 150, abs_tol=30)
    assert math.isclose(est_pose.getTheta(), np.deg2rad(-90), abs_tol=5)


def test_self_localize2() -> None:
    landmarks = {
        1: (0.0, 0.0),
        2: (0.0, 300.0),
    }
    for _ in range(10):
        particles = ParticlesWrapper(1200, landmarks)

        # see marker 1 in between 1 and 2, rotate 180 deg and see marker 2
        measurements = {1: (150.0, 0.0)}
        particles.resample(measurements)

        particles.move_particles(0, np.deg2rad(180))
        particles.add_uncertainty(Constants.Robot.DISTANCE_NOISE, Constants.Robot.ANGULAR_NOISE)

        measurements = {2: (150.0, 0.0)}
        particles.add_uncertainty(Constants.Robot.DISTANCE_NOISE, Constants.Robot.ANGULAR_NOISE)
        particles.resample(measurements)
        resample_times = 1
        est_pose = particles.estimate_pose()
        # while not est_pose.checkLowVarianceMinMaxes(5) and resample_times < 100:
        #     particles.add_uncertainty(Constants.Robot.DISTANCE_NOISE, Constants.Robot.ANGULAR_NOISE)
        #     particles.resample(measurements)
        #     est_pose = particles.estimate_pose()
        #     (min_x, max_x), (min_y, max_y) = est_pose.min_maxes
        #     print(max_x - min_x, max_y - min_y)
        #     print(est_pose.min_maxes)
        #     resample_times += 1

        assert math.isclose(est_pose.getX(), 0, abs_tol=20), est_pose.getPos()
        assert math.isclose(est_pose.getY(), 150, abs_tol=20), est_pose.getPos()
        assert math.isclose(
            est_pose.getTheta(), np.deg2rad(90), abs_tol=np.deg2rad(10)
        ), np.rad2deg(est_pose.getTheta())
    print(f"distance_noise: {distance_noise}, angular_noise: {angular_noise}")


@pytest.mark.parametrize("distance_noise", range(0, 100, 5))
@pytest.mark.parametrize(
    "angular_noise", range(0, int(np.deg2rad(360)) * 100, int(np.deg2rad(360) * 100) // 10)
)
def test_self_localize3(distance_noise: int, angular_noise: int) -> None:
    landmarks = {
        1: (0.0, 0.0),
        2: (0.0, 300.0),
    }
    for _ in range(10):
        particles = ParticlesWrapper(1200, landmarks)

        # see marker 1 in between 1 and 2, rotate 180 deg and see marker 2
        measurements = {1: (150.0, 0.0)}
        particles.resample(measurements)

        particles.move_particles(0, np.deg2rad(180))
        particles.add_uncertainty(distance_noise, angular_noise / 100)

        measurements = {2: (150.0, 0.0)}
        particles.add_uncertainty(distance_noise, angular_noise / 100)
        particles.resample(measurements)
        resample_times = 1
        est_pose = particles.estimate_pose()
        # while not est_pose.checkLowVarianceMinMaxes(5) and resample_times < 100:
        #     particles.add_uncertainty(Constants.Robot.DISTANCE_NOISE, Constants.Robot.ANGULAR_NOISE)
        #     particles.resample(measurements)
        #     est_pose = particles.estimate_pose()
        #     (min_x, max_x), (min_y, max_y) = est_pose.min_maxes
        #     print(max_x - min_x, max_y - min_y)
        #     print(est_pose.min_maxes)
        #     resample_times += 1

        assert math.isclose(est_pose.getX(), 0, abs_tol=20), est_pose.getPos()
        assert math.isclose(est_pose.getY(), 150, abs_tol=20), est_pose.getPos()
        assert math.isclose(
            est_pose.getTheta(), np.deg2rad(90), abs_tol=np.deg2rad(10)
        ), np.rad2deg(est_pose.getTheta())
    print(f"distance_noise: {distance_noise}, angular_noise: {angular_noise}")
