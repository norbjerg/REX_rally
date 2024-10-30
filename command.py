import os
import time
from abc import ABC

import numpy as np

from constants import Constants
from particle import Particle, ParticlesWrapper
import math_utils
import itertools

if Constants.World.running_on_arlo:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from robot import Robot

else:
    Robot = object

IS_ARLO = Constants.World.running_on_arlo
ROTATIONAL_SPEED = Constants.Robot.ROTATIONAL_SPEED
FORWARD_SPEED = Constants.Robot.FORWARD_SPEED


class Command(ABC):
    def __init__(self, robot, particles):
        self.robot: ControlWrapper = robot
        if particles is None:
            ParticlesWrapper(0, [])
        else:
            self.particles: ParticlesWrapper = particles

        self.start_time = None  # None means not started

        self.command_time = 0
        self.begun = False
        self.finished = False
        self.mov_dirs = (0, 0)
        self.power = 0
        self.angle = 0
        self.dist = 0
        self.avoidance_mode = False
        self.sonars = None

    def run_command(self):
        if self.robot.isArlo and self.dist > 0 and not self.finished:
            left, front, right = self.robot.read_sonars()
            if front < 10:
                self.robot.stop()
                self.avoidance_mode = True
                self.sonars = (left, front, right)
                return

        if self.finished is True:
            return

        if self.start_time is None and self.command_time == 0:
            self.finished = True
            self.begun = True
            return

        if self.start_time is None:
            self.start_time = time.time()
            self.robot.go_diff(self.power, self.power, *self.mov_dirs)
            self.begun = True

        elif time.time() - self.start_time >= self.command_time:
            self.robot.stop()
            self.particles.move_particles(self.dist, self.angle)
            self.particles.add_uncertainty(
                Constants.Robot.DISTANCE_NOISE, Constants.Robot.ANGULAR_NOISE
            )
            self.finished = True

    def check_sensors(self):
        self.sonars = self.robot.read_sonars()


# wraps robot for the purpose of interchangability with debug/Arlo
class ControlWrapper:
    def __init__(self, isArlo=False):
        self.robot = Robot()
        self.isArlo = isArlo

    def go_diff(self, l, r, L, R):
        if self.isArlo:
            self.robot.go_diff(l, r, L, R)
        else:
            print(f"executing command diff({l, r, L, R}).")

    def stop(self):
        if self.isArlo:
            self.robot.stop()

    def read_sonars(self):
        if self.isArlo:
            return (
                self.robot.read_left_ping_sensor(),
                self.robot.read_front_ping_sensor(),
                self.robot.read_right_ping_sensor(),
            )
        else:
            return (0, 0, 0)


# TODO: Make command abstract + implement angle handling from calibrate.py


class Rotate(Command):
    def __init__(self, robot, delta_angle, particles=None) -> None:
        super().__init__(robot, particles)
        self.start_time = None
        self.finished = False
        self.mov_dirs = (1, 0)
        self.power = 32
        angle = delta_angle
        if angle < 0:
            self.mov_dirs = self.mov_dirs[1], self.mov_dirs[0]
        if angle > 2 * np.pi or angle <= -2 * np.pi:
            angle %= 2 * np.pi
        if angle > np.pi:
            angle -= np.pi
            angle = -angle
        elif angle < -np.pi:
            angle += np.pi
            angle = -angle
        self.angle = angle

        self.command_time = abs(angle) / ROTATIONAL_SPEED


class Straight(Command):
    def __init__(self, robot, distance, particles=None) -> None:
        super().__init__(robot, particles)

        self.mov_dirs = (1, 1)
        self.power = 64

        if distance < 0:
            self.mov_dirs = (0, 0)

        self.dist = distance
        self.command_time = abs(distance) / FORWARD_SPEED


def too_close(left, right, front):
    return front < 200 or left < 200 or right < 200


class Wait(Command):
    def __init__(self, robot, grace_time, particles=None) -> None:
        super().__init__(robot, particles)
        self.command_time = grace_time
        self.power = 0


class Approach(Command):
    def __init__(self, landmark_pos, robot, particles):
        super().__init__(robot, particles)
        est_pos = particles.estimate_pose()
        dist, angle = math_utils.polar_diff(est_pos.getPos(), est_pos.getTheta(), landmark_pos)
        # approach command is actually an infinite list of commands
        self.sub_plan = (Straight(robot, 10, particles) for _ in itertools.count())
        
        self.current_command = next(self.sub_plan)
        self.run_command()
        

    def run_command(self):
        self.current_command.run_command()

        # export for future use
        self.begun = self.current_command.begun
        # self.finished = self.current_command.finished
        self.avoidance_mode = self.current_command.avoidance_mode
        self.check_sensors()

        _l_sonar, f_sonar, _r_sonar = self.sonars 
        if self.finished:
            return
        if f_sonar < 50:
            print("stopping in approach")
            self.finished = True
            self.robot.stop()
            return 
        if self.current_command.finished:
            self.current_command = next(self.sub_plan)




# testing code
if __name__ == "__main__":
    # arlo = None
    arlo = ControlWrapper(IS_ARLO)
    queue: list[Command] = [
        # (Rotate(arlo, np.deg2rad(90))),
        # (Straight(arlo, 100)),
        # (Wait(arlo, 5)),
        # (Straight(arlo, -100)),
        # (Rotate(arlo, -np.deg2rad(90)))
        Approach(np.array([0,0]), arlo, ParticlesWrapper(100,Constants.World.landmarks))
        ]

    while len(queue) > 0:
        command = queue.pop(0)
        command.run_command()
        while not command.finished:
            command.run_command()
    print("Commands finished")
