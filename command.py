import os
import time
from abc import ABC

import numpy as np
from constants import Constants

if "PICAM" in os.environ:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from robot import Robot

    IS_ARLO = True
else:
    Robot = object
    IS_ARLO = False


ROTATIONAL_SPEED = Constants.Robot.ROTATIONAL_SPEED
FORWARD_SPEED = Constants.Robot.FORWARD_SPEED


class Command(ABC):
    def __init__(self, robot):
        self.robot = robot

        self.start_time = None  # None means not started

        self.command_time = 0
        self.begun = False
        self.finished = False
        self.mov_dirs = (0, 0)
        self.power = 0

    def run_command(self):
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
            self.finished = True



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


# TODO: Make command abstract + implement angle handling from calibrate.py


class Rotate(Command):
    def __init__(self, robot, delta_angle) -> None:
        super().__init__(robot)
        self.start_time = None
        self.finished = False
        self.mov_dirs = (1, 0)
        self.power = 32
        angle = delta_angle
        if angle < 0:
            self.mov_dirs = self.mov_dirs[1], self.mov_dirs[0]
        if angle > 2*np.pi or angle <= -2*np.pi:
            angle %= 2*np.pi
        if angle > np.pi:
            angle -= np.pi
            angle = -angle
        elif angle < -np.pi:
            angle += np.pi
            angle = -angle

        self.command_time = abs(angle) / ROTATIONAL_SPEED

        


class Straight(Command):
    def __init__(self, robot, distance) -> None:
        super().__init__(robot)

        self.mov_dirs = (1, 1)
        self.power = 64

        if distance < 0:
            self.mov_dirs = (0, 0)

        self.command_time = abs(distance) / FORWARD_SPEED



class Wait(Command):
    def __init__(self, robot, grace_time) -> None:
        super().__init__(robot)
        self.command_time = grace_time
        self.power = 0


# testing code
if __name__ == "__main__":
    # arlo = None
    arlo = ControlWrapper(IS_ARLO)
    queue: list[Command] = []
    queue.append(Rotate(arlo, np.deg2rad(90)))
    queue.append(Straight(arlo, 100))
    queue.append(Wait(arlo, 5))
    queue.append(Straight(arlo, -100))
    queue.append(Rotate(arlo, -np.deg2rad(90)))
    
    while len(queue) > 0:
        command = queue.pop(0)
        command.run_command()
        while not command.finished:
            command.run_command()
    print("Commands finished")
