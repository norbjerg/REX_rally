import os
import sys
from enum import Enum
from time import sleep, time
from typing import Optional

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from command import Command
from particle import Particle, ParticlesWrapper

LOW_VARIANCE = 0.1
MEDIUM_VARIANCE = 0.5
SEARCH_DEGREE = 10

class RobotState(Enum):
    following_path = 0
    is_checking = 1
    is_processing = 2

class VarianceState(Enum):
    low_variance = 0
    high_variance = 1
    no_variance = 2


class StateRobot:
    def __init__(self, arlo, particles):
        self.state = RobotState.is_checking
        self.particles: ParticlesWrapper = particles
        self.variance_state = VarianceState.no_variance
        self.variance = np.inf
        self.current_command: Optional[Command] = None
        self.next_command: Optional[Command] = None
        self.command_robot_state: RobotState = RobotState.is_checking
        self.arlo = arlo
        self.grace_time = 1
        self.processing_time = 7
        self.start_grace_time: Optional[float] = None

    def set_variance(self):
        mean_point = self.particles.estimate_pose()
        self.variance = np.var([np.sqrt((p.getX() - mean_point.getX())**2+(p.getY() - mean_point.getY())**2) for p in self.particles.particles]) / len(self.particles.particles)
        # self.variance = np.var([p.getWeight() for p in self.particles])
        if self.variance <= LOW_VARIANCE:
            self.variance_state = VarianceState.low_variance
        else:
            self.variance_state = VarianceState.high_variance

    def check_variance(self):
        self.set_variance()
        if self.variance_state == VarianceState.low_variance:
            self.state = RobotState.following_path
        else:
            self.state = RobotState.is_checking

    def grace(self, time_to_grace: Optional[float] = None):
        if self.start_grace_time is None:
            self.start_grace_time = time()
        elif time() - self.start_grace_time > (time_to_grace or self.grace_time):
            self.start_grace_time = None
            return True
        return False

    def compute_next_action(self, dist, angle, found_ids):
        if self.current_command is not None and self.current_command.finished:
            if self.grace() is False:
                return
        elif self.state == RobotState.is_processing:
            if self.grace(self.processing_time) is False:
                return
        if self.state == RobotState.following_path:
            if self.command_robot_state == self.state and self.current_command is not None and self.current_command.finished is False:
                self.current_command.update_command_state()
            else:
                self.current_command = Command(self.arlo, dist, angle)
                self.command_robot_state = RobotState.following_path
        elif self.state == RobotState.is_checking:
            if found_ids is not None and len(found_ids) > 1:
                self.state = RobotState.is_processing
            elif self.command_robot_state == self.state and self.current_command is not None and self.current_command.finished is False:
                self.current_command.update_command_state()
            else:
                self.current_command = Command(self.arlo, 0, np.deg2rad(SEARCH_DEGREE))
                self.command_robot_state = RobotState.is_checking

    def update(self, particles, dist, angle, found_ids):
        self.particles = particles
        self.check_variance()
        print("variance",self.variance, self.variance_state, self.state, self.command_robot_state, found_ids, (self.current_command.angle, self.current_command.distance, self.current_command.finished) if self.current_command is not None else None)
        self.compute_next_action(dist, angle, found_ids)

