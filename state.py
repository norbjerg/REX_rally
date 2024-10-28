import time
from enum import Enum
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

import camera
import command
import info
import math_utils
import particle
import rrt
from constants import Constants
from particle import Particle, ParticlesWrapper


class RobotState(Enum):
    lost = 0
    moving = 1
    checking = 2
    avoidance = 3


class State:
    def __init__(self) -> None:
        self.on_arlo = Constants.World.running_on_arlo
        self.landmarks = Constants.World.landmarks
        self.landmarkIDs = Constants.World.landmarkIDs
        self.num_particles = Constants.World.num_particles
        self.show_preview = Constants.PID.ENABLE_PREVIEW

        self.state = RobotState.lost
        self.particles = ParticlesWrapper(Constants.World.num_particles, Constants.World.landmarks)
        self._cam: camera.Camera
        self.WIN_RF1 = "Robot view"
        self.arlo = command.ControlWrapper(self.on_arlo)
        self.goals = Constants.World.goals

        if self.on_arlo:
            self._cam = camera.Camera(0, robottype="arlo", useCaptureThread=True)
        else:
            self._cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=True)

        self.info = info.Info()

        self.particles = particle.ParticlesWrapper(self.num_particles, self.landmarks)
        self.obstacles = dict()
        self.est_pos: Optional[Particle] = None
        self.route: Optional[list[command.Command]] = None

        self._lost = self.Lost(self)
        self._moving = self.Moving(self)
        self._checking = self.Checking(self)
        self._avoidance = self.Avoidance(self)
        self.current_state = self._lost

    def show_gui(self):
        self.next_frame()
        est_pose = self.particles.estimate_pose()

        self.info.draw_world(self.particles, est_pose)
        self.info.show_frame(self.colour)

    @property
    def cam(self):
        if self._cam is None:
            if self.on_arlo:
                self._cam = camera.Camera(0, robottype="arlo", useCaptureThread=True)
            else:
                self._cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=True)
        return self._cam

    def next_frame(self):
        self.colour = self._cam.get_next_frame()
        return self.colour

    def reset_particles(self):
        self.particles = particle.ParticlesWrapper(self.num_particles, self.landmarks)

    class Lost:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.arlo = outer_instance.arlo
            self.outer_instance = outer_instance

            self.initialize()

        def initialize(self) -> None:
            print("lost")

            def gen_command():
                while True:
                    yield command.Wait(self.arlo, 2, self.outer_instance.particles)
                    yield command.Rotate(self.arlo, 0.5, self.outer_instance.particles)

            self.queue = iter(gen_command())
            self.current_command = next(self.queue)

            self.outer_instance.reset_particles()

            # Movement command
            self.current_command.run_command()
            self.seen_landmarks = dict()

        def update(self):
            # Detect objects
            objectIDs, dists, angles = self.cam.detect_aruco_objects(self.outer_instance.colour)

            measurements = dict()

            # Pick the closest marker
            if (
                not isinstance(objectIDs, type(None))
                and not isinstance(dists, type(None))
                and not isinstance(angles, type(None))
            ):
                for objectID, dist, angle in zip(objectIDs, dists, angles):
                    measurements.setdefault(objectID, (np.inf, np.inf))
                    exist_dist, exist_angle = measurements[objectID]
                    if dist < exist_dist:
                        measurements[objectID] = (dist, angle)

            useful_measurements = set(measurements).intersection(
                set(self.outer_instance.landmarkIDs)
            )
            useful_measurements_dict = {
                useful_key: measurements[useful_key] for useful_key in useful_measurements
            }
            self.seen_landmarks.update(useful_measurements_dict)

            self.outer_instance.obstacles = (  # TODO: find a way to get obstacles in world coordinates
                {
                    obstacle_key: measurements[obstacle_key]
                    for obstacle_key in set(measurements).difference(
                        set(self.outer_instance.landmarkIDs)
                    )
                }
            )

            if len(useful_measurements) > 0:  # perform resampling on the particles
                self.outer_instance.particles.update(useful_measurements_dict)

            if len(self.seen_landmarks) >= 2:  # Go to checking when possible to localize
                self.outer_instance.set_state(RobotState.checking)
                return

            if self.current_command.finished:
                self.current_command = next(self.queue)

            self.current_command.run_command()

    class Checking:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance
            self.initialize()

        def initialize(self):
            print("checking")
            self.goal = self.outer_instance.goals[0]

        def update(self):
            def gen_command(positions):
                self.outer_instance.est_pos = self.outer_instance.particles.estimate_pose()
                for position in positions:
                    dist, angle = math_utils.polar_diff(
                        self.outer_instance.est_pos.getPos(),
                        self.outer_instance.est_pos.getTheta(),
                        position,
                    )
                    yield command.Rotate(
                        self.outer_instance.arlo, angle, self.outer_instance.particles
                    )
                    for _ in range(4):
                        yield command.Straight(
                            self.outer_instance.arlo, dist / 4, self.outer_instance.particles
                        )
                        yield command.Wait(
                            self.outer_instance.arlo, 0.1, self.outer_instance.particles
                        )

            self.outer_instance.est_pos = self.outer_instance.particles.estimate_pose()
            est_pos = self.outer_instance.est_pos
            offset_vec = np.array([-np.sin(est_pos.getTheta()), np.cos(est_pos.getTheta())]) * (
                Constants.Obstacle.SHAPE_RADIUS_CM + 20
            )
            local_goal_pos = est_pos.getPos() - self.goal + offset_vec

            if self.outer_instance.route is not None:
                self.outer_instance.set_state(RobotState.moving)
            else:
                print("found route")

                self.outer_instance.route = list(gen_command([local_goal_pos]))

    class Moving:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance

        def initialize(self):
            print("moving")
            self.current_command = (
                self.outer_instance.route.pop()
            )  # route should not be None at this point
            self.current_command.run_command()

        def update(self):
            if len(self.outer_instance.route) == 0:
                self.outer_instance.goals.pop(0)
                self.outer_instance.set_state(RobotState.lost)
            elif self.current_command.avoidance_mode:
                self.outer_instance.set_state(
                    RobotState.avoidance, sonars=self.current_command.sonars
                )
            elif self.current_command.finished:
                self.current_command = self.outer_instance.route.pop()
                self.outer_instance.set_state(RobotState.checking)
            else:
                self.current_command.run_command()

    class Avoidance:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance
            self.initialize((0, 0, 0))

        def initialize(self, sonars):
            print("avoidance")
            self.free_threshold = 50
            self.sonars = sonars
            self.max_var_backwards = 50
            self.state_index = 0
            # 0: mandatory backwards (30 cm)
            # 1: variable backwards
            # 2: right turn (45 deg)
            # 3: move forward mandatory backwards distance
            # 4: left turn same angle as 2.
            self.commands = [
                command.Straight(self.outer_instance.arlo, -30, self.outer_instance.particles),
                command.Straight(
                    self.outer_instance.arlo, -self.max_var_backwards, self.outer_instance.particles
                ),
                command.Rotate(self.outer_instance.arlo, np.pi / 4, self.outer_instance.particles),
                command.Straight(self.outer_instance.arlo, 30, self.outer_instance.particles),
                command.Rotate(self.outer_instance.arlo, -np.pi / 4, self.outer_instance.particles),
            ]

        def update(self):
            if self.state_index == 0:
                if (
                    self.commands[self.state_index].finished
                    or self.commands[self.state_index].avoidance_mode
                ):
                    self.state_index += 1
                    self.commands[self.state_index].run_command()
                else:
                    self.commands[self.state_index].run_command()

    def set_state(self, state: RobotState, **kwargs):
        self.state = state
        self.current_state = (
            self.lost or self.moving or self.checking or self.avoidance or self._lost
        )
        self.current_state.initialize(**kwargs)

    @property
    def lost(self):
        return self._lost if self.state == RobotState.lost else None

    @property
    def moving(self):
        return self._moving if self.state == RobotState.moving else None

    @property
    def checking(self):
        return self._checking if self.state == RobotState.checking else None

    @property
    def avoidance(self):
        return self._avoidance if self.state == RobotState.avoidance else None

    def update(self):
        self.show_gui()
        self.current_state.update()


if __name__ == "__main__":
    state = State()
    try:
        while True:
            action = cv2.waitKey(10)
            if action == ord("q"):
                break
            state.update()

    finally:
        # Make sure to clean up even if an exception occurred

        # Close all windows
        cv2.destroyAllWindows()

        # Clean-up capture thread
        state.cam.terminateCaptureThread()
