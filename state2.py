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
        self.goal_order = Constants.World.goal_order
        self.current_goal = 0

        if self.on_arlo:
            self._cam = camera.Camera(0, robottype="arlo", useCaptureThread=True)
        else:
            self._cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=True)

        self.info = info.Info()

        self.particles = particle.ParticlesWrapper(self.num_particles, self.landmarks)
        self.obstacles = dict()
        self.est_pos: Optional[Particle] = None
        self.est_theta: Optional[float] = None
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
            def gen_command():
                while True:
                    yield command.Wait(self.arlo, 2, self.outer_instance.particles)
                    yield command.Rotate(self.arlo, 0.5, self.outer_instance.particles)

            self.outer_instance.reset_particles()
            self.queue = iter(gen_command())
            self.current_command = next(self.queue)
            self.current_command.run_command()
            self.seen_landmarks = dict()
            self.resampled = False

        def update(self):
            objectIDs, dists, angles = self.cam.detect_aruco_objects(
                self.outer_instance.colour
            )  # Detect objects

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

                # Afsted ingen vrøvl!
                if (
                    len(self.seen_landmarks) >= 2
                    and self.outer_instance.goal_order[self.outer_instance.current_goal]
                    in objectIDs
                ):
                    self.outer_instance.set_state(RobotState.moving)

            useful_measurements = set(measurements).intersection(
                set(self.outer_instance.landmarkIDs)
            )
            useful_measurements_dict = {
                useful_key: measurements[useful_key] for useful_key in useful_measurements
            }
            self.seen_landmarks.update(useful_measurements_dict)

            if (
                len(useful_measurements) > 0 and self.resampled
            ):  # perform resampling on the particles
                self.outer_instance.particles.resample(useful_measurements_dict)
                self.resampled = False

            if len(self.seen_landmarks) >= 2:
                self.outer_instance.particles.resample(useful_measurements_dict)

            if self.current_command.finished:
                self.current_command = next(self.queue)

            self.current_command.run_command()

    class Checking:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance
            self.initialize()

        def initialize(self):
            self.goal = self.outer_instance.goals[
                self.outer_instance.goal_order[self.outer_instance.current_goal]
            ]

        def update(self):
            def gen_command(positions):
                yield command.Straight(self.outer_instance.arlo, 100, self.outer_instance.particles)

            self.outer_instance.est_pos = self.outer_instance.particles.estimate_pose()
            est_pos = self.outer_instance.est_pos

            # IDK OM DET HER VIRKER HIHI. TJEKKER BARE OM ROBOTTENS EST POS PEGER MOD MÅL
            """"
            estX, estY, estTheta = est_pos.getX(), est_pos.getY(), est_pos.getTheta()
            delta_x = self.outer_instance.goals[self.outer_instance.goal_order[self.outer_instance.current_goal]][0] - estX
            delta_y = self.outer_instance.goals[self.outer_instance.goal_order[self.outer_instance.current_goal]][1] - estY
            theta = np.arctan2(delta_y, delta_x)
            if est_pos.getTheta() > theta + Constants.Robot.ANGULAR_NOISE and est_pos.getTheta() < theta - Constants.Robot.ANGULAR_NOISE :
                print("found route")
                self.outer_instance.set_state(RobotState.moving)
            """

            est_pos = self.outer_instance.est_pos
            offset_vec = np.array([-np.sin(est_pos.getTheta()), np.cos(est_pos.getTheta())]) * (
                Constants.Obstacle.SHAPE_RADIUS_CM + 20
            )
            local_goal_pos = est_pos.getPos() - self.goal + offset_vec

            if self.outer_instance.route is not None:
                self.outer_instance.set_state(RobotState.moving)
                self.outer_instance.route = list(gen_command([local_goal_pos]))
            else:
                print("No route found")

    class Moving:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance
            self.left, self.right, self.front = self.outer_instance.arlo.read_sonars()

        def initialize(self):
            if command.too_close(self.left, self.right, self.front):
                self.outer_instance.set_state(
                    RobotState.avoidance, sonars=(self.left, self.front, self.right)
                )
                return
            else:
                self.current_command = command.Straight(
                    self.outer_instance.arlo, 50, self.outer_instance.particles
                )
                self.current_command.run_command()
            self.outer_instance.set_state(RobotState.lost)

        def update(self):
            if command.too_close(self.left, self.right, self.front):
                self.outer_instance.set_state(
                    RobotState.avoidance, sonars=(self.left, self.front, self.right)
                )
                return

    class Avoidance:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance

            self.initialize()

        def initialize(self):
            command.Straight(self.outer_instance.arlo, 0, self.outer_instance.particles)

        def update(self):
            self.left, self.right, self.front = self.outer_instance.arlo.read_sonars()

            if self.left > 30 and self.right > 30 and self.front > 200:
                self.outer_instance.set_state(RobotState.moving)
                return

            if self.right > self.left:
                command.Rotate(self.outer_instance.arlo, -np.pi / 4, self.outer_instance.particles)
            else:
                command.Rotate(self.outer_instance.arlo, np.pi / 4, self.outer_instance.particles)

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
            print("State: ", state.state)
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
