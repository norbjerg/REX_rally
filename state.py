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
        self._checking = None  # self.Checking(self)
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
            self.rotate_amount = np.deg2rad(15)

            self.initialize()

        def initialize(self) -> None:
            print("lost")

            def gen_command():
                while True:
                    yield command.Wait(self.arlo, 1, self.outer_instance.particles)
                    yield command.Rotate(
                        self.arlo, self.rotate_amount, self.outer_instance.particles
                    )

            if self.outer_instance.est_pos is not None:
                pos = self.outer_instance.est_pos.getPos()
                if (
                    pos[0] < Constants.World.landmarkMin[0] - Constants.World.threshold_outside
                    or pos[0] > Constants.World.landmarkMax[0] + Constants.World.threshold_outside
                    or pos[1] < Constants.World.landmarkMin[1] - Constants.World.threshold_outside
                    or pos[1] > Constants.World.landmarkMax[1] + Constants.World.threshold_outside
                ):
                    self.outer_instance.reset_particles()

            self.queue = iter(gen_command())
            self.current_command = next(self.queue)
            self.current_command.run_command()
            self.seen_landmarks = dict()
            self.initial_resample = True
            self.rotated_times = 0
            self.measurements = dict()

        def update(self):
            objectIDs, dists, angles = self.cam.detect_aruco_objects(
                self.outer_instance.colour
            )  # Detect objects
            target_id = self.outer_instance.goal_order[self.outer_instance.current_goal]

            if target_id == -1:
                print("Finished all targets. Stopping")
                command.Straight(self.outer_instance.arlo, 0, self.outer_instance.particles)
                return

            if (
                not isinstance(objectIDs, type(None))
                and not isinstance(dists, type(None))
                and not isinstance(angles, type(None))
            ):
                for objectID, dist, angle in zip(objectIDs, dists, angles):
                    # measurements.setdefault(objectID, (np.inf, np.inf))
                    self.measurements[objectID] = (dist, angle)

            # if target_id in self.measurements:
            #     print("Found target")
            #     if self.measurements[target_id][0] > 80:
            #         self.outer_instance.set_state(RobotState.moving, togoal_theta=0)

            if (
                self.outer_instance.est_pos is not None
                and self.outer_instance.est_pos.checkLowVarianceMinMaxes()
            ):
                print("Low variance found")
                dist = np.linalg.norm(
                    self.outer_instance.est_pos.getPos()
                    - np.array(
                        self.outer_instance.landmarks[
                            self.outer_instance.goal_order[self.outer_instance.current_goal]
                        ]
                    )
                )
                self.outer_instance.est_pos.getPos()
                # find angle to target
                togoal_dist, togoal_theta = math_utils.polar_diff(
                    self.outer_instance.est_pos.getPos(),
                    self.outer_instance.est_pos.getTheta(),
                    self.outer_instance.landmarks[
                        self.outer_instance.goal_order[self.outer_instance.current_goal]
                    ],
                )

                print("dist from target", dist)
                if dist < 80:
                    print(
                        "Found target reached. Moving to next target",
                        self.outer_instance.goal_order[self.outer_instance.current_goal + 1],
                    )
                    self.outer_instance.current_goal += 1
                else:
                    print("Found target, but too far away. Moving to target")
                    self.outer_instance.set_state(
                        RobotState.moving, togoal_dist=togoal_dist / 2, togoal_theta=togoal_theta
                    )

            if len(self.measurements) == 1:
                if self.initial_resample:
                    self.outer_instance.particles.update(self.measurements)
                    self.initial_resample = False

            if len(self.measurements) >= 2:
                self.outer_instance.particles.update(self.measurements)

            if self.current_command.finished:
                self.current_command = next(self.queue)
                self.rotated_times += 1

            if self.rotated_times >= np.deg2rad(360) // self.rotate_amount:
                self.outer_instance.est_pos = self.outer_instance.particles.estimate_pose()
                x1, y1 = self.outer_instance.est_pos.getX(), self.outer_instance.est_pos.getY()
                x2, y2 = (
                    self.outer_instance.landmarks[
                        self.outer_instance.goal_order[self.outer_instance.current_goal]
                    ][0],
                    self.outer_instance.landmarks[
                        self.outer_instance.goal_order[self.outer_instance.current_goal]
                    ][1],
                )
                delta_x = x2 - x1
                delta_y = y2 - y1

                togoal_theta = np.arctan2(delta_y, delta_x)
                if togoal_theta < 0:
                    togoal_theta += 2 * np.pi

                est_theta = self.outer_instance.est_pos.getTheta()
                if est_theta < togoal_theta + 0.3 and est_theta > togoal_theta - 0.3:
                    print("Rotated fully, trying to move randomly")
                    self.outer_instance.set_state(RobotState.moving)

            self.current_command.run_command()

    class Checking:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance
            self.initialize()

        def initialize(self):
            print("checking")
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

        def initialize(self, togoal_theta=0, togoal_dist=-10):
            print("moving")

            self.togoal_theta = togoal_theta
            self.togoal_dist = togoal_dist

            def gen_command():
                yield command.Rotate(
                    self.outer_instance.arlo, self.togoal_theta, self.outer_instance.particles
                )
                yield command.Straight(
                    self.outer_instance.arlo, togoal_dist, self.outer_instance.particles
                )

            self.commands = iter(gen_command())
            self.current_command = next(self.commands)

        def update(self):
            if self.current_command.finished:
                next_command = next(self.commands, None)
                if next_command is None:
                    self.outer_instance.set_state(RobotState.lost)
                    return
                else:
                    self.current_command = next_command
                self.current_command.run_command()
            else:
                self.current_command.run_command()

    class Avoidance:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance
            self.initialize(1)

        def initialize(self, first=0):
            if not first:
                print("avoidance")
            command.Straight(self.outer_instance.arlo, 0, self.outer_instance.particles)
            self.current_command = None

        def update(self):
            self.left, self.right, self.front = self.outer_instance.arlo.read_sonars()

            if self.current_command is None:
                if self.right > self.left:
                    self.current_command = command.Rotate(
                        self.outer_instance.arlo, 0.5, self.outer_instance.particles
                    )
                else:
                    self.current_command = command.Rotate(
                        self.outer_instance.arlo, -0.5, self.outer_instance.particles
                    )
            elif self.current_command.finished:
                self.outer_instance.set_state(RobotState.lost)
            else:
                self.current_command.run_command()

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

    def next_target(self):
        self.current_goal += 1

    def update(self):
        self.show_gui()
        self.current_state.update()
        self.est_pos = self.particles.estimate_pose()


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
