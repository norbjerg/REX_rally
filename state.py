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
        self.particles_reset = True

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
        old_particles = self.particles.particles[: len(self.particles.particles) // (2/3)]
        new_particles = particle.ParticlesWrapper(self.num_particles // 4, self.landmarks)
        new_particles.particles.extend(old_particles)
        self.particles = new_particles
        self.particles.num_particles = self.num_particles

    class Lost:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.arlo = outer_instance.arlo
            self.outer_instance = outer_instance
            self.rotate_amount = np.deg2rad(15)
            self.initialize()

        def initialize(self) -> None:
            print("lost")

            def gen_command(degree=self.rotate_amount):
                while True:
                    yield command.Wait(self.arlo, 1, self.outer_instance.particles)
                    yield command.Rotate(self.arlo, degree, self.outer_instance.particles)

            if self.outer_instance.est_pos is not None:
                pos = self.outer_instance.est_pos.getPos()
                if (
                    pos[0] < Constants.World.landmarkMin[0] - Constants.World.threshold_outside
                    or pos[0] > Constants.World.landmarkMax[0] + Constants.World.threshold_outside
                    or pos[1] < Constants.World.landmarkMin[1] - Constants.World.threshold_outside
                    or pos[1] > Constants.World.landmarkMax[1] + Constants.World.threshold_outside
                ):
                    self.outer_instance.reset_particles()

            self.inner_instance_particles = self.outer_instance.particles
            self.queue = iter(gen_command())
            self.current_command = next(self.queue)
            self.current_command.run_command()
            self.initial_resample = True
            self.measurements = dict()
            self.rotated_times = 0

            if self.outer_instance.particles_reset:
                self.outer_instance.reset_particles()

        def update(self):
            # Check if we have reached all targets
            target_id = self.outer_instance.goal_order[self.outer_instance.current_goal]
            if target_id == -1:
                print("Finished all targets. Stopping")
                command.Straight(self.outer_instance.arlo, 0, self.outer_instance.particles)
                return

            # Detect objects and update measurements and current measurements
            objectIDs, dists, angles = self.cam.detect_aruco_objects(
                self.outer_instance.colour
            )  # Detect objects
            current_measurements = dict()  # measurements we see now
            if not isinstance(objectIDs, type(None)):
                for objectID, dist, angle in zip(objectIDs, dists, angles):
                    self.measurements[objectID] = (dist, angle)
                    current_measurements[objectID] = (dist, angle)

            # Resample particles if we have measurements
            if len(current_measurements) > 0:
                # If we have one measurement we update once
                if len(self.measurements) == 1 and self.initial_resample:
                    self.outer_instance.particles.update(self.measurements)
                    self.outer_instance.est_pos = self.outer_instance.particles.estimate_pose()
                    self.initial_resample = False

                # If we have more than one measurement we update continuously
                elif len(self.measurements) >= 2:
                    self.outer_instance.particles.update(current_measurements)
                    self.outer_instance.est_pos = self.outer_instance.particles.estimate_pose()

            # Shortcut: If camera sees target now, then we move forward
            if target_id in current_measurements:
                print("Found target")
                if self.measurements[target_id][0] > 40:
                    print("Target too far. Moving toward it")
                    self.outer_instance.particles_reset = False
                    self.outer_instance.set_state(RobotState.moving)

            # Shortcut: Saw target earlier, but lost it
            if (
                target_id in self.measurements
                and self.outer_instance.est_pos.checkLowVarianceMinMaxes()
            ):
                currentX_pos, currentY_pos = self.outer_instance.est_pos.getPos()
                targetX_pos, targetY_pos = self.outer_instance.landmarks[
                    self.outer_instance.goal_order[self.outer_instance.current_goal]
                ]

                # If distance and low variance we believe it was close enough to target
                dist = math_utils.distance(currentX_pos, currentY_pos, targetX_pos, targetY_pos)
                if dist < 40:
                    print("Saw target earlier")
                    self.outer_instance.particles_reset = True
                    self.outer_instance.current_goal += 1
                    self.outer_instance.set_state(RobotState.lost)

                # If it were to far we rotate the robot to find it again
                angle = math_utils.angle_diff(currentX_pos, currentY_pos, targetX_pos, targetY_pos)
                if (
                    angle < self.outer_instance.est_pos.getTheta() + 0.2
                    and angle > self.outer_instance.est_pos.getTheta() - 0.2
                ):
                    self.outer_instance.particles_reset = True
                    self.outer_instance.set_state(RobotState.moving)

            if (
                self.outer_instance.est_pos is not None
                and self.outer_instance.est_pos.checkLowVarianceMinMaxes()
            ):
                currentX_pos, currentY_pos = self.outer_instance.est_pos.getPos()
                targetX_pos, targetY_pos = self.outer_instance.landmarks[
                    self.outer_instance.goal_order[self.outer_instance.current_goal]
                ]
                angle = math_utils.angle_diff(currentX_pos, currentY_pos, targetX_pos, targetY_pos)
                if (
                    angle < self.outer_instance.est_pos.getTheta() + 0.2
                    and angle > self.outer_instance.est_pos.getTheta() - 0.2
                ):
                    self.outer_instance.particles_reset = True
                    self.outer_instance.set_state(RobotState.moving)
                    return

            # Rotate and increment rotation counter
            if self.current_command.finished:
                self.current_command = next(self.queue)
                self.rotated_times += self.rotate_amount

            # If we have rotated 360 degrees, takes takes from est_pos to goal
            if self.rotated_times >= np.deg2rad(360) // self.rotate_amount:
                currentX_pos, currentY_pos = self.outer_instance.est_pos.getPos()
                targetX_pos, targetY_pos = self.outer_instance.landmarks[
                    self.outer_instance.goal_order[self.outer_instance.current_goal]
                ]
                angle = math_utils.angle_diff(currentX_pos, currentY_pos, targetX_pos, targetY_pos)
                if (
                    angle < self.outer_instance.est_pos.getTheta() + 0.2
                    and angle > self.outer_instance.est_pos.getTheta() - 0.2
                ):
                    self.outer_instance.particles_reset = True
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
            def gen_command():
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
            self.distance = 40
            self.initialize()

        def initialize(self):
            def gen_command():
                while 1:
                    yield command.Straight(
                        self.outer_instance.arlo, 1, self.outer_instance.particles
                    )

            self.commands = iter(gen_command())
            self.current_command = next(self.commands)
            self.startTime = time.time()

        def update(self):
            self.left, self.right, self.front = self.outer_instance.arlo.read_sonars()
            if self.left < 350 or self.right < 350 or self.front < 350:
                self.outer_instance.set_state(RobotState.avoidance)
                return

            if self.outer_instance.particles_reset == False:
                currentX_pos, currentY_pos = self.outer_instance.est_pos.getPos()
                targetX_pos, targetY_pos = self.outer_instance.landmarks[
                    self.outer_instance.goal_order[self.outer_instance.current_goal]
                ]
                dist = math_utils.distance(currentX_pos, currentY_pos, targetX_pos, targetY_pos)
                if dist < 40:
                    print("Found target")
                    self.outer_instance.particles_reset = True
                    self.outer_instance.current_goal += 1
                    self.outer_instance.set_state(RobotState.lost)
                    return

            if not self.outer_instance.est_pos.checkLowVarianceMinMaxes():
                self.outer_instance.set_state(RobotState.lost)
                return

            if time.time() - self.startTime > 5:
                self.commands = command.Straight(
                    self.outer_instance.arlo, 0, self.outer_instance.particles
                )
                self.outer_instance.set_state(RobotState.lost)
                return

            if self.current_command.finished:
                self.current_command = next(self.commands)
            self.current_command.run_command()

    class Avoidance:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.outer_instance = outer_instance
            self.initialize()

        def initialize(self):
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
