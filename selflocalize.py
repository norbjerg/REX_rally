import os
import sys
import time
from copy import copy
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
import numpy as np

# own imports:
# import scipy.stats as stats
from numpy import random

import camera
import particle
from command import Command
from constants import Constants
from info import *
from math_utils import normal, polar_diff

# from staterobot import StateRobot

# randomness:
rng = random.default_rng()

# Flags
showGUI = True  # Whether or not to open GUI windows
showPreview = True
onRobot = False  # Whether or not we are running on the Arlo robot


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


# # Some color constants in BGR format
# CRED = (0, 0, 255)
# CGREEN = (0, 255, 0)
# CBLUE = (255, 0, 0)
# CCYAN = (255, 255, 0)
# CYELLOW = (0, 255, 255)
# CMAGENTA = (255, 0, 255)
# CWHITE = (255, 255, 255)
# CBLACK = (0, 0, 0)

# # Landmarks.
# # The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
# world = Constants.World
# landmarks = world.landmarks
# landmarkIDs = world.landmarkIDs
# goal = world.goal


# def jet(x):
#     """Colour map for drawing particles. This function determines the colour of
#     a particle from its weight."""
#     r = (
#         (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (4.0 * x - 3.0 / 2.0)
#         + (x >= 5.0 / 8.0 and x < 7.0 / 8.0)
#         + (x >= 7.0 / 8.0) * (-4.0 * x + 9.0 / 2.0)
#     )
#     g = (
#         (x >= 1.0 / 8.0 and x < 3.0 / 8.0) * (4.0 * x - 1.0 / 2.0)
#         + (x >= 3.0 / 8.0 and x < 5.0 / 8.0)
#         + (x >= 5.0 / 8.0 and x < 7.0 / 8.0) * (-4.0 * x + 7.0 / 2.0)
#     )
#     b = (
#         (x < 1.0 / 8.0) * (4.0 * x + 1.0 / 2.0)
#         + (x >= 1.0 / 8.0 and x < 3.0 / 8.0)
#         + (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (-4.0 * x + 5.0 / 2.0)
#     )

#     return (255.0 * r, 255.0 * g, 255.0 * b)


# def draw_world(est_pose, particles, world):
#     """Visualization.
#     This functions draws robots position in the world coordinate system."""

#     # Fix the origin of the coordinate system
#     offsetX = 100
#     offsetY = 250

#     # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
#     ymax = world.shape[0]

#     world[:] = CWHITE  # Clear background to white

#     # Find largest weight
#     max_weight = 0
#     for particle in particles.particles:
#         max_weight = max(max_weight, particle.getWeight())

#     # Draw particles
#     for particle in particles.particles:
#         x = int(particle.getX() + offsetX)
#         y = ymax - (int(particle.getY() + offsetY))
#         colour = jet(particle.getWeight() / max_weight)
#         cv2.circle(world, (x, y), 2, colour, 2)
#         b = (
#             int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
#             ymax
#             - (int(particle.getY() + 15.0 * np.sin(particle.getTheta())) + offsetY),
#         )
#         cv2.line(world, (x, y), b, colour, 2)

#     # Draw landmarks
#     for i in range(len(landmarkIDs)):
#         ID = landmarkIDs[i]
#         lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
#         cv2.circle(world, lm, 5, landmark_colors[i], 2)

#     # Draw estimated robot pose
#     a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
#     b = (
#         int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
#         ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
#     )
#     cv2.circle(world, a, 5, CMAGENTA, 2)
#     cv2.line(world, a, b, CMAGENTA, 2)


# def do_direct_path(source_pos, source_theta, goal_pos):
#     distance, theta = polar_diff(source_pos, source_theta, goal_pos)
#     return Command(arlo, distance, theta)
# def do_direct_path(source_pos, source_theta, goal_pos):
#     distance, theta = polar_diff(source_pos, source_theta, goal_pos)
#     return Command(arlo, distance, theta)


# Main program #
if __name__ == "__main__":
    try:
        # Initialize particles
        num_particles = 600
        particles = particle.ParticlesWrapper(num_particles, landmarks)

        est_pose = particles.estimate_pose()  # The estimate of the robots current pose

        # Driving parameters
        velocity = 0.0  # cm/sec
        angular_velocity = 0.0  # radians/sec

        # # Initialize the robot (XXX: You do this)
        # if onRobot:
        #     arlo = robot.Robot()
        #     robot_state = StateRobot(arlo, particles)
        #     try_goto_goal = False
        # else:
        #     arlo = None
        #     robot_state = StateRobot(arlo, particles)

        # Draw map
        info = Info()
        info.draw_world(particles, est_pose)

        print("Opening and initializing camera")
        if isRunningOnArlo():
            # cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
            cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)
        else:
            # cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
            cam = camera.Camera(0, robottype="frindo", useCaptureThread=False)

        i = 0
        while True:
            # Move the robot according to user input (only for testing)
            action = cv2.waitKey(10)
            if action == ord("q"):  # Quit
                break

            if not isRunningOnArlo():
                if action == ord("w"):  # Forward
                    velocity += 2.0
                elif action == ord("x"):  # Backwards
                    velocity -= 2.0
                elif action == ord("s"):  # Stop
                    velocity = 0.0
                    angular_velocity = 0.0
                elif action == ord("a"):  # Left
                    angular_velocity += 0.1
                elif action == ord("d"):  # Right
                    angular_velocity -= 0.1

            # Use motor controls to update particles
            # XXX: Make the robot drive
            # XXX: You do this
            particles.move_particles(velocity, angular_velocity)
            particles.add_uncertainty(2.5, 0.125)

            # Fetch next frame
            colour = cam.get_next_frame()

            # Detect objects
            objectIDs, dists, angles = cam.detect_aruco_objects(colour)

            if (
                not isinstance(objectIDs, type(None))
                and not isinstance(dists, type(None))
                and not isinstance(angles, type(None))
            ):
                measurement = {}
                for objectID, dist, angle in zip(objectIDs, dists, angles):
                    measurement.setdefault(objectID, (np.inf, np.inf))
                    exist_dist, exist_angle = measurement[objectID]
                    if dist < exist_dist:
                        measurement[objectID] = (dist, angle)

                # intersection between measurments and world model
                useful_measurements = set(measurement).intersection(set(landmarkIDs))

                if len(useful_measurements) != 0:
                    # Compute particle weights
                    particles.particle_likelihoods(measurement)

                    # Resampling
                    # XXX: You do this
                    particles.resample_particles()

                    # Draw detected objects
                    cam.draw_aruco_objects(colour)
            else:
                # No observation - reset weights to uniform distribution
                particles.add_uncertainty(10, 0.1)
                particles.set_uniform_weights()

            est_pose = particles.estimate_pose()

            i += 1
            # if i % 100 == 0:
            #     command = do_direct_path(
            #         np.array([est_pose.getX(), est_pose.getY()]),
            #         est_pose.getTheta(),
            #         goal
            #         )
            # command.update_command_state()
            distance, theta = polar_diff(
                np.array([est_pose.getX(), est_pose.getY()]), est_pose.getTheta(), goal
            )

            # robot_state.update(
            #     particles,
            #     distance,
            #     theta,
            #     set(objectIDs).intersection(landmarks.keys()) if objectIDs is not None else None,
            # )

            # if robot_state.current_command is not None:
            #     if hasattr(robot_state.current_command, "velocity"):
            #         velocity = robot_state.current_command.velocity / 10
            #     else:
            #         velocity = 0.0
            #     if hasattr(robot_state.current_command, "rotation_speed"):
            #         angular_velocity = robot_state.current_command.rotation_speed / 10
            #     else:
            #         angular_velocity = 0.0

            if showGUI:
                # Draw map
                info.draw_world(particles, est_pose)

                info.show_frame(colour)

    finally:
        # Make sure to clean up even if an exception occurred

        # Close all windows
        cv2.destroyAllWindows()

        # Clean-up capture thread
        cam.terminateCaptureThread()
