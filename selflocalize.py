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



# Main program #
if __name__ == "__main__":
    try:
        # Initialize particles
        num_particles = 600
        particles = particle.ParticlesWrapper(num_particles, landmarks)
        print(landmarks)

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
            
            print(particles.estimate_pose().getPos())
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
