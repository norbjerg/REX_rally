import os
import sys
import time
from copy import copy
from timeit import default_timer as timer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
import numpy as np
import particle
from command import Command

# own imports:
# import scipy.stats as stats
from numpy import random
from staterobot import StateRobot

import camera

# randomness:
rng = random.default_rng()

# Flags
showGUI = True  # Whether or not to open GUI windows
showPreview = False
onRobot = True  # Whether or not we are running on the Arlo robot


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




# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarks = {
    2: (0.0, 0.0),  # Coordinates for landmark 1
    3: (180.0, 0.0)  # Coordinates for landmark 2
}
landmarkIDs = list(landmarks)
goal = np.array([100.0, 100.])


landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

def normal(x, mu, sigma):
    y =  np.exp(-((x-mu)**2/(2.0*sigma**2)))/np.sqrt(2*np.pi*sigma**2)
    return y 

def polar_diff(src_x, src_theta, target_x):
    # Distance from particle to landmark
    particle_dist = np.linalg.norm(src_x - target_x)

    # Direction from particle to landmark (unit vector)
    landmark_e = (target_x - src_x) / particle_dist

    # Particle's heading as unit vector
    particle_e = np.array([np.cos(src_theta), np.sin(src_theta)])
    ortho_particle_e = np.array([-np.sin(src_theta), np.cos(src_theta)])  # Orthogonal vector

    # Calculate the relative angle between particle's heading and the landmark
    particle_theta = (np.sign(np.dot(landmark_e, ortho_particle_e)) *
        np.arccos(np.dot(landmark_e, particle_e)) )

    return particle_dist, particle_theta

def particle_likelihood(particle, measurements):
    likelihood = 1

    part_pos = np.array([particle.getX(), particle.getY()])

    for l_id, (m_dist, m_ang) in measurements.items():
        # If observed landmark is not known, ignore it
        if l_id not in landmarks.keys():
            # print(f"alert: {l_id} seen and ignored")
            continue
        else:
            # print(f" {l_id} {m_dist} {np.rad2deg(m_ang)}")
            pass
        
        land_pos = np.array(landmarks[l_id])

        dist, theta = polar_diff(part_pos, particle.getTheta(), land_pos)
        likelihood *= normal(theta - m_ang, 0 , 0.25) + sys.float_info.min*2
        likelihood *= normal(dist - m_dist, 0, 10)   + sys.float_info.min*2

    return likelihood

def resample_particles(particles, num_particles):
    pmf = np.zeros(len(particles), dtype=np.float64)
    for i, p in enumerate(particles):
        pmf[i] = p.getWeight()
    # choice as indexes:
    choices = rng.choice(len(particles), num_particles, p=pmf)
    new_arr = []
    for i in choices:
        new_arr.append(copy(particles[i]))
    return new_arr




def jet(x):
    """Colour map for drawing particles. This function determines the colour of
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX,
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX,
                                 ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

def do_direct_path(source_pos, source_theta, goal_pos):
    distance, theta = polar_diff( source_pos, source_theta, goal_pos)
    return Command(arlo, distance, theta)
    

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points.
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles


# Main program #
try:
    if showGUI:
        # Open windows
        if showPreview:
            WIN_RF1 = "Robot view"
            cv2.namedWindow(WIN_RF1)
            cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)


    # Initialize particles
    num_particles = 600
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    if onRobot:
        arlo = robot.Robot()
        robot_state = StateRobot(arlo, particles)
        try_goto_goal = False
    else:
        arlo = None
        robot_state = StateRobot(arlo, particles)

        

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        #cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        cam = camera.Camera(0, robottype='frindo', useCaptureThread=False)

    i = 0
    while True:

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break

        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 2.0
            elif action == ord('x'): # Backwards
                velocity -= 2.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.1
            elif action == ord('d'): # Right
                angular_velocity -= 0.1




        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this
        for parti in particles:
            theta = parti.getTheta()
            # unit vector pointing in the direction of the particle
            heading =  np.array([np.cos(theta), np.sin(theta)])
            # scale with velocity
            deltaXY = heading * velocity

            # do the update
            particle.move_particle(parti, deltaXY[0], deltaXY[1], angular_velocity )
        particle.add_uncertainty(particles, 2.5, 0.125)

        # Fetch next frame
        colour = cam.get_next_frame()

        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        
        if not isinstance(objectIDs, type(None)):

            measurement = {}
            for i in range(len(objectIDs)):
                measurement.setdefault(objectIDs[i], (np.inf, np.inf))
                exist_dist, exist_angle = measurement[objectIDs[i]]
                if dists[i] < exist_dist:
                    measurement[objectIDs[i]] = (dists[i], angles[i])
            # measurement = dict()
            # for i in range(len(objectIDs)):
            #     if objectIDs[i] in measurement:
            #           measurement[objectIDs[i]] += np.array([dists[i], angles[i]])
            #           measurement[objectIDs[i]] /= 2
            #           # DISCLAIMER: if there exists more than 2 observations of the same objID the result is not the mean.
            #     else:
            #         measurement[objectIDs[i]] = np.array([dists[i], angles[i]])
        
            #intersection between measurments and world model
            useful_measurements = set(measurement).intersection(set(landmarkIDs))

            if len(useful_measurements) != 0:
                # Compute particle weights
                weights = np.array(
                    [particle_likelihood(part, measurement) 
                    for part in particles], dtype=float
                    )

                # normalization step
                if sum(weights) != 0:
                    weights /= sum(weights)
                for part, w in zip(particles, weights):
                    part.setWeight(w)

                # Resampling
                # XXX: You do this
                particles = resample_particles(particles, num_particles)

                # Draw detected objects
                cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            particle.add_uncertainty(particles, 10, 0.1)
            for p in particles:
                p.setWeight(1.0/num_particles)

        est_pose = particle.estimate_pose(particles) 
        
    
        i += 1 
        # if i % 100 == 0:
        #     command = do_direct_path(
        #         np.array([est_pose.getX(), est_pose.getY()]), 
        #         est_pose.getTheta(), 
        #         goal
        #         )
        # command.update_command_state()
        distance, theta = polar_diff( 
            np.array([est_pose.getX(), est_pose.getY()]), 
            est_pose.getTheta(), 
            goal
        )

        robot_state.update(particles, distance, theta, set(objectIDs).intersection(landmarks.keys()) if objectIDs is not None else None)

        if robot_state.current_command is not None:
            if hasattr(robot_state.current_command, "velocity"):
                velocity = robot_state.current_command.velocity/10
            else:
                velocity = 0.0
            if hasattr(robot_state.current_command, "rotation_speed"):
                angular_velocity = robot_state.current_command.rotation_speed/10
            else:
                angular_velocity = 0.0

        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)

            # Show frame
            if showPreview:
                cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)


finally:
    # Make sure to clean up even if an exception occurred

    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()

