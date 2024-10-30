import cv2
import numpy as np

from constants import Constants


def jet(x):
    """Colour map for drawing particles. This function determines the colour of
    a particle from its weight."""
    r = (
        (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (4.0 * x - 3.0 / 2.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0)
        + (x >= 7.0 / 8.0) * (-4.0 * x + 9.0 / 2.0)
    )
    g = (
        (x >= 1.0 / 8.0 and x < 3.0 / 8.0) * (4.0 * x - 1.0 / 2.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0) * (-4.0 * x + 7.0 / 2.0)
    )
    b = (
        (x < 1.0 / 8.0) * (4.0 * x + 1.0 / 2.0)
        + (x >= 1.0 / 8.0 and x < 3.0 / 8.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (-4.0 * x + 5.0 / 2.0)
    )

    return (255.0 * r, 255.0 * g, 255.0 * b)


# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

landmark_colors = [
    CRED,
    CGREEN,
    CBLUE,
    CCYAN,
    CYELLOW,
    CMAGENTA,
    CWHITE,
    CBLACK,
]  # Colors used when drawing the landmarks


# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
world = Constants.World
landmarks = world.landmarks
landmarkIDs = world.landmarkIDs
goal = world.goals


class Info:
    def __init__(self):
        if Constants.PID.ENABLE_GUI:
            # Open windows
            if Constants.PID.ENABLE_PREVIEW:
                self.WIN_RF1 = "Robot view"
                cv2.namedWindow(self.WIN_RF1)
                cv2.moveWindow(self.WIN_RF1, 50, 50)

            self.WIN_World = "World view"
            cv2.namedWindow(self.WIN_World)
            cv2.moveWindow(self.WIN_World, 500, 50)

        # Allocate space for world map
        self.world = np.zeros((600, 500, 3), dtype=np.uint8)

    def draw_world(self, particles, est_pose=None):
        """Visualization.
        This functions draws robots position in the world coordinate system."""

        # Fix the origin of the coordinate system
        offsetX = 50
        offsetY = 100

        # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
        ymax = self.world.shape[0]

        self.world[:] = CWHITE  # Clear background to white

        # Find largest weight
        max_weight = 0
        for particle in particles.particles:
            max_weight = max(max_weight, particle.getWeight())

        # Draw particles
        for particle in particles.particles:
            x = int(particle.getX() + offsetX)
            y = ymax - (int(particle.getY() + offsetY))
            colour = jet(particle.getWeight() / max_weight)
            cv2.circle(self.world, (x, y), 2, colour, 2)
            b = (
                int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
                ymax - (int(particle.getY() + 15.0 * np.sin(particle.getTheta())) + offsetY),
            )
            cv2.line(self.world, (x, y), b, colour, 2)

        # Draw landmarks
        for i in range(len(landmarkIDs)):
            ID = landmarkIDs[i]
            lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
            cv2.circle(self.world, lm, 5, landmark_colors[i], 2)

        if est_pose is not None:
            # Draw estimated robot pose
            a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
            b = (
                int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
                ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
            )
            cv2.circle(self.world, a, 5, CMAGENTA, 2)
            cv2.line(self.world, a, b, CMAGENTA, 2)

    def show_frame(self, colour):
        # Show frame
        if Constants.PID.ENABLE_GUI:
            if Constants.PID.ENABLE_PREVIEW:
                cv2.imshow(self.WIN_RF1, colour)

            # Show world
            cv2.imshow(self.WIN_World, self.world)
