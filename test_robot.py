import numpy as np

from constants import Constants


class TestRobot(object):
    """Defines the Arlo robot API

    DISCLAIMER: This code does not contain error checking - it is the responsibility
    of the caller to ensure proper parameters and not to send commands to the
    Arduino too frequently (give it time to process the command by adding a short sleep wait
    statement). Failure to do some may lead to strange robot behaviour.

    In case you experience trouble - consider using only commands that do not use the wheel
    encoders.
    """

    def __new__(cls):  # Make singleton
        if not hasattr(cls, "instance"):
            cls.instance = super(TestRobot, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        """Set pos and angle of the robot for testing purposes."""
        self.pos = (0, 0)
        self.angle = 0

    def _power_checker(self, power):
        """Checks if a power value is in the set {0, [30;127]}.
        This is an internal utility function."""
        return (power == 0) or (power >= 30 and power <= 127)

    def go_diff(self, powerLeft, powerRight, dirLeft, dirRight, time=0):

        if [dirLeft, dirRight] == [1, 0]:
            self.angle += Constants.Robot.ROTATIONAL_SPEED * time
        elif [dirLeft, dirRight] == [0, 1]:
            self.angle -= Constants.Robot.ROTATIONAL_SPEED * time
        elif {dirLeft, dirRight} == {1}:
            velocity = Constants.Robot.FORWARD_SPEED
            self.pos = (
                self.pos[0] + velocity * time * np.cos(self.angle),
                self.pos[1] + velocity * time * np.sin(self.angle),
            )
        elif {dirLeft, dirRight} == {0}:
            velocity = -Constants.Robot.FORWARD_SPEED
            self.pos = (
                self.pos[0] + velocity * time * np.cos(self.angle),
                self.pos[1] + velocity * time * np.sin(self.angle),
            )

    def stop(self):
        """Send a stop command to stop motors. Sets the motor power on both wheels to zero.

        This does NOT use wheel encoders."""
        pass

    def read_sensor(self, sensorid):
        pass

    def read_front_ping_sensor(self):
        """Read the front sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(0)

    def read_back_ping_sensor(self):
        """Read the back sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(1)

    def read_left_ping_sensor(self):
        """Read the left sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(2)

    def read_right_ping_sensor(self):
        """Read the right sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(3)

    def read_left_wheel_encoder(self):
        """Reads the left wheel encoder counts since last reset_encoder_counts command.
        The encoder has 144 counts for one complete wheel revolution."""
        pass

    def read_right_wheel_encoder(self):
        """Reads the right wheel encoder counts since last clear reset_encoder_counts command.
        The encoder has 144 counts for one complete wheel revolution."""
        pass

    def reset_encoder_counts(self):
        """Reset the wheel encoder counts."""
        pass
