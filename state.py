from enum import Enum
import time

import cv2
import camera
import command
import selflocalize
from particle import Particle, ParticlesWrapper
from constants import Constants

class RobotState(Enum):
    lost = 0
    moving = 1
    checking = 2




class State:

    def __init__(self) -> None:
        self.on_arlo = Constants.World.running_on_arlo
        self.show_preview = Constants.PID.ENABLE_PREVIEW
        self.state = RobotState.lost
        self.particles = ParticlesWrapper(Constants.World.num_particles, Constants.World.landmarks)
        self._cam: camera.Camera
        self.WIN_RF1 = "Robot view"
        self.arlo = command.ControlWrapper(self.on_arlo)

        if self.on_arlo:
            self._cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        else:
            self._cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)

        if self.show_preview:
            cv2.namedWindow(self.WIN_RF1)
            cv2.moveWindow(self.WIN_RF1, 50, 50)

        self._lost = self.Lost(self)
        self._moving = self.Moving(self)
        self._checking = self.Checking(self)
        self.current_state = self._lost


    @property
    def cam(self):
        if self._cam is None:
            if self.on_arlo:
                self._cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
            else:
                self._cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        return self._cam

    def next_frame(self):
        colour = self._cam.get_next_frame()
        if self.show_preview:
            cv2.imshow(self.WIN_RF1, colour)
        return colour

    class Lost:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
            self.arlo = outer_instance.arlo
            # Fetch next frame
            colour = outer_instance.next_frame()
            self.start_time = time.time()  # TODO: delete this

            # Detect objects
            objectIDs, dists, angles = self.cam.detect_aruco_objects(colour)

            def gen_command():
                while True:
                    yield command.Wait(self.arlo, 2)
                    yield command.Rotate(self.arlo, 0.5)
            self.queue = iter(gen_command())
            self.current_command = next(self.queue)
                    
            # Reset filter, plan
            # TODO:

            # Movement command
            self.current_command.run_command()
            self.seen_landmarks = set()
            

        def update(self):
            
            if not len(self.seen_landmarks) >= 2:
                self.current_command.run_command()
            if self.current_command.finished:
                self.current_command = next(self.queue)
                self.current_command.run_command()
            else:
                self.current_command.run_command()
            

    class Moving:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam
        
        def update(self):
            pass

    class Checking:
        def __init__(self, outer_instance: "State") -> None:
            self.cam: camera.Camera = outer_instance.cam

        def update(self):
            pass

    def set_state(self, state: RobotState):
        self.state = state
        self.current_state = self.lost or self.moving or self.checking

    @property
    def lost(self):
        return self._lost if self.state == RobotState.lost else None

    @property
    def moving(self):
        return self._moving if self.state == RobotState.moving else None

    @property
    def checking(self):
        return self._checking if self.state == RobotState.checking else None

    def update(self):
        self.current_state.update()

state = State()
try:
    while True:
        state.update()
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print("Exiting program")
    exit(0)