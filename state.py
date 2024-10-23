from enum import Enum
import selflocalize
from particle import Particle, ParticlesWrapper
from constants import Constants

class RobotState(Enum):
    lost = 0
    moving = 1
    checking = 2




class State:

    def __init__(self) -> None:
        self.state = RobotState.lost
        self._lost = self.Lost()
        self._moving = self.Moving()
        self._checking = self.Checking()
        self.current_state = self._lost
        self.particles = ParticlesWrapper(Constants.World.num_particles, Constants.World.landmarks)

    class Lost:
        def __init__(self) -> None:
            pass

        def update(self):
            pass

    class Moving:
        def __init__(self) -> None:
            pass

    class Checking:
        def __init__(self) -> None:
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
        pass
