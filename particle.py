import sys
from copy import copy

import numpy as np
from numpy import random

import random_numbers as rn
from math_utils import normal, polar_diff


class Particle(object):
    """Data structure for storing particle information (state and weight)"""

    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=0.0, min_maxes=None):
        self.x = x
        self.y = y
        self.theta = np.mod(theta, 2.0 * np.pi)
        self.weight = weight
        self.min_maxes = min_maxes

    def __copy__(self):
        return type(self)(self.getX(), self.getY(), self.getTheta(), self.getWeight())

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getPos(self):
        return np.array((self.x, self.y))

    def getTheta(self):
        return self.theta

    def getWeight(self):
        return self.weight

    def getMinMaxes(self):
        return self.min_maxes

    def checkLowVarianceMinMaxes(self):
        if self.min_maxes is None:
            return False

        (min_x, max_x), (min_y, max_y) = self.min_maxes
        return max_x - min_x < 25 and max_y - min_y < 25

    def setX(self, val):
        self.x = val

    def setY(self, val):
        self.y = val

    def setTheta(self, val):
        self.theta = np.mod(val, 2.0 * np.pi)

    def setWeight(self, val):
        self.weight = val

    def move_particle(self, delta_x, delta_y, delta_theta):
        """Move the particle by (delta_x, delta_y, delta_theta)"""
        self.setX(self.x + delta_x)
        self.setY(self.y + delta_y)
        self.setTheta(self.theta + delta_theta)

    def particle_likelihood(self, measurements, landmarks: dict[int, tuple[float, float]]):
        likelihood = 1

        part_pos = np.array([self.getX(), self.getY()])

        for l_id, (m_dist, m_ang) in measurements.items():
            # If observed landmark is not known, ignore it
            if l_id not in landmarks.keys():
                # print(f"alert: {l_id} seen and ignored")
                continue
            else:
                # print(f" {l_id} {m_dist} {np.rad2deg(m_ang)}")
                pass

            land_pos = np.array(landmarks[l_id])

            dist, theta = polar_diff(part_pos, self.getTheta(), land_pos)
            likelihood *= normal(theta - m_ang, 0, 0.25) + sys.float_info.min * 2
            likelihood *= normal(dist - m_dist, 0, 10) + sys.float_info.min * 2

        return likelihood


class ParticlesWrapper:
    def __init__(self, num_particles, landmarks) -> None:
        self.particles: list[Particle] = self.initialize_particles(num_particles)
        self.num_particles: int = num_particles
        self.rng = random.default_rng()
        self.landmarks = landmarks

    @staticmethod
    def initialize_particles(num_particles) -> list[Particle]:
        particles = []
        for _ in range(num_particles):
            # Random starting points.
            p = Particle(
                550.0 * np.random.ranf() - 50.0,
                600.0 * np.random.ranf() - 100.0,
                np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
                1.0 / num_particles,
            )
            particles.append(p)

        return particles

    def estimate_pose(self):
        """Estimate the pose from particles by computing the average position and orientation over all particles.
        This is not done using the particle weights, but just the sample distribution."""
        x_sum = 0.0
        y_sum = 0.0
        cos_sum = 0.0
        sin_sum = 0.0

        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf

        for particle in self.particles:
            w = particle.getWeight()

            x = particle.getX()
            y = particle.getY()

            x_sum += x
            y_sum += y

            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

            cos_sum += np.cos(particle.getTheta())
            sin_sum += np.sin(particle.getTheta())

        flen = len(self.particles)
        if flen != 0:
            x = x_sum / flen
            y = y_sum / flen
            theta = np.arctan2(sin_sum / flen, cos_sum / flen)
        else:
            x = x_sum
            y = y_sum
            theta = 0.0

        return Particle(x, y, theta, min_maxes=((min_x, max_x), (min_y, max_y)))

    def add_uncertainty(self, sigma, sigma_theta):
        """Add some noise to each particle in the list. Sigma and sigma_theta is the noise
        variances for position and angle noise."""
        for particle in self.particles:
            particle.x += rn.randn(0.0, sigma)
            particle.y += rn.randn(0.0, sigma)
            particle.theta = np.mod(particle.theta + rn.randn(0.0, sigma_theta), 2.0 * np.pi)

    def move_particles(self, distance: float, angle: float):
        for parti in self.particles:
            theta = parti.getTheta()
            # unit vector pointing in the direction of the particle
            heading = np.array([np.cos(theta), np.sin(theta)])
            # scale with velocity
            deltaXY = heading * distance

            # do the update
            parti.move_particle(deltaXY[0], deltaXY[1], angle)
        # print("moved particles", distance, angle)

    def resample_particles(self):
        pmf = np.zeros(self.num_particles, dtype=np.float64)
        if not self.particles:
            return

        for i, p in enumerate(self.particles):
            pmf[i] = p.getWeight()
            if np.isnan(pmf[i]):
                print(pmf[i])

        # choice as indexes:
        choices = self.rng.choice(self.num_particles, size=self.num_particles, p=pmf)
        resampled_particles = [copy(self.particles[choice]) for choice in choices]
        self.particles = resampled_particles[:self.num_particles // 2] + self.initialize_particles(
            self.num_particles // 2
        )

    def set_weights(self, weights):
        for p, w in zip(self.particles, weights):
            p.setWeight(w)

    def set_uniform_weights(self):
        self.set_weights([1.0 / self.num_particles] * self.num_particles)

    def particle_likelihoods(self, measurements):
        weights = np.array(
            [
                particle.particle_likelihood(measurements, self.landmarks)
                for particle in self.particles
            ],
            dtype=float,
        )
        if sum(weights) != 0:
            weights /= sum(weights)
        self.set_weights(weights)
        return weights

    def update(self, measurements):
        self.particle_likelihoods(measurements)
        self.resample_particles()
