import numpy as np

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