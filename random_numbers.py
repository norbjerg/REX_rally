import numpy as np


def randn(mu, sigma):
    """Normal random number generator
    mean mu, standard deviation sigma"""
    return sigma * np.random.randn() + mu
    
    
def rand_von_mises(mu, kappa):
    """Generate random samples from the Von Mises distribution"""
    if kappa < 1e-6:
        # kappa is small: sample uniformly on circle
        theta = 2.0 * np.pi * np.random.ranf() - np.pi
    else:
        a = 1.0 + np.sqrt(1.0 + 4.0 * kappa * kappa)
        b = (a - np.sqrt(2.0 * a)) / (2.0 * kappa)
        r = (1.0 + b*b) / (2.0 * b)

        not_done = True
        while not_done:
            u1 = np.random.ranf()
            u2 = np.random.ranf()

            z = np.cos(np.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)

            not_done = (u2 >= c * (2.0 - c)) and (np.log(c) - np.log(u2) + 1.0 - c < 0.0)
            
            u3 = np.random.ranf()
            theta = mu + np.sign(u3 - 0.5) * np.arccos(f)

    return theta


if __name__ == '__main__':
    # Tests
    
    print("Gaussian distribution:")
    r = np.zeros(1000)
    for i in range(1000):
        r[i] = randn(1.0, 2.0)
        
    print("True mean 1.0 == Estimated mean ", np.mean(r))
    print("True std 2.0 == Estimated std ", np.std(r))
    
    print("Von Mises distribution:")
    
    t = np.zeros(1000)
    for i in range(1000):
        t[i] = rand_von_mises(1.0, 8)
        
    print("True mean 1.0 == Estimated mean ", np.mean(t))
