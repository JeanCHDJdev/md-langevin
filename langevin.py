import numpy as np

## parameters
class Langevin3D():
    # atomic mass unit
    amu = 1.660538921e-27 #kg 

    def __init__(self):
        # radius at equilibrium
        self.r_0 = 1.275e-10 #m
        # D Morse potential
        self.D = 4.6141 #eV
        # alpha Morse potential
        self.alpha = 1.81e10 #m^-1

        # masses
        self.m_Cl = 35.43 * self.amu #kg
        self.m_H = 1.00784 * self.amu #kg
        # reduced mass
        self.mu = (self.m_Cl * self.m_H) / (self.m_Cl + self.m_H) #kg

        self.gamma = 1e14 #s^-1
        self.k_B = 1.380649e-23 #J/K
        self.T = 300 #K

        self.dt = 1 / (self.gamma * 100) #s

    def run(self, n_steps, r_init, v_init):
        """Run the Langevin dynamics simulation for n_steps"""
        r = r_init
        v = v_init
        trajectory = np.zeros(n_steps)
        time = [self.dt * i for i in range(n_steps)]
        speed = np.zeros(n_steps)
        for i in range(n_steps):
            r, v = self.langevin(r, v)
            trajectory[i] = r
            speed[i] = v
        return time, trajectory, speed

    def potential_Morse(self, r):
        """Morse potential between two atoms at distance r"""
        return self.D * (np.exp(-self.alpha * (r - self.r_0)) - 1)**2
    
    def force_Morse(self, r):
        """Gradient of the Morse potential"""
        return 2 * self.alpha * self.D * (np.exp(-self.alpha * (r - self.r_0)) - np.exp(-2 * self.alpha * (r - self.r_0)))
    
    def force_Random(self, m):
        """Random force according to the fluctuation-dissipation theorem"""
        r_amp = np.sqrt(2 * self.k_B * self.T * self.gamma * m / self.dt)
        return np.random.normal(0, 1) * r_amp
    
    def compute_force(self, r):
        return -self.force_Morse(r) + self.force_Random(self.mu)

    def _compute_ri(self, ri, ri_m1):
        """Returns ri+1 with ri, ri-1"""
        ri_p1 = 2*ri - ri_m1 + self.compute_force(ri) * self.dt**2 / self.mu
        return ri_p1

    def _compute_vi(self, r, v):
        """Velocity update for a particle"""
        F = -self.force_Morse(r)
        
        return r, v
    

    def langevin(self, r, v):
        """Langevin dynamics update for a particle"""
        F = -self.force_Morse(r)
        
        return r, v
    