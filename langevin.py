import numpy as np

## parameters
class Langevin3D():
    # atomic mass unit
    amu = 1.660538921e-27 #kg 

    def __init__(self, T=None, seed=123):
        # radius at equilibrium
        self.r_0 = 1.275 #A #1.275e-10 #m
        # D Morse potential
        self.D = 46.141 # (in A) # 4.6141 * 1.60218e-19 #J
        # alpha Morse potential
        self.alpha = 1.81 #A^-1 1.81e10 #m^-1

        # masses
        self.m_Cl = 35.43 * self.amu #kg
        self.m_H = 1.00784 * self.amu #kg
        # reduced mass
        self.mu = (self.m_Cl * self.m_H) / (self.m_Cl + self.m_H) #kg

        self.gamma = 1e14 #s^-1
        self.k_B = 1.380649e-3 #(in A) #1.380649e-23 #J/K 
 
        self.dt = 1 / (self.gamma * 100) #s

        self.rng = np.random.default_rng(seed)
        
        self.T = T

    def potential_Morse(self, r):
        """Morse potential between two atoms at distance r"""
        return self.D * (np.exp(-self.alpha * (r - self.r_0)) - 1)**2
    
    def force_Morse(self, r):
        """Gradient of the Morse potential"""
        return 2 * self.alpha * self.D * (np.exp(-self.alpha * (r - self.r_0)) - np.exp(-2 * self.alpha * (r - self.r_0)))
    
    def force_Random(self):
        """Random force according to the fluctuation-dissipation theorem"""
        r_amp = np.sqrt(2 * self.k_B * self.T * self.gamma * self.mu / self.dt)
        return self.rng.normal(0, 1) * r_amp
    
    def compute_force(self, r):
        return -self.force_Morse(r) + self.force_Random()

    def _compute_ri(self, ri, ri_m1):
        """Returns ri+1 with ri, ri-1"""
        ri_p1 = 2*ri - ri_m1 + self.compute_force(ri) * self.dt**2 / self.mu
        return ri_p1

    def _compute_vi_verlet(self, ri_p1, ri_m1):
        """Velocity update for a particle"""
        return (ri_p1 - ri_m1) / (2 * self.dt)
    
    def verlet(self, trajectory):
        """Langevin dynamics update for a particle"""
        ri_p1 = self._compute_ri(trajectory[-1], trajectory[-2])
        vi = self._compute_vi_verlet(ri_p1, trajectory[-2])
        return ri_p1, vi
    
    def langevin(self, trajectory):
        ri_p1 = self._compute_ri(trajectory[-1], trajectory[-2])
        vi = 3 * trajectory[-1] - 4 * trajectory[-2] + trajectory[-3] / (2 * self.dt)
        return ri_p1, vi

    def run(self, n_steps, r_init, v_init, mode='langevin'):
        """Run the Langevin dynamics simulation for n_steps"""
        assert n_steps >= 2, "n_steps must be at least 2"
        if mode == 'verlet':
            alg = self.verlet
        elif mode == 'langevin':
            alg = self.langevin
        else:
            raise ValueError("mode must be 'verlet' or 'langevin'")
        time = np.arange(n_steps) * self.dt

        r = r_init
        v = v_init

        trajectory = [r_init, r_init + v_init * self.dt, r_init + v_init * 2 * self.dt]
        speed = [v_init, v_init, v_init]

        for _ in range(3, n_steps):
            r, v = alg(trajectory)
            trajectory.append(r)
            speed.append(v)

        return time, trajectory, speed
    