import numpy as np

## parameters
class Langevin3D():
    # atomic mass unit
    amu = 1.660538921e-27 #kg 

    def __init__(self, T=None, seed=123, dt=1e-15):
        self.dimension = 3
        # radius at equilibrium
        self.r_0 = 1.275 #A #1.275e-10 m
        # D Morse potential
        self.D = 4.6141 #(in A) #4.6141 * 1.60218e-19 J
        # alpha Morse potential
        self.alpha = 1.81 #A^-1 1.81e10 m^-1

        # masses
        self.m_Cl = 35.43 * self.amu #kg
        self.m_H = 1.00784 * self.amu #kg
        # reduced mass
        self.mu = (self.m_Cl * self.m_H) / (self.m_Cl + self.m_H) #kg

        self.gamma = 6e11 #s^-1
        self.k_B = 1.380649e-23 / 1.60218e-19 #J/K
        self.h_bar = 1.05457182e-34 / 1.60218e-19

        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.T = T

    def potential_Morse(self, r):
        """Morse potential between two atoms at distance r"""
        return self.D * (np.exp(-self.alpha * (r - self.r_0)) - 1)**2
    
    def force_Morse(self, r):
        """Gradient of the Morse potential"""
        return 2 * self.alpha * self.D * (np.exp(-self.alpha * (r - self.r_0)) - np.exp(-2 * self.alpha * (r - self.r_0)))
    
    def force_Random(self,  mass, n_draws=1):
        """Random force according to the fluctuation-dissipation theorem"""
        r_amp = np.sqrt(2 * self.k_B * self.T * self.gamma * mass / self.dt)
        return self.rng.normal(0, 1, size=n_draws) * r_amp
    
    def force_viscosity(self, v, mass):
        """Viscous force according to Langevin equation"""
        return - self.gamma * mass * v

    def compute_force(self, r_rel, v, mass):
        fm = -self.force_Morse(r_rel) * np.ones(self.dimension)
        if self.T is None or mass is None:
            return fm
        fv = self.force_viscosity(v, mass) * np.ones(self.dimension)
        fr = self.force_Random(n_draws=self.dimension, mass=mass)
        return fm + fr + fv

    def verlet(self, pos, speed, mass, r_rel):
        """Langevin dynamics update for a particle"""
        ri_p1 = 2*pos[-1] - pos[-2] + self.compute_force(r_rel=r_rel, v=speed[-1], mass=mass) * self.dt**2 / mass
        vi_p1 = (ri_p1 - pos[-2]) / (2 * self.dt)
        return ri_p1, vi_p1

    def langevin(self, pos, speed, mass, r_rel):
        force = self.compute_force(r_rel=r_rel, v=speed[-1], mass=mass)
        ri_p1 = 2*pos[-1] - pos[-2] + force * self.dt**2 / mass
        vi_p1 = (3 * ri_p1 - 4 * pos[-1] + pos[-2]) / (2 * self.dt)
        return ri_p1, vi_p1, force

    def run(self, n_steps, r_init_relative, v_init=None, T_init=None, mode='langevin'):
        """Run the Langevin dynamics simulation for n_steps"""
        assert n_steps >= 2, "n_steps must be at least 2"
        if mode == 'verlet':
            alg = self.verlet
        elif mode == 'langevin':
            alg = self.langevin
        else:
            raise ValueError("mode must be 'verlet' or 'langevin'")
        time = np.arange(n_steps) * self.dt

        r_Cl_0 = np.array([0, 0, 0])
        r_H_0 = np.array([r_init_relative, 0, 0])

        v_Cl_0 = np.array([0, 0, 0])
        v_H_0 = np.array([0, 0, 0])
        r_rel_0 = np.sqrt(np.sum((r_H_0 - r_Cl_0)**2))

        force_Cl_0 = self.compute_force(r_rel=r_rel_0, v=v_Cl_0, mass=self.m_Cl)
        r_Cl_1 = r_Cl_0 + v_Cl_0 * self.dt + 0.5 * force_Cl_0 * self.dt**2 / (2*self.m_Cl)
        r_Cl = [r_Cl_0, r_Cl_1]
        v_Cl_1 = v_Cl_0 + force_Cl_0 * self.dt / self.m_Cl
        v_Cl = [v_Cl_0, v_Cl_1]

        force_H_0  = self.compute_force(r_rel=r_rel_0, v=v_H_0, mass=self.m_H)
        r_H_1 = r_H_0 + v_H_0 * self.dt + 0.5 * force_H_0 * self.dt**2 / (2*self.m_H)
        r_H = [r_H_0, r_H_1]
        v_H_1 = v_H_0 + force_H_0 * self.dt / self.m_H
        v_H = [v_H_0, v_H_1]

        r_rel_1 = np.sqrt(np.sum((r_H_1 - r_Cl_1)**2))
        force_H_1 = self.compute_force(r_rel=r_rel_1, v=v_H_1, mass=self.m_H)
        force_Cl_1 = self.compute_force(r_rel=r_rel_1, v=v_Cl_1, mass=self.m_Cl)
        
        r_rel = [r_rel_0, r_rel_1]
        force_H = [force_H_0, force_H_1]
        force_Cl = [force_Cl_0, force_Cl_1]

        for _ in range(2, len(time)):
            r_rel_this = r_rel[-1]
            r_H_new, v_H_new, force_H_new = alg(pos=r_H, speed=v_H, mass=self.m_H, r_rel=r_rel_this)
            r_Cl_new, v_Cl_new, force_Cl_new = alg(pos=r_Cl, speed=v_Cl, mass=self.m_Cl, r_rel=r_rel_this)

            r_H.append(r_H_new)
            r_Cl.append(r_Cl_new)
            v_H.append(v_H_new)
            v_Cl.append(v_Cl_new)
            force_H.append(force_H_new)
            force_Cl.append(force_Cl_new)
            
            r_rel.append(np.sqrt(np.sum((r_H_new - r_Cl_new)**2)))

        data = {
            "time": np.array(time),
            "r_Cl": np.array(r_Cl),
            "v_Cl": np.array(v_Cl),
            "force_Cl": np.array(force_Cl),
            "r_H": np.array(r_H),
            "v_H": np.array(v_H),
            "force_H": np.array(force_H),
            "r_rel": np.array(r_rel)
        }
        ## compute by-products
        data['kinetic_energy_H'] = 0.5 * self.m_H * np.sum(data['v_H']**2, axis=1)
        data['kinetic_energy_Cl'] = 0.5 * self.m_Cl * np.sum(data['v_Cl']**2, axis=1)
        data['potential_energy'] = self.potential_Morse(data['r_rel'])
 
        return data

    def rotational_energy(self, traj, speed):
        rot_energy = []
        for t,v in zip(traj,speed): 
            p = self.mu * v
            vectorial = np.cross(t,p)
            rot_energy.append((vectorial ** 2)/((t ** 2)*2*self.mu))
        return rot_energy

    def vibrational_energy(self, traj, speed):
        vib_energy = -((self.h_bar ** 2)/2*self.mu*traj) # to complete
        return vib_energy
    
