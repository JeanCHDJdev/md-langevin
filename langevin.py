import numpy as np

## parameters
class Langevin3D():
    # atomic mass unit
    amu = 1.660538921e-27 #kg 

    def __init__(self, T=None, seed=123, dt=1e-15):
        # radius at equilibrium
        self.r_0 = 1.275 #A #1.275e-10 #m
        # D Morse potential
        self.D = 46.141 #(in A) # 4.6141 * 1.60218e-19 #J
        # alpha Morse potential
        self.alpha = 1.81 #A^-1 1.81e10 #m^-1

        # masses
        self.m_Cl = 35.43 * self.amu #kg
        self.m_H = 1.00784 * self.amu #kg
        # reduced mass
        self.mu = (self.m_Cl * self.m_H) / (self.m_Cl + self.m_H) #kg

        self.gamma = 1e15 #s^-1
        self.k_B = 1.380649e-3 #(in A) #1.380649e-23 #J/K
        self.h_bar = 1.05457182e-14 # (in A) # 1.05457182e-34 Js

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

    def compute_force(self, r_rel, mass):
        if self.T is None or mass is None:
            return -self.force_Morse(r_rel)
        f_rand = self.force_Random(n_draws=3, mass=mass)
        return -self.force_Morse(r_rel) + f_rand

    def verlet(self, pos, mass, r_rel):
        """Langevin dynamics update for a particle"""
        ri_p1 = 2*pos[-1] - pos[-2] + self.compute_force(r_rel=r_rel, mass=None) * self.dt**2 / mass
        vi = (ri_p1 - pos[-2]) / (2 * self.dt)
        return ri_p1, vi
    
    def langevin(self, pos, mass, r_rel):
        ri_p1 = 2*pos[-1] - pos[-2] + self.compute_force(r_rel=r_rel, mass=mass) * self.dt**2 / mass
        vi = (3 * pos[-1] - 4 * pos[-2] + pos[-3]) / (2 * self.dt)
        return ri_p1, vi

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

        r_Cl_init =  np.array([0, 0, 0])
        r_H_init =  np.array([r_init_relative, 0, 0])

        v_Cl_init =  np.array([0, 0, 0])
        v_H_init =  np.array([0, 0, 0])

        r_Cl = [r_Cl_init, r_Cl_init + v_Cl_init * self.dt, r_Cl_init + 2 * v_Cl_init * self.dt]
        r_H = [r_H_init, r_H_init + v_H_init * self.dt, r_H_init + 2 * v_H_init * self.dt]
        v_Cl = [v_Cl_init, v_Cl_init, v_Cl_init]
        v_H = [v_H_init, v_H_init, v_H_init]

        for _ in range(3, len(time)):
            r_rel = np.sqrt(np.sum((r_H[-1] - r_Cl[-1])**2))
            r_H_new, v_H_new = alg(r_H, mass=self.m_H, r_rel=r_rel)
            r_Cl_new, v_Cl_new = alg(r_Cl, mass=self.m_Cl, r_rel=r_rel)

            r_H.append(r_H_new)
            r_Cl.append(r_Cl_new)
            v_H.append(v_H_new)
            v_Cl.append(v_Cl_new)

        return np.array(time), np.array(r_Cl), np.array(r_H), np.array(v_Cl), np.array(v_H)

    def distribute_v_to_3D(self, v):
        """Distribute a scalar velocity to 3D components"""
        theta = self.rng.uniform(0, np.pi)
        phi = self.rng.uniform(0, 2 * np.pi)

        vx = v * np.sin(theta) * np.cos(phi)
        vy = v * np.sin(theta) * np.sin(phi)
        vz = v * np.cos(theta)

        return [vx, vy, vz]
    
    def v_to_pos(self, speed):
        """Convert 1D pos and speed to 3D pos and speed"""
        assert len(speed) >= 1, "Speed list must contain at least one element"
        traj_3D = [[0, 0, 0]]
        speed_3D = [self.distribute_v_to_3D(speed[0])]
        for v in speed:
            vx, vy, vz = self.distribute_v_to_3D(v)
            # integrate position components
            rx = traj_3D[-1][0] + self.dt * vx
            ry = traj_3D[-1][1] + self.dt * vy
            rz = traj_3D[-1][2] + self.dt * vz
            traj_3D.append((rx, ry, rz))
            speed_3D.append((vx, vy, vz))

        return np.array(traj_3D), np.array(speed_3D)
    
    def kinetic_energy(self,speed):
        return 0.5 * self.mu * speed ** 2

    def potential_energy(self, traj):
        return self.potential_Morse(traj)
    
    def total_energy(self, traj, speed):
        return self.kinetic_energy(speed) + self.potential_energy(traj)

    def temperature(self, speed):
        return (self.mu * np.mean(speed ** 2))/(3 * self.k_B)

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
    
