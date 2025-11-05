import numpy as np
import pandas as pd
import datetime

from pathlib import Path

class Langevin3D():

    def __init__(self, T=None, gamma=5e11, potential='morse', seed=123, dt=1e-15):
        self.dimension = 3
        # radius at equilibrium
        self.r_0 = 1.275 #A #1.275e-10 m
        # D morse potential
        self.D = 4.6141 #(in A) #4.6141 * 1.60218e-19 J
        # alpha morse potential
        self.alpha = 1.81 #A^-1 1.81e10 m^-1

        self.k = self.D * self.alpha**2 # harmonic constant (taylor series expansion)

        # masses
        # atomic mass unit
        amu = 1.660538921e-27 #kg 
        self.m_Cl = 35.43 * amu #kg
        self.m_H = 1.00784 * amu #kg
        self.mu = (self.m_Cl * self.m_H) / (self.m_Cl + self.m_H) #kg

        ev = 1.60218e-19
        self.k_B = 1.380649e-23 / ev #J/K
        self.h_bar = 1.05457182e-34 / ev

        self.gamma = gamma #s^-1
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.T = T

        self.potential_types = ['morse', 'harmonic']
        if potential not in self.potential_types:
            raise ValueError(f"potential must be one of {self.potential_types}")
        self.potential = potential

    def potential_morse(self, r):
        """morse potential between two atoms at distance r"""
        return self.D * (np.exp(-self.alpha * (r - self.r_0)) - 1)**2
    
    def force_morse(self, r):
        """Gradient of the morse potential"""
        force = np.zeros(self.dimension)
        r_rel = np.sqrt(np.sum(r**2))
        for i in range(self.dimension):
            force[i] = -2 * self.alpha * self.D * (np.exp(-self.alpha * (r_rel - self.r_0)) - np.exp(-2 * self.alpha * (r_rel - self.r_0))) * r[i]/r_rel
        return force

    def potential_harmonic(self, r):
        """Harmonic potential between two atoms at distance r"""
        return self.k * (r - self.r_0)**2
    
    def force_harmonic(self, r):
        """Gradient of the harmonic potential"""
        force = -2 * self.k * (r - self.r_0)
        return force

    def force_random(self, mass, n_draws=1):
        """Random force according to the fluctuation-dissipation theorem"""
        r_amp = np.sqrt(2 * self.k_B * self.T * self.gamma * mass / self.dt)
        return self.rng.normal(0, 1, size=n_draws) * r_amp
    
    def force_viscosity(self, v, mass):
        """Viscous force according to Langevin equation"""
        return - self.gamma * mass * v

    def compute_force(self, r_rel, speed, mass):
        if self.potential == 'harmonic':
            fp = self.force_harmonic(r_rel)
        else:
            fp = self.force_morse(r_rel)
        assert fp.shape == (self.dimension,), f"force_morse should return a vector of dimension {self.dimension}"
        if self.T is None or mass is None:
            return fp
        fv = self.force_viscosity(speed, mass)
        fr = self.force_random(n_draws=self.dimension, mass=mass)
        return fp + fr + fv

    def verlet(self, pos, speed, mass, r_rel):
        """Langevin dynamics update for a particle"""
        force = self.compute_force(r_rel=r_rel[-1], speed=speed[-1], mass=mass)
        ri_p1 = 2*pos[-1] - pos[-2] + force * self.dt**2 / mass
        vi_p1 = (ri_p1 - pos[-2]) / (2 * self.dt)
        return ri_p1, vi_p1, force

    def langevin(self, pos, speed, mass, r_rel):
        force = self.compute_force(r_rel=r_rel[-1], speed=speed[-1], mass=mass)
        ri_p1 = 2*pos[-1] - pos[-2] + force * self.dt**2 / mass
        vi_p1 = (3 * ri_p1 - 4 * pos[-1] + pos[-2]) / (2 * self.dt)
        return ri_p1, vi_p1, force

    def run(
            self, 
            n_steps, 
            r_init, 
            v_init=None, 
            T_init=None, 
            mode='langevin', 
            temp_window=100, 
            filename=None,
            do_byproducts=True
            ):
        """Run the Langevin dynamics simulation for n_steps"""
        assert n_steps >= 2, "n_steps must be at least 2"
        if mode == 'verlet':
            alg = self.verlet
        elif mode == 'langevin':
            alg = self.langevin
        else:
            raise ValueError("mode must be 'verlet' or 'langevin'")
        
        if filename is not None:
            if not Path(filename).parent.exists():
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
        ## save characteristics of the run to json file
        run_info = {
            "n_steps": n_steps,
            "r_init": r_init,
            "v_init": v_init,
            "T_init": T_init,
            "constants": { 
                "temp_window": temp_window,
                "steps" : n_steps,
                "dt": self.dt,
                "gamma": self.gamma,
                "potential": self.potential,
                "m_Cl": self.m_Cl,
                "m_H": self.m_H,
                "seed" : self.seed,
                "k_B": self.k_B,
                "h_bar": self.h_bar,
                "r_0": self.r_0,
                "D": self.D,
                "alpha": self.alpha,
                "k": self.k,
                "T": self.T,
                "mode": mode
            },
            "run_time": str(datetime.datetime.now()),
            "filename": str(filename)
        }
        if filename is not None:
            pd.Series(run_info).to_json(Path(filename).with_suffix('.json'))
            ## save the actual data to a npz file
            filename = Path(filename).with_suffix('.npz')
        
        time = np.arange(n_steps) * self.dt

        r_Cl_0 = np.array([0, 0, 0])
        r_H_0 = np.array([r_init, 0, 0])

        v_Cl_0 = np.array([0, 0, 0])
        v_H_0 = np.array([0, 0, 0])
        r_rel_0 = r_H_0 - r_Cl_0

        force_Cl_0 = self.compute_force(r_rel=-r_rel_0, speed=v_Cl_0, mass=self.m_Cl)
        r_Cl_1 = r_Cl_0 + v_Cl_0 * self.dt + 0.5 * force_Cl_0 * self.dt**2 / (self.m_Cl)
        r_Cl = [r_Cl_0, r_Cl_1]
        v_Cl_1 = v_Cl_0 + force_Cl_0 * self.dt / self.m_Cl
        v_Cl = [v_Cl_0, v_Cl_1]

        force_H_0 = self.compute_force(r_rel=r_rel_0, speed=v_H_0, mass=self.m_H)
        r_H_1 = r_H_0 + v_H_0 * self.dt + 0.5 * force_H_0 * self.dt**2 / (self.m_H)
        r_H = [r_H_0, r_H_1]
        v_H_1 = v_H_0 + force_H_0 * self.dt / self.m_H
        v_H = [v_H_0, v_H_1]

        r_rel_1 = r_H_1 - r_Cl_1
        force_Cl_1 = self.compute_force(r_rel=-r_rel_1, speed=v_Cl_1, mass=self.m_Cl)
        force_H_1 = self.compute_force(r_rel=r_rel_1, speed=v_H_1, mass=self.m_H)
        
        r_rel = [r_rel_0, r_rel_1]
        force_H = [force_H_0, force_H_1]
        force_Cl = [force_Cl_0, force_Cl_1]

        for _ in range(2, len(time)):
            r_H_new, v_H_new, force_H_new = alg(
                pos=r_H, 
                speed=v_H, 
                mass=self.m_H, 
                r_rel=np.array(r_rel)
                )
            r_Cl_new, v_Cl_new, force_Cl_new = alg(
                pos=r_Cl, 
                speed=v_Cl, 
                mass=self.m_Cl, 
                r_rel=-np.array(r_rel)
                )

            r_H.append(r_H_new)
            r_Cl.append(r_Cl_new)
            v_H.append(v_H_new)
            v_Cl.append(v_Cl_new)
            force_H.append(force_H_new)
            force_Cl.append(force_Cl_new)
            
            r_rel.append(r_H_new - r_Cl_new)

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
        ### append byproducts to data
        if do_byproducts:
            byproducts = self.compute_byproducts(data, temp_window=temp_window)
            data.update(byproducts)

        if filename is not None:
            np.savez_compressed(filename, **data)

        return data, run_info
    
    def compute_byproducts(self, data, temp_window=100):
        byproducts = {}
        #byproducts["rotational_energy"] = self.rotational_energy()
        #byproducts["vibrational_energy"] = self.vibrational_energy()
        kin_h = self.kinetic_energy(data["v_H"], self.m_H)
        kin_cl = self.kinetic_energy(data["v_Cl"], self.m_Cl)
        kin_tot = kin_h + kin_cl
        byproducts["kinetic_energy_H"] = kin_h
        byproducts["kinetic_energy_Cl"] = kin_cl
        byproducts["kinetic_energy_total"] = kin_tot

        byproducts["potential_energy"] = self.potential_energy(data["r_rel"])

        byproducts["temperature"] = self.temperature(
            kinetic_energy=kin_tot,
            temp_window=temp_window
        )
        return byproducts
    
    @classmethod
    def load_from_file(cls, filename):
        data = np.load(filename)
        ## data has more keys than the __init__ parameters
        init_params = cls.__init__.__code__.co_varnames
        instance_data = {k: data[k] for k in data.files if k in init_params}
        instance = cls(**instance_data)

        return instance, data

    def temperature(self, kinetic_energy, temp_window):
        temp = (2/3) * kinetic_energy / self.k_B
        if temp_window is not None:
            window_size = int(temp_window / self.dt)
            if window_size % 2 == 0:
                window_size += 1
            temp = pd.Series(temp).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
        return temp
    
    @staticmethod
    def kinetic_energy(speed, mass):
        kin_energy = 0.5 * mass * np.sum(speed ** 2, axis=1)
        return kin_energy
    
    def potential_energy(self, r_rel):
        pot_energy = []
        if self.potential == 'harmonic':
            pot_energy = self.potential_harmonic(np.sqrt(np.sum(r_rel**2, axis=1)))
        elif self.potential == 'morse':
            pot_energy =  self.potential_morse(np.sqrt(np.sum(r_rel**2, axis=1)))
        else:
            raise ValueError(f"Unknown potential type: {self.potential}")
        return pot_energy
    
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
    
