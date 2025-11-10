import numpy as np
import pandas as pd
import datetime

from scipy import signal
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

        self.k = 2*self.D * self.alpha**2 # harmonic constant (taylor series expansion)

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
        return 0.5 * self.k * (r - self.r_0)**2
    
    def force_harmonic(self, r):
        """Gradient of the harmonic potential"""
        force = np.zeros(self.dimension)
        r_rel = np.sqrt(np.sum(r**2))
        for i in range(self.dimension):
            force[i] = -self.k * (r_rel - self.r_0) * r[i]/r_rel
        return force

    def force_random(self, mass, n_draws=1):
        """Random force according to the fluctuation-dissipation theorem"""
        r_amp = np.sqrt(2 * mass * self.k_B * self.T * self.gamma / self.dt)
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
        kin_h = self.kinetic_energy(data["v_H"], self.m_H)
        kin_cl = self.kinetic_energy(data["v_Cl"], self.m_Cl)
        kin_tot = kin_h + kin_cl
        byproducts["kinetic_energy_H"] = kin_h
        byproducts["kinetic_energy_Cl"] = kin_cl
        byproducts["kinetic_energy_total"] = kin_tot
        byproducts["rotation_energy"] = self.rotation_energy(
            r_H=data["r_H"],
            r_Cl=data["r_Cl"],
            v_H=data["v_H"],
            v_Cl=data["v_Cl"]
        )
        byproducts["vibration_energy"] = self.vibration_energy(
            r_H=data["r_H"],
            r_Cl=data["r_Cl"],
            v_H=data["v_H"],
            v_Cl=data["v_Cl"]
        )
        byproducts["translation_energy"] = self.translation_energy(
            r_H=data["r_H"],
            r_Cl=data["r_Cl"],
            v_H=data["v_H"],
            v_Cl=data["v_Cl"]
        )
        byproducts["potential_energy"] = self.potential_energy(data["r_rel"])
        byproducts["total_energy"] = byproducts["kinetic_energy_total"] + byproducts["potential_energy"]

        byproducts["temperature"] = self.temperature(
            energy=byproducts["total_energy"],
            temp_window=temp_window
        )

        return byproducts

    def qt_to_COM(self, qt_H, qt_Cl):
        return (self.m_H * qt_H + self.m_Cl * qt_Cl) / (self.m_Cl + self.m_H)
    
    @classmethod
    def load_from_file(cls, filename):
        data = np.load(filename)
        data_dict = {k: data[k] for k in data.files}
        run_opt = pd.read_json(Path(filename).with_suffix('.json')).to_dict()
        cls_data = {**data_dict, **run_opt['constants']}
        ## data has more keys than the __init__ parameters
        init_params = cls.__init__.__code__.co_varnames
        instance_data = {k: cls_data[k] for k in cls_data.keys() if k in init_params}
        instance = cls(**instance_data)

        return instance, data

    def temperature(self, energy, temp_window):
        temp = (2/7) * energy / self.k_B
        if temp_window is not None:
            if temp_window % 2 == 0:
                temp_window += 1
            temp = pd.Series(temp).rolling(window=temp_window, min_periods=1).mean().to_numpy()
        return temp
    
    @staticmethod
    def kinetic_energy(v, mass):
        return 0.5 * mass * np.sum(v ** 2, axis=1)
    
    def potential_energy(self, r_rel):
        pot_energy = []
        if self.potential == 'harmonic':
            pot_energy = self.potential_harmonic(np.sqrt(np.sum(r_rel**2, axis=1)))
        elif self.potential == 'morse':
            pot_energy =  self.potential_morse(np.sqrt(np.sum(r_rel**2, axis=1)))
        else:
            raise ValueError(f"Unknown potential type: {self.potential}")
        return pot_energy

    def rotation_energy(self, r_H, r_Cl, v_H, v_Cl):
        L_components = self.mu * np.cross(r_H - r_Cl, v_H - v_Cl, axisa=1, axisb=1)
        J = np.sum(L_components**2, axis=1)
        rot_e = 0.5 * J / (self.mu * np.sum((r_H - r_Cl)**2, axis=1))
        return rot_e
    
    def vibration_energy(self, r_H, r_Cl, v_H, v_Cl):
        dots = []
        for i in range(len(r_H)):
            dots.append(np.dot(v_H[i] - v_Cl[i], r_H[i] - r_Cl[i]))
        vib_e = 0.5 * self.mu * (np.array(dots) ** 2) / (np.sum((r_H - r_Cl)** 2, axis=1))
        return vib_e
    
    def translation_energy(self, r_H, r_Cl, v_H, v_Cl):
        v_cm = (self.m_H * v_H + self.m_Cl * v_Cl) / (self.m_H + self.m_Cl)
        transl_e = 0.5 * (self.m_H + self.m_Cl) * np.sum(v_cm**2, axis=1)
        return transl_e
    
def crosscorr_fft(x, y, maxlag=None):
    """
    Cross-correlation via FFT (positive lags only: y after x).
    x, y : 1D arrays, same length N
    Returns:
      lags (samples, 0..maxlag),
      rho (normalized cross-correlation for those lags)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape, "x and y must have same shape"
    N = x.size
    x -= x.mean()
    y -= y.mean()
    if maxlag is None:
        maxlag = N-1
    maxlag = min(maxlag, N-1)

    nfft = 1 << (2*N-1).bit_length()   # power of two >= 2*N
    fx = np.fft.rfft(x, nfft)
    fy = np.fft.rfft(y, nfft)
    Sxy = fx * np.conjugate(fy)
    cc_full = np.fft.irfft(Sxy, nfft)
    
    # cc_full[k] approximates sum_n x[n]*y[n+k] (with circular conv)
    cc = cc_full[:N] / (np.arange(N, 0, -1))  # divide by (N-k) to be unbiased-ish
    cc = cc[:maxlag+1]
    # normalize by stds (population std)
    denom = np.std(x, ddof=0) * np.std(y, ddof=0)
    rho = cc / denom
    lags = np.arange(0, maxlag+1, dtype=int)
    return lags, rho

def find_peak_lag(lags, rho, dt):
    """
    Find lag (in seconds) of the maximum correlation and its value.
    """
    k = np.argmax(rho)
    return lags[k], rho[k], k*dt

def block_bootstrap_crosscorr(x, y, dt, block_size_seconds=0.1, nboot=500, maxlag_seconds=None):
    """
    Block bootstrap to get 95% CI on cross-correlation curve.
    x, y : arrays
    block_size_seconds : block size for bootstrap (choose ~correlation time or a fraction)
    nboot : number of bootstrap samples
    Returns:
       lags_seconds, rho_mean, lower95, upper95
    """
    N = len(x)
    block_size = max(1, int(round(block_size_seconds / dt)))
    nblocks = int(np.ceil(N / block_size))
    maxlag = None
    if maxlag_seconds is not None:
        maxlag = int(round(maxlag_seconds / dt))
    lags, rho0 = crosscorr_fft(x, y, maxlag=maxlag)
    R = np.zeros((nboot, len(lags)))
    for i in range(nboot):
        starts = np.random.randint(0, N - block_size + 1, size=nblocks)
        xb = np.concatenate([x[s:s+block_size] for s in starts])[:N]
        yb = np.concatenate([y[s:s+block_size] for s in starts])[:N]
        _, rboot = crosscorr_fft(xb, yb, maxlag=len(lags)-1)
        R[i] = rboot
    lower = np.percentile(R, 2.5, axis=0)
    upper = np.percentile(R, 97.5, axis=0)
    return lags * dt, rho0, lower, upper

def coherence_spectrum(x, y, fs, nperseg=None):
    """
    Compute magnitude-squared coherence between x and y.
    fs : sampling frequency (1/dt)
    Returns f, Cxy(f)
    """
    if nperseg is None:
        nperseg = min(1024, len(x)//8)
    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg)
    return f, Cxy