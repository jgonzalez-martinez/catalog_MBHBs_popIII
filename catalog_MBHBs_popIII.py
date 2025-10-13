#!/usr/bin/env python
# coding: utf-8

# ### Exercise 2. Compute $N_{exp}$ and generate a catalog of observations

# In[7]:


import numpy as np
from scipy.integrate import cumtrapz, simps
from scipy.stats import beta
from astropy.cosmology import Planck15 as cosmo

#np.random.seed(0) 

# In[8]:

# Rate density models

def R_z_popIII_base(z):    
    return np.where(z <= 9, 2.11 * z, -1.8 * z + 35.2)    

#Target total rate (per year). Barausse MBHB catalog, extrapolated
TARGET_RATE_PER_YEAR_popIII = 352.6  # yr^-1

#Normalisation so the integral of Rz_Q3d dz = TARGET_RATE_PER_YEAR_Q3D 
zmin, zmax = 0.0, 19.0  
z_grid = np.linspace(zmin, zmax, 1000)
raw_integral = np.trapz(R_z_popIII_base(z_grid), z_grid)  
f_norm = TARGET_RATE_PER_YEAR_popIII / raw_integral      

def R_z_popIII(z):
    return f_norm * R_z_popIII_base(z)  # normalised dN/dz [yr^-1]


# In[9]:

# Redshift grid

#Need to specify the upper limit for redshift depending on R(z) chosen
z_grid = np.linspace(0, 19, 1000) 

# Merger rate
R_vals = R_z_popIII(z_grid)

# Comoving volume element in Gpc^3
dVc_dz = cosmo.differential_comoving_volume(z_grid).value  # [Mpc³/sr]
dVc_dz *= 4 * np.pi  # Full sky [Mpc³]
dVc_dz /= 1e9  # Convert to Gpc³

# Time dilation: observer-frame rate
#integrand = R_vals * dVc_dz / (1 + z_grid)
#integrand = R_vals * dVc_dz
integrand = R_vals

# Integrate to get N_expected per year
N_per_year = simps(integrand, z_grid)

# Total expected number of events
T_obs = 4  # years
N_expected = T_obs * N_per_year

print(f"Expected number of events: {N_expected:.1f}")


# In[10]:


N_drawn = np.random.poisson(N_expected)

print(f"Number of events: {N_drawn:.1f}")


# ### Now we generate the catalog


def sample_redshifts_from_Rz(N, z_min=0, z_max=19.0, grid_size=1000):
    z_vals = np.linspace(z_min, z_max, grid_size)

    # Comoving volume element [Gpc^3 / z]
    dVc_dz = cosmo.differential_comoving_volume(z_vals).value * 4.0 * np.pi / 1e9

    #dN_dz = R_z_popIII(z_vals) * dVc_dz / (1.0 + z_vals)
    dN_dz = R_z_popIII(z_vals)

    # Build CDF via cumulative integral and normalize
    cdf = cumtrapz(dN_dz, z_vals, initial=0.0)
    cdf /= cdf[-1]

    # Inverse-transform sampling
    u = np.random.rand(N)
    z_samples = np.interp(u, cdf, z_vals)
    return z_samples


# Primary mass: broken power-law
def sample_broken_power_law(N, alpha1, alpha2, m_min, m_break, m_max, grid_size):
    def p(m):
        return np.where(
            m < m_break,
            m**(-alpha1),
            (m_break**(alpha2 - alpha1)) * m**(-alpha2)
        )
    
    m_vals = np.linspace(m_min, m_max, grid_size)
    p_vals = p(m_vals)
    p_vals /= np.trapz(p_vals, m_vals)
    cdf_vals = np.cumsum(p_vals)
    cdf_vals /= cdf_vals[-1]
    return np.interp(np.random.rand(N), cdf_vals, m_vals)

# Mass ratio
def sample_mass_ratio(N, q_min=0.1, q_max=1.0):
    return np.random.uniform(q_min, q_max, size=N)

# Spins from beta distribution
def sample_spins(N, alpha=30, beta_param=3, seed=None):
    rng = np.random.default_rng(seed)
    return rng.beta(alpha, beta_param, size=N), rng.beta(alpha, beta_param, size=N)
    
# Geocentric times
def sample_geoctime(N, geoctime_i, geoctime_f):
    return np.random.uniform(geoctime_i, geoctime_f, size=N)

# Generate full synthetic population

def generate_synthetic_MBHB_population(N, geoctime_i, geoctime_f,                                       
                                       alpha1=1.5, alpha2=2.5,
                                       m_min=1e3, m_break=1e6, m_max=1e7,
                                       q_min=0.1, q_max=1.0,
                                       spin_alpha=30, spin_beta=3,
                                       # NEW options
                                       z_min=0, z_max=19.0, grid_size=1000):
    assert geoctime_f > geoctime_i, "geoctime_f must be greater than geoctime_i"

    # Redshift sampling 
    z = sample_redshifts_from_Rz(        
            N,
            z_min=z_min, z_max=z_max,
            grid_size=grid_size
        )

    # Primary mass (broken power law)
    m1 = sample_broken_power_law(N, alpha1, alpha2, m_min, m_break, m_max, grid_size)

    # Mass ratio and secondary mass
    q  = sample_mass_ratio(N, q_min, q_max)
    m2 = q * m1

    # Spins
    a_1, a_2 = sample_spins(N, spin_alpha, spin_beta)

    # Geocentric times 
    geoc_time = sample_geoctime(N, geoctime_i, geoctime_f)

    return {
        "z": z,
        "m1": m1,
        "m2": m2,
        "q": q,
        "a_1": a_1,
        "a_2": a_2,
        "geoctime": geoc_time,
    }

# In[13]:


#population = generate_synthetic_MBHB_population(N=N_drawn, geoctime_i, geoctime_f)



