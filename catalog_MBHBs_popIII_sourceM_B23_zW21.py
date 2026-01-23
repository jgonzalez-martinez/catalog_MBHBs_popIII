#!/usr/bin/env python
# coding: utf-8

# ### Exercise 2. Compute $N_{exp}$ and generate a catalog of observations

# In[7]:


import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.stats import beta
from astropy.cosmology import Planck18 as cosmo

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
N_per_year = simpson(integrand, x=z_grid)

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
    cdf = cumulative_trapezoid(dN_dz, z_vals, initial=0.0)
    cdf /= cdf[-1]

    # Inverse-transform sampling
    z_samples = np.interp(np.random.rand(N), cdf, z_vals)
    return z_samples


def sample_M_power_law(N, alpha, M_min, M_max, grid_size):
    def p(M):
        return M**(alpha)        
    
    M_vals = np.linspace(M_min, M_max, grid_size)    
    p_vals = p(M_vals)
    p_vals /= np.trapz(p_vals, M_vals)
    cdf_vals = np.cumsum(p_vals)
    cdf_vals /= cdf_vals[-1]
    return np.interp(np.random.rand(N), cdf_vals, M_vals)
    
# Mass ratio
def sample_mass_ratio_broken_power_law(N, alpha1, alpha2, q_min, q_break, q_max, grid_size):
    q_vals = np.linspace(q_min, q_max, grid_size)
    pdf = np.zeros_like(q_vals, dtype=float)

    # Compute normalisation constant A1 and continuity-adjusted A2
    def normalisation_constant(alpha1, alpha2, q_min, q_break, q_max):
        term1 = (q_break**(alpha1 + 1) - q_min**(alpha1 + 1)) / (alpha1 + 1)
        term2 = (q_max**(alpha2 + 1) - q_break**(alpha2 + 1)) / (alpha2 + 1)
        return 1.0 / (term1 + q_break**(alpha1 - alpha2) * term2)

    A1 = normalisation_constant(alpha1, alpha2, q_min, q_break, q_max)
    A2 = A1 * q_break**(alpha1 - alpha2)

    # Evaluate PDF
    mask1 = (q_vals >= q_min) & (q_vals <= q_break)
    mask2 = (q_vals > q_break) & (q_vals <= q_max)

    pdf[mask1] = A1 * q_vals[mask1]**(alpha1)
    pdf[mask2] = A2 * q_vals[mask2]**(alpha2)
    
    p_vals = pdf
    p_vals /= np.trapz(p_vals, q_vals)
    cdf_vals = np.cumsum(p_vals)
    cdf_vals /= cdf_vals[-1]
    return np.interp(np.random.rand(N), cdf_vals, q_vals)

# Spins from beta distribution
def sample_spins(N, alpha=400, beta_param=23, seed=None):
    rng = np.random.default_rng(seed)
    return rng.beta(alpha, beta_param, size=N), rng.beta(alpha, beta_param, size=N)
    
# Geocentric times
def sample_geoctime(N, geoctime_i, geoctime_f):
    return np.random.uniform(geoctime_i, geoctime_f, size=N)

# Generate full synthetic population

def generate_synthetic_MBHB_population(N, geoctime_i, geoctime_f,                                       
                                       alpha=-1.30, #-1.20 yielded 320 detected mergers in 4 years; original -0.90d
                                       M_min=1e3, M_max=1e8,
                                       q_alpha1=0.80, q_alpha2=-0.14,
                                       q_min=0.1, q_break=0.88, q_max=1.0,
                                       spin_alpha=400, spin_beta=2,                                      
                                       z_min=0, z_max=19.0, grid_size=50000):
    assert geoctime_f > geoctime_i, "geoctime_f must be greater than geoctime_i"

    # Redshift sampling 
    z = sample_redshifts_from_Rz(           
            N,
            z_min=z_min, z_max=z_max,
            grid_size=grid_size
            )

    # total source-frame mass (broken power law)
    M = sample_M_power_law(N, alpha, M_min, M_max, grid_size)

    # Mass ratio and secondary mass
    q  = sample_mass_ratio_broken_power_law(N, q_alpha1, q_alpha2, q_min, q_break, q_max, grid_size)    
    
    m1 = M/(1 + q)
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


