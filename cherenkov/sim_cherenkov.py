"""
Simulation and plot fields of Cherenkov radiation using Meep. Also plot the case where no Cherenkov radiation is expected (v<c/n).
This is largely from https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py
and with guidance from https://www.mail-archive.com/meep-discuss@ab-initio.mit.edu/msg06406.html to diminish artifacts.

Note to self: conda activate meep
"""


import meep as mp
from meep.visualization import _add_colorbar
import numpy as np
import matplotlib.pyplot as plt
import os


def setup_run_2d_sim(sx, sy, resolution, dpml, vq, n, sim_data_dict, num_S=2, force_complex_fields=False, plot_last_timestep=False):
    """Setup 2d Cherenkov simulation following the example:
    https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py
    Cherenkov raditation occurs if `vq` > 1/`n` (c=1 units in Meep).

    Parameters
    ----------
    sx, sy: ints
        Total simulation grid size in x, y direction; grid centered at origin  
    resolution: int
        *Computational* grid resolution per distance unit, i.e. computational grid size is sx*resolution
        https://meep.readthedocs.io/en/latest/Python_User_Interface/#simulation
    dpml : int
        Number of grid points to 'pad' the grid
    vq : float
        Velocity of charged particle (units of c)
    n: float
        Index of refraction of the medium
    sim_data_dict : dict
        Dictionary to store simulation data at various time steps. This will be modified in place in this function.
    force_complex_fields : bool
        Meep simulation parameter https://meep.readthedocs.io/en/master/FAQ/#how-do-i-compute-the-time-average-of-the-harmonic-fields
    num_S : int
        Save Poynting vector plot(s) with meep utils every `num_S` time steps. 
        Save electric and magnetic field plot(s) with meep utils every 5*`num_S` time steps. 
        For the case of no Cherenkov radiation, consider using a smaller `num_S` because the particle travels more slowly so there are more time steps.
    plot_last_timestep : bool
        If True, plot various fields at last time step (and last time step only) of the simulation after the simulation is complete.
    """
    # 2D simulation, so `sz=0`
    cell_size = mp.Vector3(sx, sy, 0)
    # PML or perfectly matched layer absorbs fields at all the boundaries (no reflection, etc); https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/
    pml_layers = [mp.PML(thickness=dpml)]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        default_material=mp.Medium(index=n),
        boundary_layers=pml_layers,
        dimensions=2,
        eps_averaging=True,
        # Recommended to use smaller Courant by https://www.mail-archive.com/meep-discuss@ab-initio.mit.edu/msg06406.html
        Courant=0.1, 
        # Recommended to use force_complex_fields=True by https://meep.readthedocs.io/en/master/FAQ/#how-do-i-compute-the-time-average-of-the-harmonic-fields in order to compute time average of (harmonic) fields
        force_complex_fields=force_complex_fields,  
        )   

    # Incorporate `force_complex_fields` in the output directory name
    field_dict = {True: "complex_fields", False: "real_fields"}
    # Check if Cherenkov radiation is expected
    if vq > 1/n:
        output_dir = f"sim_output/cherenkov/{field_dict[force_complex_fields]}/res{resolution}_v{vq:.2f}c_n{n}"
    if vq <= 1/n:
        output_dir = f"sim_output/no_cherenkov/{field_dict[force_complex_fields]}/res{resolution}_v{vq:.2f}c_n{n}"
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Plots saved by `output_png`
    sim.use_output_directory(os.path.join(output_dir, "meep_plots"))
    
    # Point charge moving along x direction 
    # Use small frequency to model a moving point charge as an electric current, 1e-10 as in https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py
    sim.sources.append(mp.Source(
        mp.ContinuousSource(frequency=1e-10),
        component=mp.Ex,
        center=mp.Vector3(-0.5 * sx + dpml),
    ))

    # mp.FluxRegion ?

    num_EH = num_S * 5
    sim.run(
        # Move source at every timestep
        lambda sim: move_source(sim, sx, vq, dpml),
        # Every `x`th timestep:
        mp.at_every(num_S, lambda sim: save_sim(sim, sim_data_dict)),
        # These plot the real parts of the field only?
        mp.at_every(num_S, mp.output_png(mp.Sx, "-vZc dkbluered -M 1")),
        mp.at_every(num_S, mp.output_png(mp.Sy, "-vZc dkbluered -M 1")),
        # Note: increase resolution to see the E, H fields better?
        # mp.at_every(num_EH, mp.output_png(mp.Ex, "-vZc dkbluered -M 1")),
        # mp.at_every(num_EH, mp.output_png(mp.Ey, "-vZc dkbluered -M 1")),
        # mp.at_every(num_EH, mp.output_png(mp.Ez, "-vZc dkbluered -M 1")),
        # mp.at_every(num_EH, mp.output_png(mp.Hx, "-vZc dkbluered -M 1")),
        # mp.at_every(num_EH, mp.output_png(mp.Hy, "-vZc dkbluered -M 1")),
        # mp.at_every(num_EH, mp.output_png(mp.Hz, "-vZc dkbluered -M 1")),
        # TODO ? `mp.output_poynting` and wrap in `synchronized_magnetic`, https://meep.readthedocs.io/en/master/Python_User_Interface/#output-functions_1
        # mp.at_every(num_S, lambda sim: sim.output_poynting(...))
        # NOTE: https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py timesteps until the full grid is traversed
        until=sx/vq,
        )
    
    plot_avg_poynting_flux(sim_data_dict, sx, sy, odir=output_dir)

    if plot_last_timestep:
        # This will plot the LAST time step of the simulation
        # Dictionary organizing values to plot
        # Format for each entry is key: {value from simulation to plot, label for colorbar}
        vals_dict = {"Hz": {"val": mp.Hz, "lbl": "$H_z$ [a.u.]"},
                    "Sy": {"val": mp.Sy, "lbl": "Poynting flux $S_y$ [a.u.]"},
                    "Ex": {"val": mp.Ex, "lbl": "$E_x$ [a.u.]"},
                    "Sx": {"val": mp.Sx, "lbl": "Poynting flux $S_x$ [a.u.]"}
                    }
        for i, key in enumerate(vals_dict.keys()):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax = sim.plot2D(fields=vals_dict[key]['val'], ax=ax, plot_sources=False, plot_geom=False)
            # For grid size 60x60 and resolution 10:
            # ax.images[0]: shape=(601, 601). Produces greyscale colorbar. Size included grid edges/`dpml`?
            # ax.images[1]: shape=(600, 600). Produces red/blue colorbar. This matches the expected grid size and the color of the radiation.
            im = ax.images[1] 
            vmin, vmax = im.get_clim()
            _add_colorbar(ax=ax, cmap=im.get_cmap(), vmin=vmin, vmax=vmax, default_label=vals_dict[key]['lbl'])
            # ax.set_xlim([-sx/5, sx/5])
            # ax.set_ylim([-sy/5, sy/5])
            # for i, im in enumerate(ax.images):
            #     print(f"ax.images[{i}]: shape={im.get_array().shape}")
            fig.savefig(os.path.join(output_dir, f'{key}.png'), dpi=600) 
            print(os.path.join(output_dir, f'{key}.png'))
            plt.close(fig)
        
    return sim, sim_data_dict


def save_sim(sim, sim_data_dict, center=mp.Vector3()):
    """Save simulation `sim` information at current timestep to `sim_data_dict` (add to dictionary in this function).
    This dict is formatted as:
    {
    meep_time1: {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez,
                  "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz,
                   "Sx": mp.Sx, "Sy": mp.Sy, "Sz": mp.Sz,    
                  },
    meep_time2: {...},
    }
    
    """
    size = sim.cell_size
    sim_data_dict[sim.meep_time()] = {"Ex": sim.get_array(center=center, size=size, component=mp.Ex), 
                                      "Ey": sim.get_array(center=center, size=size, component=mp.Ey),
                                      "Ez": sim.get_array(center=center, size=size, component=mp.Ez),
                                      "Hx": sim.get_array(center=center, size=size, component=mp.Hx),
                                      "Hy": sim.get_array(center=center, size=size, component=mp.Hy),
                                      "Hz": sim.get_array(center=center, size=size, component=mp.Hz),
                                      "Sx": sim.get_array(center=center, size=size, component=mp.Sx),
                                      "Sy": sim.get_array(center=center, size=size, component=mp.Sy),
                                      "Sz": sim.get_array(center=center, size=size, component=mp.Sz),
                                      }
    return


def move_source(sim, sx, vq, dpml): 
    """This is from https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py"""
    sim.change_sources(
        [
            mp.Source(
                # https://meep.readthedocs.io/en/latest/FAQ/#how-do-i-model-a-moving-point-charge
                # Use small frequency to model a moving point charge as an electric current
                # Using the given frequency in https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py
                mp.ContinuousSource(frequency=1e-10),
                # Move current/source along x direction
                component=mp.Ex,
                # Center of the current/source in the cell
                # x = x0 + vt, where x0=-sx/2 + dpml puts the current/source out of the PML (via +dpml) and at the left side of the simulation grid (grid centered at 0)
                center=mp.Vector3(-0.5 * sx + dpml + vq * sim.meep_time()),
            )
        ]
    )


def plot_avg_poynting_flux(sim_data_dict, sx, sy, odir, calc_tot=False):
    """Plot the average(?) Poynting flux over all time steps using the data in `sim_data_dict`.
    
    Whether or not this is the average depends on other simulation parameters `force_complex_fields`):
    'For a linear system, you can use a ContinuousSource with force_complex_fields=True and time-step the fields until all transients have disappeared. Once the fields have reached steady state, the instantaneous intensity |E|2/2 or Poynting flux Real[E* \times H]/2 is equivalent to the time average. If you don't use complex fields, then these are just the instantaneous values at a given time, and will oscillate.'
    - https://meep.readthedocs.io/en/master/FAQ/#how-do-i-compute-the-time-average-of-the-harmonic-fields

    The medium is linear if \vec{P} \propto \vec{E} ...
    The electric field of the moving charge will never disappear, so in that sense the system will never reach steady state?
    """
    if calc_tot:
        # Total of each Poynting vector component over the simulation
        Sx_tot, Sy_tot, Sz_tot = 0, 0, 0
    # Total number of time steps that were saved
    num_timesteps = len(sim_data_dict.keys())
    for i, k in enumerate(sim_data_dict.keys()):
        Sx = sim_data_dict[k]['Sx']
        Sy = sim_data_dict[k]['Sy']
        Sz = sim_data_dict[k]['Sz']
        if calc_tot:
            Sx_tot += Sx
            Sy_tot += Sy
            Sz_tot += Sz
            # Average over the number of time steps? This won't change the shape of the radiation pattern, just the magnitude which is in arbitrary units anyway.
            S_avg = np.sqrt(Sx_tot**2 + Sy_tot**2 + Sz_tot**2) / num_timesteps
            # TODO? plot the average?
        S =  np.sqrt(Sx**2 + Sy**2 + Sz**2)
        # cmap = plt.cm.Greys.copy()
        cmap = plt.cm.inferno.copy()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(S.T, origin='lower', cmap=cmap)
        fig.colorbar(im, ax=ax, label=r'Poynting flux $|\langle S \rangle|$ [a.u.]')
        plt.xlabel('X [a.u.]')
        plt.ylabel('Y [a.u.]')
        # matplotlib plots
        output_dir = f"{odir}/mpl_plots/poynting_flux"
        # Check if output directory exists, create if not
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        fn = os.path.join(output_dir, f'S_mag_{i}.png')
        plt.savefig(fn, dpi=600)
        print(fn)
        plt.close(fig)

    return


def plot_sim(sim, vq, n, center=mp.Vector3(), iter=''):
    """Plot various simulation `sim` results:
    
    Parameters
    ----------
    sim : meep simulation object
    vq : float
        Velocity of charged particle (units of c)
    n: float
        Index of refraction of the medium
    iter : int or empty string
        This is for the case where multiple iterations (time steps) of the same simulation are to be plotted separately; `iter` will be appended to the plot filename.
    """
    # Get field components, for cross product, etc, over the full simulation grid
    if isinstance(sim, dict):
        # Electric field
        Ex = sim["Ex"]
        Ey = sim["Ey"]
        Ez = sim["Ez"]
        # Magnetic field
        Hx = sim["Hx"]
        Hy = sim["Hy"]
        Hz = sim["Hz"]
    else:
        Ex = sim.get_array(center=center, size=sim.cell_size, component=mp.Ex)
        Ey = sim.get_array(center=center, size=sim.cell_size, component=mp.Ey)
        Ez = sim.get_array(center=center, size=sim.cell_size, component=mp.Ez)
        Hx = sim.get_array(center=center, size=sim.cell_size, component=mp.Hx)
        Hy = sim.get_array(center=center, size=sim.cell_size, component=mp.Hy)
        Hz = sim.get_array(center=center, size=sim.cell_size, component=mp.Hz)
    E = np.stack([Ex, Ey, Ez], axis=-1) 
    H = np.stack([Hx, Hy, Hz], axis=-1)
    # Poynting flux / radiation
    S = np.cross(E, H)
    S_mag = np.linalg.norm(S, axis=-1)

    # Check if Cherenkov radiation is expected
    if vq > 1/n:
        title = rf'for $v_q > c/n$'
        # matplotlib plots
        output_dir = "sim_output/cherenkov/plot_sim"
    if vq <= 1/n:
        title = rf'for $v_q \leq c/n$'
        output_dir = "sim_output/no_cherenkov/plot_sim"

    # Dictionary organizing values to plot
    # Format for each entry is key: {value from simulation to plot, name of plot, label for colorbar}
    vals_dict = {"Hz": {"val": Hz, "fn": f"Hz{iter}.png", "lbl": "$H_z$"},
                 # "Ey": {"val": Ey, "fn": f"Ey{iter}.png", "lbl": "$E_y$"},
                 "S": {"val": S_mag, "fn": f"S{iter}.png", "lbl": "Poynting flux $|S|$"},
                 }
    
    for i, key in enumerate(vals_dict.keys()):
        # Reliable Python precision
        t = 1e-16
        # Plot zeros in white   
        val_masked = np.ma.masked_where(abs(vals_dict[key]["val"]) <= t, vals_dict[key]["val"])  
        cmap = plt.cm.vanimo.copy()
        cmap.set_under('white')
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(val_masked.T, cmap=cmap, origin='lower')
        #im = ax.imshow(vals_dict[key]["val"].T, cmap=cmap, origin='lower')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(rf'{vals_dict[key]["lbl"]} [a.u.]')
        ax.set_title(rf'{vals_dict[key]["lbl"]} {title}, $\vec{{v}}_q=v_q\hat{{x}}$')
        ax.set_xlabel('x [a.u.]')
        ax.set_ylabel('y [a.u.]')
        ax.set_aspect('equal')
        # plt.tight_layout()
        fig.savefig(f'{output_dir}/{vals_dict[key]["fn"]}')
        # Zoom in to where the cone is propagating at the center of the grid
        # ax = plt.gca()
        # xlim = ax.get_xlim()
        # xrange = xlim[1] - xlim[0]
        # # Second half of x grid
        # ax.set_xlim(0.5*xlim[1], xlim[1])
        # # Middle half of y grid
        # ax.set_ylim(0.25*xrange, 0.75*xrange)
        # ax.set_aspect('equal')
        # fig.savefig(f'{output_dir}/zoom_{vals_dict[key]["fn"]}')
        plt.close(fig)

    return None


if __name__ == "__main__":
    # Index of refraction of medium (1.5 used https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py)
    N = 1.5
    # Size of simulation grid in x and y direction (60 used in https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py)
    SX, SY = 60, 60 
    # Resolution of simulation grid (10 used in https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py)
    RES = 10
    # Speed of charged particle in units of speed of light. This speed will produce Cherenkov radiation, v > 1/n (c=1).
    # (0.7 used in https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py)
    VQ_CHERENKOV = 0.7
    # This speed will NOT produce Cherenkov radiation, v <= 1/n (c=1)    
    VQ_BELOW_THRESHOLD = 0.35
    # (1 used in https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py)
    DPLM = 1.0  # PML thickness
  
    # cf: complex field 
    for cf in [True, False]:
    # `mp.at_every(2, ...)` is used in the example at  https://github.com/NanoComp/meep/blob/master/python/examples/cherenkov-radiation.py -> x=2
        sim, sim_data_dict = setup_run_2d_sim(sx=SX, sy=SY, resolution=RES, dpml=DPLM, vq=VQ_BELOW_THRESHOLD, n=N, sim_data_dict={}, force_complex_fields=cf, num_S=4)
        sim, sim_data_dict = setup_run_2d_sim(sx=SX, sy=SY, resolution=RES, dpml=DPLM, vq=VQ_CHERENKOV, n=N, sim_data_dict={}, force_complex_fields=cf, num_S=2)
