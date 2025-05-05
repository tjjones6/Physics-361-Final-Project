import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator
import os
import h5py
from datetime import datetime

class ParticleAdvectionSimulation:
    """
    Simulate inviscid, incompressible flow particle advection around a cylinder,
    compute lift and drag coefficients vs iterations, animate tracer particles,
    and efficiently store flow-field snapshots for later POD analysis.
    """
    def __init__(self,
                 grid_size=100,
                 aspect_ratio=3,
                 max_particles=np.inf,
                 dt=0.1,
                 rho=1.0,
                 U_inf=1.0,
                 data_directory='flow_data'):
        # Basic parameters
        self.s             = grid_size
        self.ar            = aspect_ratio
        self.dt            = dt
        self.rho           = rho
        self.U_inf         = U_inf
        self.max_particles = max_particles
        self.time_elapsed  = 0.0
        self.iteration     = 0

        # Cylinder geometry
        self.radius   = 10.0
        self.D        = 2 * self.radius
        self.center_x = self.s * self.ar / 4.0
        self.center_y = self.s / 2.0

        # Computational grid
        self.grid_y = np.arange(1, self.s + 1)
        self.grid_x = np.arange(1, self.s * self.ar + 1)
        self.X, self.Y = np.meshgrid(self.grid_x, self.grid_y)

        # Flow fields
        self.p  = np.zeros((self.s, self.s * self.ar))
        self.vx = np.zeros_like(self.p)
        self.vy = np.zeros_like(self.p)

        # Derived (not stored)
        self.vorticity          = np.zeros_like(self.p)
        self.velocity_magnitude = np.zeros_like(self.p)

        # Cylinder mask & boundary
        self.cyl_mask = ((self.X - self.center_x)**2 + (self.Y - self.center_y)**2) <= self.radius**2
        self._find_cylinder_boundary()

        # Tracer particles
        self._init_particles()

        # Jacobi kernel
        self.J = np.array([[0,1,0],[1,0,1],[0,1,0]], float) / 4.0

        # Histories
        self.iterations = []
        self.times      = []
        self.Cd_list    = []
        self.Cl_list    = []
        self.Cp_history = []

        # Snapshots
        self.snapshot_iterations = []
        self.vx_snapshots        = []
        self.vy_snapshots        = []
        self.p_snapshots         = []

        # Angular bins
        self.theta_deg = np.linspace(0, 360, 72, endpoint=False)

        # Data directory
        self.data_directory = data_directory
        os.makedirs(self.data_directory, exist_ok=True)

    def _find_cylinder_boundary(self):
        cyl = self.cyl_mask
        ny, nx = cyl.shape
        boundary = np.zeros_like(cyl, bool)
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            shifted = np.zeros_like(cyl)
            shifted[max(0,di):min(ny,ny+di),
                    max(0,dj):min(nx,nx+dj)] = \
                cyl[max(0,-di):min(ny,ny-di),
                    max(0,-dj):min(nx,nx-dj)]
            boundary |= cyl & ~shifted

        self.boundary_idx = np.argwhere(boundary)
        self.boundary_angles = np.degrees(
            np.arctan2(self.boundary_idx[:,0] - self.center_y,
                       self.boundary_idx[:,1] - self.center_x)
        ) % 360

    def _init_particles(self):
        self.px  = np.full((self.s,), 10.0)
        self.py  = np.arange(1, self.s + 1, dtype=float)
        self.pxo = self.px.copy()
        self.pyo = self.py.copy()

    def interp2d(self, field, x, y):
        interp = RegularGridInterpolator(
            (self.grid_y, self.grid_x),
            field,
            bounds_error=False,
            fill_value=0.0
        )
        pts = np.column_stack((y.flatten(), x.flatten()))
        return interp(pts).reshape(x.shape)

    def rk4(self, px, py, h):
        k1x = self.interp2d(self.vx, px, py)
        k1y = self.interp2d(self.vy, px, py)
        k2x = self.interp2d(self.vx, px + 0.5*h*k1x, py + 0.5*h*k1y)
        k2y = self.interp2d(self.vy, px + 0.5*h*k1x, py + 0.5*h*k1y)
        k3x = self.interp2d(self.vx, px + 0.5*h*k2x, py + 0.5*h*k2y)
        k3y = self.interp2d(self.vy, px + 0.5*h*k2x, py + 0.5*h*k2y)
        k4x = self.interp2d(self.vx, px +   h*k3x, py +   h*k3y)
        k4y = self.interp2d(self.vy, px +   h*k3x, py +   h*k3y)
        new_px = px + (h/6)*(k1x + 2*k2x + 2*k3x + k4x)
        new_py = py + (h/6)*(k1y + 2*k2y + 2*k3y + k4y)
        return new_px, new_py

    def compute_derived_quantities(self):
        dvdx = np.gradient(self.vy, axis=1)
        dudy = np.gradient(self.vx, axis=0)
        self.vorticity          = dvdx - dudy
        self.velocity_magnitude = np.sqrt(self.vx**2 + self.vy**2)

    def compute_forces(self):
        F = np.zeros(2)
        ds = 1.0
        for i, j in self.boundary_idx:
            nx = (self.X[i,j] - self.center_x)
            ny = (self.Y[i,j] - self.center_y)
            norm = np.hypot(nx, ny)
            nx, ny = nx/norm, ny/norm
            F += -self.p[i,j] * np.array([nx, ny]) * ds
        q  = 0.5 * self.rho * self.U_inf**2
        Cd = F[0] / (q * self.D)
        Cl = F[1] / (q * self.D)
        return Cd, Cl

    def compute_pressure_coefficient(self):
        q      = 0.5 * self.rho * self.U_inf**2
        p_vals = self.p[self.boundary_idx[:,0], self.boundary_idx[:,1]]
        Cp_vals = p_vals / q
        bins    = np.digitize(self.boundary_angles, np.linspace(0,360,len(self.theta_deg)+1))
        Cp_binned = np.zeros(len(self.theta_deg))
        counts    = np.zeros(len(self.theta_deg))
        for idx, b in enumerate(bins):
            if 1 <= b <= len(self.theta_deg):
                Cp_binned[b-1] += Cp_vals[idx]
                counts[b-1]    += 1
        mask = counts > 0
        Cp_binned[mask] /= counts[mask]
        return Cp_binned

    def update(self):
        # zero inside cylinder
        self.vx[self.cyl_mask] = 0.0
        self.vy[self.cyl_mask] = 0.0

        # pressure solve
        rhs = 0.5*(np.gradient(self.vx, axis=1) +
                  np.gradient(self.vy, axis=0))
        for _ in range(100):
            self.p = convolve2d(self.p, self.J, mode='same') - rhs
            self.p[0,:], self.p[-1,:] = self.p[1,:], self.p[-2,:]
            self.p[:,:5], self.p[:,-5:] = 1.0, 0.0

        # velocity correction
        dpdx = np.gradient(self.p, axis=1)
        dpdy = np.gradient(self.p, axis=0)
        self.vx[1:-1,1:-1] -= dpdx[1:-1,1:-1]
        self.vy[1:-1,1:-1] -= dpdy[1:-1,1:-1]
        self.vx[self.cyl_mask] = 0.0
        self.vy[self.cyl_mask] = 0.0

        # semi-lagrangian
        backx, backy = self.rk4(self.X, self.Y, -self.dt)
        self.vx = self.interp2d(self.vx, backx, backy)
        self.vy = self.interp2d(self.vy, backx, backy)

        # derived
        self.compute_derived_quantities()

        # time
        self.time_elapsed += self.dt
        self.iteration    += 1

        # tracers
        self.px, self.py = self.rk4(self.px, self.py, self.dt)
        self.px = np.concatenate((self.px, self.pxo))
        self.py = np.concatenate((self.py, self.pyo))
        if self.px.size > self.max_particles:
            self.px = self.px[-int(self.max_particles):]
            self.py = self.py[-int(self.max_particles):]
        mask = ((self.px>=0)&(self.px<self.s*self.ar)&
                ((self.px-self.center_x)**2 + (self.py-self.center_y)**2 > self.radius**2))
        self.px, self.py = self.px[mask], self.py[mask]

        # forces & Cp
        Cd, Cl = self.compute_forces()
        Cp     = self.compute_pressure_coefficient()

        self.iterations.append(self.iteration)
        self.times.append(self.time_elapsed)
        self.Cd_list.append(Cd)
        self.Cl_list.append(Cl)
        self.Cp_history.append(Cp)

    def save_forces(self, filepath):
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('iteration',   data=self.iterations)
            f.create_dataset('time',        data=self.times)
            f.create_dataset('Cd',          data=self.Cd_list)
            f.create_dataset('Cl',          data=self.Cl_list)
            f.create_dataset('theta_deg',   data=self.theta_deg)
            f.create_dataset('Cp_history',  data=self.Cp_history)
            f.attrs.update({
                'rho':       self.rho,
                'U_inf':     self.U_inf,
                'D':         self.D,
                'timestamp': datetime.now().isoformat()
            })

    def save_flow_field_hdf5(self, filepath):
        vx = np.array(self.vx_snapshots, dtype=np.float32)
        vy = np.array(self.vy_snapshots, dtype=np.float32)
        p  = np.array(self.p_snapshots,  dtype=np.float32)

        if vx.shape[0] == 0:
            print("Warning: no snapshots to save.")
            return

        with h5py.File(filepath, 'w') as f:
            f.create_dataset('iterations', data=self.snapshot_iterations)
            chunk0 = 1
            f.create_dataset('vx', data=vx,
                             dtype='f4', compression='gzip',
                             compression_opts=4,
                             chunks=(chunk0,)+vx.shape[1:])
            f.create_dataset('vy', data=vy,
                             dtype='f4', compression='gzip',
                             compression_opts=4,
                             chunks=(chunk0,)+vy.shape[1:])
            f.create_dataset('p', data=p,
                             dtype='f4', compression='gzip',
                             compression_opts=4,
                             chunks=(chunk0,)+p.shape[1:])
            f.attrs.update({
                'grid_size':       self.s,
                'aspect_ratio':    self.ar,
                'cylinder_center': (self.center_x, self.center_y),
                'cylinder_radius': self.radius,
                'dt':              self.dt,
                'rho':             self.rho,
                'U_inf':           self.U_inf,
                'timestamp':       datetime.now().isoformat()
            })
        print(f"Compressed flow fields saved -> {filepath}")

    def run(self,
            LENGTH=3.0,
            HEIGHT=1.0,
            final_time=10.0,
            snapshot_interval=0.5,
            save_anim='cylinder_advection.mp4',
            save_forces='forces.h5',
            save_fields='flow_fields.h5'):
        # iterations and snapshots
        n_iters   = int(final_time / self.dt)
        snap_step = max(1, int(snapshot_interval / self.dt))

        dx_anim = LENGTH / (self.s * self.ar - 1)
        dy_anim = HEIGHT / (self.s - 1)

        fig, ax = plt.subplots(figsize=(14,8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_xlim(0, LENGTH)
        ax.set_ylim(0, HEIGHT)
        ax.set_aspect('equal','box')
        ax.set_xlabel('x', color='white')
        ax.set_ylabel('y', color='white')
        ax.set_title('Particle Advection Around Cylinder', color='white')
        ax.tick_params(colors='white')

        center_x_anim = (self.center_x - 1)*dx_anim
        center_y_anim = (self.center_y - 1)*dy_anim
        radius_anim   = self.radius*dx_anim
        circ = plt.Circle((center_x_anim, center_y_anim),
                          radius_anim, edgecolor='cyan', fill=True, alpha=0.7)
        ax.add_patch(circ)

        scatter = ax.scatter([],[],s=0.5,c='white',alpha=0.75)

        def init():
            scatter.set_offsets(np.empty((0,2)))
            return scatter,

        def animate(frame):
            self.update()
            # snapshot?
            if self.iteration % snap_step == 0:
                self.snapshot_iterations.append(self.iteration)
                self.vx_snapshots.append(self.vx.copy())
                self.vy_snapshots.append(self.vy.copy())
                self.p_snapshots.append(self.p.copy())

            x_part = (self.px - 1)*dx_anim
            y_part = (self.py - 1)*dy_anim
            scatter.set_offsets(np.column_stack((x_part, y_part)))

            # progress
            perc = (frame+1)/n_iters
            bar_n = int(perc*40)
            bar = '#' * bar_n + '-'*(40-bar_n)
            print(f"\rIter {self.iteration}/{n_iters}: |{bar}| {perc*100:5.1f}% ", end='', flush=True)

            return scatter,

        ani = FuncAnimation(fig, animate, frames=n_iters,
                            init_func=init, blit=True,
                            interval=self.dt*1000, repeat=False)
        ani.save(save_anim, writer='ffmpeg', fps=60, dpi=300)
        plt.close(fig)
        print(f"\nAnimation saved -> {save_anim}")

        # save data
        self.save_forces   (os.path.join(self.data_directory, save_forces))
        self.save_flow_field_hdf5(os.path.join(self.data_directory, save_fields))

        # plot forces
        plt.figure(figsize=(10,6))
        plt.plot(self.iterations, self.Cd_list, label='C_D')
        plt.plot(self.iterations, self.Cl_list, label='C_L')
        plt.xlabel('Iteration'); plt.ylabel('Coefficient')
        plt.legend(); plt.grid(True)
        plt.title('Force Coefficients vs Iteration')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_directory,'force_coefficients.png'),dpi=300)
        plt.close()

        # plot Cp
        avg_Cp = np.mean(self.Cp_history,axis=0)
        plt.figure(figsize=(10,6))
        plt.plot(self.theta_deg,avg_Cp,'-o')
        plt.xlabel('Angle (deg)'); plt.ylabel('Average Cp')
        plt.grid(True); plt.title('Average Pressure Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_directory,'pressure_coefficient.png'),dpi=300)
        plt.close()

        return ani

if __name__=='__main__':
    sim = ParticleAdvectionSimulation(
        grid_size=128,
        aspect_ratio=3,
        max_particles=np.inf,
        dt=0.1,
        rho=1.0,
        U_inf=1.0,
        data_directory='cylinder_flow_data'
    )
    sim.run(
        LENGTH=3.0,
        HEIGHT=1.0,
        final_time=500.0,
        snapshot_interval=0.5,
        save_anim='cylinder_advection.mp4',
        save_forces='forces.h5',
        save_fields='flow_fields.h5'
    )
