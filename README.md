### Quasi-Geostrophisches Modell geschrieben in der C-Programmiersprache (Model of Quasi-Geotrophy written in C programming language)

* Desktop app which creates a Quasi-Geotrophy (QG) mode ans is written in C-language.
* Stores model configuration parameters like number of layers, grid size, time step (dt), beta parameter, Coriolis parameter (f0), gravity (g), layer thickness (H), geometry, solver type, boundary conditions, wind forcing, and topography scale.
* Holds simulation data, including arrays for streamfunction (psi), potential vorticity (q), previous PV (q_prev), velocities (u, v), topography, and FFTW arrays for spectral computations.
* Allocates memory for model arrays based on grid size (default 64x64) and number of layers (2–4).
* Initializes the streamfunction with a zonal jet (sinusoidal pattern) plus perturbations to simulate realistic flow.
* Sets up topography as a Gaussian ridge and configures FFTW plans for fast Fourier transforms used in PV inversion.
* Calculates velocities (u, v) from the streamfunction using finite differences.
* Transforms Potential Vorticity or PV (q) to spectral space using FFT and inverts the PV to obtain the streamfunction (ψ) using a Helmholtz-like equation, accounting for stretching due to stratification.
* Supports 2-layer or single-layer inversion, adjusting for the stretching term (f0^2/(g*H)).
* Applies inverse FFT to return the streamfunction to physical space.
* Calculates velocities (u, v) from the streamfunction using finite differences.
* Computes PV tendencies (dq_dt) including:
  * Advection: Nonlinear term -u * dq/dx - v * dq/dy.
  * Beta effect: Accounts for Coriolis variation (-β * v).
  * Wind forcing: Double-gyre wind stress applied to the top layer.
  * Friction: Linear damping in the bottom layer.
  * Topography: Interaction with bottom topography in the bottom layer.
  * Hyperdiffusion: High-order diffusion to damp small-scale noise.
  * Sponge layer: For open boundaries, damps PV near domain edges.
* Utilizes Runge-Kutta 4, Leapfrog, and Semi-Implicit time stepping solvers.
* Computes the average kinetic energy per grid point based on velocities and estimates the speed of Rossby waves using the beta parameter and a typical wavenumber.
* The canvas displays the selected field using a color map, with values normalized to [-1, 1] for visualization and updates energy and Rossby speed in real-time.
* Study ocean gyres, Rossby waves, and topographic effects; Compare numerical solvers for stability and accuracy; Explore the impact of boundary conditions or geometry on flow patterns.

---

![](https://github.com/KMORaza/Model_of_Quasi-Geotrophy/blob/main/Model%20of%20Quasi%20Geotrophy/Improved/screen.png)
