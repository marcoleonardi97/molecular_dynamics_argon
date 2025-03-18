import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import imageio
from mpl_toolkits.mplot3d import Axes3D

class Atom(object):
    def __init__(self, element="argon", temperature=0.1, position=None, velocity=None):
        """
        element (str): Chemical element (default: "argon")
        temperature (float): Temperature in reduced units (only used in init)
        position (numpy.array): Initial position in reduced units
        velocity (numpy.array): Initial velocity in reduced units
        """
        kb = 1 #1.38064852e-23
        # Element properties: epsilon in J (converted to reduced units in simulation)
        elements = {"argon": [119.8]}
        self.epsilon = elements[element][0] * kb
        
        self.temperature = temperature
        self.element = element

        if position is not None:
            self.position = position
        else:
            self.position = np.zeros(3)
        if velocity is not None:
            self.velocity = velocity
        else:
            self.velocity = np.random.normal(0, (self.temperature) ** 0.5, size=3) # * self.change_temp

        self.mass = 1  # Mass in reduced units
        self.change_vel = np.zeros(3)  # Velocity change in current step
        self.change_pos = np.zeros(3)  # Position change in current step
        self.pot_energy = 0  
        self.kin_energy = self.mass * np.linalg.norm(self.velocity) ** 2 / 2 

    def _dimensionless_interaction(self, other, dt, box_size, verbose=False):
        """
        Calculates the interaction between two atoms using Lennard-Jones potential.
        The calculations are done in dimensionless units.
        -----------
        other (Atom): Another atom to interact with
        dt (float): Timestep in reduced units
        box_size (float): Simulation box size in reduced units
        verbose (bool): Whether to print diagnostic information
        """
        # Calculate distance vector with periodic boundary conditions
        r = self.position - other.position
        mag_r = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        
        # Apply minimum image convention for periodic boundary conditions
        if mag_r > box_size:
            r = np.sign(r) * (abs(r) - 2*box_size)
            mag_r = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        
        # Calculate gradient of the dimensionless Lennard-Jones potential
        # U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
        # ∇U(r) = 24ε[-(σ/r)^13 + (σ/r)^7] * r/r
        GradU_tilde = 24*(-2*mag_r**-13 + mag_r**-7) 
        acc_tilde = -GradU_tilde * r/mag_r  # Acceleration in reduced units

        # Calculate potential energy using the Lennard-Jones potential
        self.pot_energy = 4*(mag_r**-12 - mag_r**-6)
        
        # Update velocity and position changes
        self.change_vel += acc_tilde * dt #* self.change_temp
        self.change_pos += (self.change_vel + self.velocity) * dt
        
        if verbose:
            print('r is:', r)
            print('mag_r is:', mag_r)
            print('gradient of U is:', GradU_tilde)
            print('acceleration is:', acc_tilde)
            print('change in position is:', self.change_pos)
            print('end position is:', self.position + self.change_pos)

class Simulation():
    def __init__(self, density=1.2, temperature=0.5, num_atoms=108, element="argon", boxsize=None):
        """
        density (float): Density in reduced units
        temperature (float): Temperature in reduced units
        num_atoms (int): Number of atoms (default: 108, suitable for 3x3x3 FCC lattice)
        element (str): Chemical element (default: "argon")
        boxsize (float): Size of simulation box (calculated from density if None)
        """
        self.num_atoms = num_atoms
        self.temperature = temperature
        self.density = density
        self.element = element
        self.time = 0
        self.frame = 0
        self.potential_energy = []
        self.kinetic_energy = []
        self.total_energy = []
        self.temperature_history = []
        self.density_history = []
        
        # Calculate box size based on density if not specified
        # For FCC lattice, there are 4 atoms per unit cell
        self.box_size = (num_atoms / self.density) ** (1/3) if boxsize is None else boxsize
        
        # Initialize positions on FCC lattice
        self.positions = self._initialize_fcc_lattice(self.density)
        self.velocities = self._initialize_velocities()
        self.atoms = [Atom(element=element, temperature=temperature, 
                          position=self.positions[i], 
                          velocity=self.velocities[i]) for i in range(num_atoms)]
        
        # Equilibrate velocities to target 
        self._equilibrate_velocities(num_steps=10)
    
    def _initialize_fcc_lattice(self, density):
        """
        Initialize positions on a face-centered cubic (FCC) lattice.
        
        Parameters:
        -----------
        density (float): Density in reduced units
        
        Returns:
        --------
        numpy.array: Array of positions
        """
        atoms_per_cell = 4  # FCC lattice has 4 atoms per unit cell
        num_cells = int(round((self.num_atoms / atoms_per_cell) ** (1/3)))
        
        # Calculate lattice spacing based on density
        lattice_spacing = (atoms_per_cell / density) ** (1/3)
        
        # FCC lattice basis vectors
        base_vectors = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        
        positions = []
        for x in range(num_cells):
            for y in range(num_cells):
                for z in range(num_cells):
                    for vec in base_vectors:
                        positions.append([(x + vec[0]) * lattice_spacing,
                                          (y + vec[1]) * lattice_spacing,
                                          (z + vec[2]) * lattice_spacing])
        return np.array(positions[:self.num_atoms])
    
    def _initialize_velocities(self):
        """
        Initialize velocities with Maxwell-Boltzmann distribution.
        
        Returns:
        --------
        numpy.array: Array of velocities
        """
        kb = 1.38064852e-23  # Boltzmann constant
        mass = 1  # Mass in reduced units
        
        # Standard deviation for Maxwell-Boltzmann distribution
        # In reduced units: p(v) ~ exp(-v^2/2kT)
        stddev = np.sqrt(self.temperature)
        
        # Generate random velocities
        velocities = np.random.normal(0, stddev, (self.num_atoms, 3))
        
        # Remove center of mass motion
        velocities -= np.mean(velocities, axis=0)
        
        return velocities
        
    def _equilibrate_velocities(self, num_steps=10):
        """
        Equilibrate velocities to target temperature through repeated rescaling.
        
        Parameters:
        -----------
        num_steps (int): Number of rescaling steps
        """
        for _ in range(num_steps):
            # Calculate current kinetic energy
            current_kinetic_energy = 0.0
            for atom in self.atoms:
                current_kinetic_energy += 0.5 * np.sum(atom.velocity**2)
            
            # Calculate target kinetic energy using equipartition theorem
            # For N particles in 3D: E_kin = 3N/2 * kT
            # Subtracting 3 degrees of freedom for conserved momentum
            target_kinetic_energy = (3 * self.num_atoms - 3) * self.temperature / 2
            
            # Calculate scaling factor
            lambda_factor = np.sqrt(target_kinetic_energy / current_kinetic_energy)
            
            # Rescale velocities
            for atom in self.atoms:
                atom.velocity *= lambda_factor
                
            # Verify the temperature
            actual_temp = self.get_current_temperature()
            if abs(actual_temp - self.temperature) / self.temperature < 0.01:
                break
                
    def _update_positions(self):
        """
        Update positions and velocities of all atoms and recalculate energies.
        """
        self.positions = []
        kinetic_energy = 0
        potential_energy = 0
        
        for atom in self.atoms:
            # Update position and velocity
            atom.position += atom.change_pos
            atom.velocity += atom.change_vel

            # Apply periodic boundary conditions
            l = self.box_size
            wrapped_position = ((atom.position + l) % (2*l)) - l
            for i in range(3):
                if wrapped_position[i] < -l:
                    wrapped_position[i] += 2*l
            atom.position = wrapped_position

            # Update energies
            kinetic_energy += np.sum(atom.velocity**2) / 2
            potential_energy += atom.pot_energy

        # Store energies and temperature for monitoring
        self.kinetic_energy.append(kinetic_energy)
        self.potential_energy.append(potential_energy)
        self.total_energy.append(kinetic_energy + potential_energy)
        self.temperature_history.append(self.get_current_temperature())
        self.density_history.append(self.density)
        
        # Update positions list
        self.positions = np.array([a.position for a in self.atoms])
    
    def is_equilibrated(self, window=10, tolerance=0.01):
        """
        Check if the system is equilibrated by monitoring total energy stability.
        
        Parameters:
        -----------
        window (int): Number of recent steps to consider
        tolerance (float): Maximum allowed relative fluctuation
        
        Returns:
        --------
        bool: True if equilibrated, False otherwise
        """
        if len(self.total_energy) < window:
            return False
        
        recent_energies = self.total_energy[-window:]
        mean_energy = np.mean(recent_energies)
        relative_fluctuation = np.std(recent_energies) / abs(mean_energy)
        
        return relative_fluctuation < tolerance

    def get_current_temperature(self):
        """
        Calculate the current temperature from the kinetic energy.
        
        Returns:
        --------
        float: Current temperature in reduced units
        """
        # Calculate kinetic energy
        kinetic_energy = 0.0
        for atom in self.atoms:
            kinetic_energy += 0.5 * np.sum(atom.velocity**2)
        
        # Temperature = 2/3 * kinetic energy / N
        # Adjusting for 3 conserved momentum degrees of freedom
        temperature = 2 * kinetic_energy / (3 * self.num_atoms - 3)
        
        return temperature

    def verify_temperature_scaling(self):
        """
        Verify that temperature scaling is working correctly.
        Only used for debugging.
        
        Returns:
        --------
        tuple: (target_temperatures, actual_temperatures)
        """
        # Store original temperature
        original_temp = self.temperature
        
        # Test a range of temperatures
        target_temps = np.linspace(0.2, 2.0, 5)
        actual_temps = []
        
        for temp in target_temps:
            self.temperature = temp
            self._equilibrate_velocities(num_steps=20)
            actual_temps.append(self.get_current_temperature())
            
        # Plot results
        plt.figure(figsize=(8, 6))
        plt.plot(target_temps, target_temps, 'k--', label='Perfect scaling')
        plt.plot(target_temps, actual_temps, 'ro-', label='Actual temperatures')
        plt.xlabel('Target Temperature (ε/k)')
        plt.ylabel('Actual Temperature (ε/k)')
        plt.title('Temperature Scaling Verification')
        plt.legend()
        plt.grid(True)
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/temperature_scaling.png")
        plt.show()
        
        # Restore original temperature
        self.temperature = original_temp
        self._equilibrate_velocities()
        
        return target_temps, actual_temps

    def add_atom(self, atom):
        """
        Add an atom manually to the simulation. It must have a specified position and velocity.
        
        Parameters:
        -----------
        atom (Atom): Atom object to add.
        """
        self.num_atoms += 1
        self.atoms.append(atom)
        self.positions = np.array([a.position for a in self.atoms])


    def evolve_system(self, dt, t_end, change_phase = "",
                      plot=False, plot3d=False, plot_energy=False, verbose=False):
        """
        Evolve the simulation over time.
        
        Parameters:
        -----------
        dt (float): Simulation timestep in reduced units
        t_end (float): Simulation end time in reduced units
        change_temperature (float): Target temperature if changing during simulation
        plot (bool): Whether to save each frame as 2D plot
        plot3d (bool): Whether to save each frame as 3D plot
        plot_energy (bool): Whether to plot energy evolution at the end
        verbose (bool): Whether to print diagnostic information
        """

        phases = {"solid": [0.5, 0.3], "liquid": [1.0, 0.8], "gas": [3.0, 0.3]}
        
        if change_phase in phases:
            delta_t, delta_rho = phases[change_phase][0], phases[change_phase][1]
        elif change_phase == "":
            delta_t, delta_rho = self.temperature, self.density
        else:
            raise ValueError("change_phase must be one of 'solid', 'liquid', or 'gas'")

        temperature_gradient = (delta_t - self.temperature) / (t_end/dt)
        density_gradient = (delta_rho - self.density) / (t_end/dt)
        steps = 0
        
        while self.time < t_end:
            potential_energy = 0
            
            for atom in self.atoms:
                # Reset changes for this step
                atom.change_vel = np.zeros(3)
                atom.change_pos = np.zeros(3)
                
                # Calculate interactions with all other atoms
                for other_atom in self.atoms:
                    if not atom == other_atom:
                        atom._dimensionless_interaction(other_atom, dt, self.box_size, verbose=verbose)
                        potential_energy += atom.pot_energy / 2  # Divide by 2 to avoid double counting

            self._equilibrate_velocities()
            
            # Update simulation temperature if needed
            if change_phase != "":
                self.temperature += temperature_gradient 
                self.density += density_gradient
                
            # Update positions and energies
            self._update_positions()
            
            self.time += dt
            self.frame += 1
            steps += 1
            
            if steps % 10 == 0:
                print(f"System evolved to {self.time:.3f} s, Kin. Temperature: {self.get_current_temperature():.2e}")
                if self.is_equilibrated():
                    print("System is equilibrated.")
            
            if verbose:
                print('new positions are:', self.positions)
            
            # Save plots if requested
            if plot:
                self.plot_system(save=True)
            if plot3d:
                self.plot_system_3d(save=True)
        
        # Plot energy evolution if requested
        if plot_energy:
            self.plot_energy(dt, t_end)

        print("Simulation completed.")

    def compute_pressure(self, n_samples=10):
        """
        Calculate the pressure of the system with error estimation.
        
        Parameters:
        -----------
        n_samples (int): Number of samples to collect for error estimation
        
        Returns:
        --------
        tuple: (pressure, error) in reduced units
        """
        # Initialize arrays to store pressure samples
        pressure_samples = np.zeros(n_samples)
        
        # Collect pressure samples over several time steps
        for _ in range(n_samples):
            # Kinetic contribution from ideal gas law: P_kin = ρkT
            kinetic_contribution = self.num_atoms * self.get_current_temperature() / (2 * self.box_size)**3
            
            # Virial contribution from forces
            virial_contribution = 0
            for atom_i in range(self.num_atoms):
                for atom_j in range(atom_i + 1, self.num_atoms):
                    r_ij = self.atoms[atom_i].position - self.atoms[atom_j].position
                    
                    # Apply minimum image convention
                    L = 2 * self.box_size
                    for dim in range(3):
                        if abs(r_ij[dim]) > L/2:
                            r_ij[dim] = abs(r_ij[dim]) - L
                    
                    r = np.linalg.norm(r_ij)
                    
                    # Skip if atoms are too close (avoid division by zero)
                    if r < 0.1:
                        continue
                    
                    # Force magnitude from LJ potential: F = -dU/dr
                    # For LJ: F = 24ε[(2σ^12/r^13) - (σ^6/r^7)]
                    force_magnitude = 24 * (2 * r**(-13) - r**(-7))
                    
                    # Virial contribution: -r·F
                    virial_contribution += force_magnitude * r
            
            # Total pressure: P = P_kin + P_vir
            # Factor of 1/3V is to convert to pressure units
            pressure_samples[i] = kinetic_contribution + virial_contribution / (3 * (2 * self.box_size)**3)
        
        # Evolve system slightly to get next sample
        self.evolve_system(dt=0.001, t_end=0.001)
    
        # Calculate mean and standard error
        pressure_mean = np.mean(pressure_samples)
        pressure_error = np.std(pressure_samples) / np.sqrt(n_samples)
        
        return pressure_mean, pressure_error

    def get_pressure(self):
        """
        Wrapper function to compute and return the system pressure.
        
        Returns:
        --------
        tuple: (pressure, error) in reduced units
        """
        return self.compute_pressure()

    def plot_energy(self, dt, time_tot):
        """
        Plot energy evolution over time.
        
        Parameters:
        -----------
        dt (float): Timestep used in simulation
        time_tot (float): Total simulation time
        """
        pot_en = self.potential_energy
        kin_en = self.kinetic_energy
        tot_en = self.total_energy
        temps = self.temperature_history
        
        time = np.linspace(0, time_tot, len(tot_en))

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Energy plot
        ax1.plot(time, pot_en, label='Potential energy')
        ax1.plot(time, kin_en, label='Kinetic energy')
        ax1.plot(time, tot_en, label='Total energy')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Evolution')
        ax1.legend()
        
        # Temperature plot
        ax2.plot(time, temps, label='Temperature', color='red')
        ax2.axhline(y=self.temperature, color='black', linestyle='--', label='Target temperature')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Temperature')
        ax2.set_title('Temperature Evolution')
        ax2.legend()
        
        # Add system information
        plt.figtext(0.7, 0.5, f"Target T = {self.temperature:.2f}")
        plt.figtext(0.7, 0.48, f"Final T = {temps[-1]:.2f}")
        plt.figtext(0.7, 0.46, f"N atoms = {self.num_atoms}")
        plt.figtext(0.7, 0.44, f"Box size = {self.box_size:.2f} σ")
        
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/energy_temperature.png")
        plt.show()
        
    def calculate_radial_distribution(self, bins=50, n_evals = 10):
        """
        Calculate the radial distribution function (RDF).
        
        Parameters:
        -----------
        bins (int): Number of distance bins for the histogram
        
        Returns:
        --------
        tuple: (r, g(r)) arrays for the RDF
        """
        L = 2 * self.box_size
        dr = L / (2 * bins)
        r_max = L / 2
        
        # Create distance bins
        r_bins = np.linspace(0, r_max, bins)
        hist = np.zeros(bins)
        
        # Calculate all pairwise distances
        for i in range(self.num_atoms):
            for j in range(i+1, self.num_atoms):
                r_ij = self.atoms[i].position - self.atoms[j].position
                
                # Apply minimum image convention
                for dim in range(3):
                    if abs(r_ij[dim]) > L/2:
                        r_ij[dim] = abs(r_ij[dim]) - L
                
                r = np.linalg.norm(r_ij)
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < bins:
                        hist[bin_idx] += 2  # Count each pair twice
        
        # Calculate g(r)
        density = self.num_atoms / (L**3)
        vol_factor = 4 * np.pi * r_bins**2 * dr
        g_r = hist / (self.num_atoms * density * vol_factor)
        
        return r_bins, g_r
    
    def plot_radial_distribution(self):
        """
        Plot the radial distribution function.
        """
        r, g_r = self.calculate_radial_distribution()
        
        plt.figure()
        plt.plot(r, g_r)
        plt.xlabel('r (σ)')
        plt.ylabel('g(r)')
        plt.title('Radial Distribution Function')
        plt.grid(True)
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/rdf.png")
        plt.show()

    def get_phase_diagram(self, temperatures, densities, n_samples=5):
        """
        Generate a phase diagram by calculating pressure at different temperatures and densities.
        
        Parameters:
        -----------
        temperatures (list): List of temperatures to sample
        densities (list): List of densities to sample
        n_samples (int): Number of samples for error estimation
        
        Returns:
        --------
        tuple: (P, T, ρ, ΔP) arrays for pressure, temperature, density, and pressure error
        """
        # Initialize arrays to store results
        n_temps = len(temperatures)
        n_dens = len(densities)
        P = np.zeros((n_temps, n_dens))
        P_err = np.zeros((n_temps, n_dens))
        
        # Store original simulation parameters
        original_temp = self.temperature
        original_density = self.density
        
        # Sample phase space
        for i, temp in enumerate(temperatures):
            for j, dens in enumerate(densities):
                print(f"Sampling T={temp:.2f}, ρ={dens:.2f}")
                
                # Set new temperature and density
                self.temperature = temp
                self.density = dens
                
                # Update box size for new density
                self.box_size = (self.num_atoms / self.density) ** (1/3)
                
                # Equilibrate system at new conditions
                self.evolve_system(dt=0.005, t_end=0.5)
                
                # Calculate pressure with error
                P[i, j], P_err[i, j] = self.compute_pressure(n_samples)
                
        # Restore original parameters
        self.temperature = original_temp
        self.density = original_density
        self.box_size = (self.num_atoms / self.density) ** (1/3)
        
        return P, temperatures, densities, P_err

    def plot_triple_point(self):
        """
        Plot the phase diagram with emphasis on the triple point region.
        """
        # Define temperature and density ranges
        temperatures = np.linspace(0.3, 1.5, 5)  # Adjust for better resolution
        densities = np.linspace(0.1, 1.0, 5)     # Adjust for better resolution
        
        # Calculate pressure at each point
        P, T, rho, P_err = self.get_phase_diagram(temperatures, densities)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D surface
        T_mesh, rho_mesh = np.meshgrid(T, rho)
        
        # Transpose P to match meshgrid dimensions
        P_mesh = P.T
        
        # Create 3D surface
        surf = ax.plot_surface(T_mesh, rho_mesh, P_mesh, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Pressure (ε/σ³)')
        
        # Add labels and title
        ax.set_xlabel('Temperature (ε/k)')
        ax.set_ylabel('Density (σ⁻³)')
        ax.set_zlabel('Pressure (ε/σ³)')
        ax.set_title('Phase Diagram with Triple Point')
        
        # Add error bars at each point
        for i in range(len(T)):
            for j in range(len(rho)):
                ax.plot([T[i], T[i]], [rho[j], rho[j]], 
                        [P[i, j] - P_err[i, j], P[i, j] + P_err[i, j]], 
                        color='k', alpha=0.5)
        
        # Save the plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/phase_diagram.png")
        plt.show()
        
        # Look for triple point
        self.analyze_triple_point(P, T, rho)

    def analyze_triple_point(self, P, T, rho):
        """
        Analyze the phase diagram to identify the triple point.
        
        Parameters:
        -----------
        P (numpy.array): 2D array of pressure values
        T (numpy.array): Array of temperature values
        rho (numpy.array): Array of density values
        """
        # Calculate the Gibbs free energy (approximate)
        # G = U + PV - TS
        # For a simple approximation, we'll look for discontinuities in P(rho)
        
        # Plot P vs rho for each temperature
        plt.figure(figsize=(10, 6))
        
        for i, temp in enumerate(T):
            plt.plot(rho, P[i, :], 'o-', label=f'T = {temp:.2f}')
        
        plt.xlabel('Density (σ⁻³)')
        plt.ylabel('Pressure (ε/σ³)')
        plt.title('Pressure vs Density at Different Temperatures')
        plt.legend()
        plt.grid(True)
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/pressure_vs_density.png")
        plt.show()

    def plot_system(self, save=False, show=False):
        """
        Plot the system in 2D.
        
        Parameters:
        -----------
        save (bool): Whether to save the plot as PNG
        show (bool): Whether to show the plot
        """
        x = [i[0] for i in self.positions]
        y = [i[1] for i in self.positions]
        
        plt.figure()
        plt.scatter(x, y, c=[i for i in range(len(self.atoms))])
        plt.xlim(-self.box_size, self.box_size)
        plt.ylim(-self.box_size, self.box_size)
        plt.xlabel(r"$\sigma [3.4 \times 10^{-10} m]$")
        plt.ylabel(r"$\sigma [3.4 \times 10^{-10} m]$")

        tx, ty = self.box_size - 5, self.box_size - 1
        
        if self.temperature > 1000 or self.temperature < 0.001:
            plt.text(tx, ty, f"T: {self.temperature:.2e}")
        else:
            plt.text(tx, ty, f"T: {self.temperature:.2f}")

        plt.title(f"System at time {self.time:.3f} s")
        if save:
            plt.savefig(f"{self.frame}.png")
            plt.close()
        if show:
            plt.show()

    def plot_system_3d(self, save=False, show=False):
        """
        Plot the system in 3D.
        
        Parameters:
        -----------
        save (bool): Whether to save the plot as PNG
        show (bool): Whether to show the plot
        """
        x = [i[0] for i in self.positions]
        y = [i[1] for i in self.positions]
        z = [i[2] for i in self.positions]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        ax.set_xlim(-self.box_size, self.box_size)
        ax.set_ylim(-self.box_size, self.box_size)
        ax.set_zlim(-self.box_size, self.box_size)
    
        ax.scatter(x, y, z, c=[i for i in range(len(self.atoms))], marker='o')
        ax.set_xlabel(r"$\sigma [3.4 \times 10^{-10} m]$")
        ax.set_ylabel(r"$\sigma [3.4 \times 10^{-10} m]$")
        ax.set_zlabel(r"$\sigma [3.4 \times 10^{-10} m]$")

        tx, ty, tz = self.box_size - 5, self.box_size - 1, self.box_size
        
        if self.temperature > 1000 or self.temperature < 0.001:
            ax.text(tx, ty, tz, f"T: {self.temperature:.2e}")
        else:
            ax.text(tx, ty, tz, f"T: {self.temperature:.2f}")

        plt.title(f"System at time {self.time:.3f} s")
        if show:
            plt.show()
        if save:
            plt.savefig(f"{self.frame}.png")
            plt.close()

    def animate_system(self, name="molecular_dynamics"):
        """
        Create an animation from saved frames.
        
        Parameters:
        -----------
        name (str): Name of the output GIF file
        """
        files = glob.glob("*.png")
        n = []
        for i in files:
            n.append(int(i[:-4]))
        sorted_files = [i for _, i in sorted(zip(n, files))]

        with imageio.get_writer(f"{name}.gif", mode='I', duration=0.1) as writer:
            for frame in sorted_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        
        # Clean up the directory 
        for frame in sorted_files:
            os.remove(frame)
        
        print(f"Animation saved as {name}.gif")
        
    def __str__(self):
        """
        String representation of the simulation.
        """
        current_temp = self.get_current_temperature() if self.atoms else 0
        
        return (f"Simulation of {len(self.atoms)} {self.atoms[0].element} atoms.\n"
                f"Target temperature: {self.temperature:.3f}\n"
                f"Current temperature: {current_temp:.3f}\n"
                f"Box size: {self.box_size:.3f} σ\n"
                f"Time: {self.time:.3f} sim time\n"
                f"Equilibrated: {self.is_equilibrated() if len(self.total_energy) > 10 else 'Not enough data'}")


