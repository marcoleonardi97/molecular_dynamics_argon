import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import imageio
import os
from matplotlib.colors import LinearSegmentedColormap
import cmocean
import numba


def create_custom_cmap(name='custom_cmap'):
    """
    Create a colormap that transitions: White -> Blue -> Black -> Red -> White.
    This is useful for this simulation because now opposing spins have different colors.
    
    Args:
        name (str): Name of the colormap.
        
    Returns:
        LinearSegmentedColormap: The created colormap.
    """
    # Define the colors for the colormap (White, Blue, Black, Red)
    colors = ['white', 'blue', 'black', 'red', 'white']
    
    # Create the colormap from the list of colors
    return LinearSegmentedColormap.from_list(name, colors)

# Create the custom colormap
custom_cmap = create_custom_cmap()


class Spin():
    def __init__(self, angle, row, col):
        self.angle = angle
        self._row = row
        self._col = col

    @property
    def row(self):
        return self._row

    @property
    def col(self):
        return self._col

    def rotate(self, new_angle):
        self.angle = new_angle
    
    def __str__(self):
        return f"Spin with angle {self.angle}, at row: {self.row}, col: {self.col}"


@numba.njit
def compute_energy_vectorized(angles):
    """Vectorized energy calculation with Numba."""
    n = angles.shape[0]
    energy = 0.0
    energy_squared = 0.0
    
    # Calculate interactions with right neighbors (without using axis parameter)
    for i in range(n):
        for j in range(n):
            # Right neighbor
            j_right = (j + 1) % n
            energy -= np.cos(angles[i, j] - angles[i, j_right])
            energy_squared -= np.cos(angles[i, j] - angles[i, j_right])**2
            
            # Down neighbor
            i_down = (i + 1) % n  
            energy -= np.cos(angles[i, j] - angles[i_down, j])
            energy_squared -= np.cos(angles[i, j] - angles[i_down, j])**2

            
    return energy, energy_squared
    
    


class XYModel():
    def __init__(self, n=3, temperature=1):
        self.n = n

        self.spins, self.angles = self.generate_state()  #self.spins is a 1D list of length N^2
                                                         #self.angles is a 2D list of shape NxN

        self.current_state = self.initialize_grid()      #2d list of spins

        self.magnetization, self.magnetization_squared = self.calculate_magnetization()
        self.energy, self.energy_squared = self.calculate_total_energy()

        self.magnetization_list = [self.magnetization]
        self.magnetization_squared_list = [self.magnetization_squared]
        self.energy_list = [self.energy]
        self.energy_squared_list = [self.energy_squared]

        self.temp = temperature
        self.frame = 0

    def initialize_grid(self):
        grid = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for spin in self.spins:
            grid[spin.row][spin.col] = spin
            
        return grid

    def generate_state(self):
        random_state = []
        random_angles = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                angle = np.random.uniform(-np.pi,np.pi)
                #angle = np.pi/2  #to test
                random_state.append(Spin(angle, i, j))
                random_angles[i][j] = angle

        return np.array(random_state), np.array(random_angles)


    def update_state(self, step, sweep):

        row, col = np.random.randint(0, self.n, 2)      #choose a random position
        random_angle = np.random.uniform(-np.pi, np.pi) #choose a random new angle
        #random_state = copy.deepcopy(self.current_state)
        #random_state[row][col].rotate(random_angle)     #make a new state with one new angle
        old_spin = self.current_state[row][col]
        #new_spin = random_state[row][col]

        old_E = 0
        new_E = 0

        neighbors = self.find_neighbors(self.current_state[row][col])
        for neighbor in neighbors:
            old_E += - np.sum(np.cos(old_spin.angle - neighbor.angle))
            new_E += - np.sum(np.cos(random_angle - neighbor.angle))
        
        delta_E = new_E - old_E
        delta_M = random_angle - old_spin.angle

        if delta_E < 0:  #update if the new energy is lower
            self.current_state[row][col].rotate(random_angle)
            self.angles[row][col] = self.current_state[row][col].angle

            # if we want to do this only after every sweep
            if step % sweep == 0:
                self.energy += delta_E
                self.energy_squared -= old_E**2
                self.energy_squared += new_E**2

                self.magnetization += delta_M
                self.magnetization_squared -= old_spin.angle**2
                self.magnetization_squared += random_angle**2

        elif np.exp(-1/(self.temp) * delta_E) > np.random.uniform(0, 1):
            # Update the system anyways with probability exp(-1/(k_b * self.temp) * delta_E)
            self.current_state[row][col].rotate(random_angle)
            self.angles[row][col] = self.current_state[row][col].angle
            
            if step % sweep == 0:
                self.energy += delta_E
                self.energy_squared -= old_E**2
                self.energy_squared += new_E**2

                self.magnetization += delta_M
                self.magnetization_squared -= old_spin.angle**2
                self.magnetization_squared += random_angle**2


        # Update lists
        self.energy_list.append(self.energy)
        self.energy_squared_list.append(self.energy_squared)
        self.magnetization_list.append(abs(self.magnetization))
        self.magnetization_squared_list.append(self.magnetization_squared)

        self.frame += 1

    def find_neighbors(self, spin):

        neighbour1 = self.current_state[(spin.row-1)%self.n] [spin.col]
        neighbour2 = self.current_state[(spin.row+1)%self.n] [spin.col]
        neighbour3 = self.current_state[spin.row] [(spin.col-1)%self.n]
        neighbour4 = self.current_state[spin.row] [(spin.col+1)%self.n]

        neighbors = [neighbour1, neighbour2, neighbour3, neighbour4]

        return neighbors

    def calculate_total_energy(self):
        return compute_energy_vectorized(self.angles)

    def calculate_magnetization(self):
        M = 0
        M_squared = 0
        for _ in range(len(self.current_state)):
            for spin in self.current_state[_]:
                M += (spin.angle)  
                M_squared += (spin.angle)**2 
        return abs(M), M_squared

    def run(self, sweeps, plot=False, plot_energy_magnetization=False,
    plot_2d_graph=False, plot_vortex = False, **kwargs):
        
        steps = sweeps * self.n**2
        self.frame = 0
        for i in range(steps):

            if plot == True and i % self.n**2 == 0:
                self.plot(show=False, save=True, **kwargs)

            if plot_vortex == True and i % self.n**2 == 0:
                self.plot_with_vortices(save=True)

            self.update_state(i, sweeps)
            
            if i % self.n**2 == 0:
                print(f"Sweep number {i/self.n**2} completed")
        
        if plot_energy_magnetization:
            self.plot_energy_magnetization()

        if plot_2d_graph:
            self.plot(show=True, save=True, **kwargs)

    def plot(self, show = False, save=False, ptype = "arrows"):

        side = np.linspace(0, 1, self.n)
        X, Y = np.meshgrid(side, side)
        u = np.cos(self.angles)
        v = np.sin(self.angles)
        #cmap = plt.cm.get_cmap('coolwarm', 4)
        #cmap = plt.cm.get_cmap('twilight_shifted', 6)  #cyclic colormap because angle(-pi) = angle(pi)

        if ptype == "arrows":
            plt.quiver(X, Y, u, v, self.angles, cmap=custom_cmap,  
            pivot='tip', headlength = 10, headwidth = 8, headaxislength = 6, linewidth=20)

        elif ptype == "cmap":
            plt.imshow(self.angles, cmap=custom_cmap) 

        else:
            print(f"{ptype} is not a valid plot type. Defaulting to arrows.")
            plt.quiver(X, Y, u, v, self.angles, cmap=custom_cmap,  
            pivot='tip', headlength = 10, headwidth = 8, headaxislength = 6, linewidth=20)
            
        plt.clim(-np.pi, np.pi)
        clb=plt.colorbar(orientation="vertical")
        clb.set_label('Angle', rotation=90)
        clb.set_ticks([-np.pi,0, np.pi], labels=['$-\pi$', '0', '$\pi$'])

        plt.tick_params(left = False, right = False , labelleft = False, 
                        labelbottom = False, bottom = False) 
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.title(f'XY Model at temperature {round(self.temp, 2)}\nafter {self.frame/self.n**2} sweeps')
        plt.gca().set_facecolor('lightgrey')

        if show:
            plt.show()

        if save:
            plt.savefig(f"{self.frame//self.n**2}.png")
            plt.close()

    def corr_func(self):
        """
        Returns correlation function and the correlation time for the list of magnetizations
        """
        mean = np.mean(self.magnetization_list)
        data = self.magnetization_list - mean
        correlation = np.correlate(data, data, mode='full') #not totally sure about this one
        correlation = correlation[len(self.magnetization_list)-1:]
        t = np.arange(len(correlation))
        tau = np.where(correlation < 0)[0][0]
        return correlation/correlation[0], tau

    def calculate_susceptibility(self):
        """
        Calculate magnetic susceptibility per spin using fluctuations in magnetization.
        
        χ_M = β/N^2 * (<M^2> - <M>^2)
        
        Returns:
            float: Magnetic susceptibility per spin
        """
        # Skip initial equilibration period
        equil_steps = int(len(self.magnetization_list) * 0.2)  # Skip first 20% as equilibration
        
        # Use absolute magnetization as mentioned in the notes
        m_values = np.array(self.magnetization_list[equil_steps:])
        m_squared_values = m_values**2
        
        # Calculate average magnetization and average squared magnetization
        avg_m = np.mean(m_values)
        avg_m_squared = np.mean(m_squared_values)
        
        # Calculate susceptibility (β = 1/T)
        susceptibility = (1.0/self.temp) * (avg_m_squared - avg_m**2) / (self.n**2)
    
        return susceptibility

    def calculate_specific_heat(self):
        """
        Calculate specific heat per spin using fluctuations in energy.
        
        C = 1/(N^2 * k_B * T^2) * (<E^2> - <E>^2)
        
        Returns:
            float: Specific heat per spin
        """
        # Skip initial equilibration period
        equil_steps = int(len(self.energy_list) * 0.2)  # Skip first 20% as equilibration
        
        e_values = np.array(self.energy_list[equil_steps:]) / (self.n**2)  # Energy per spin
        e_squared_values = np.array(self.energy_squared_list[equil_steps:]) / (self.n**2) #Energy squared per spin
        
        # Calculate averages
        avg_e = np.mean(e_values)
        avg_e_squared = np.mean(e_squared_values)
        
        # Calculate specific heat (k_B = 1 in natural units)
        specific_heat = (1.0 / (self.temp**2)) * (avg_e_squared - avg_e**2)
        
        return specific_heat

    def calculate_correlation_time(self):
        """
        Calculate the autocorrelation time of the magnetization.
        
        Returns:
            float: Correlation time in Monte Carlo sweeps
        """
        # Skip initial equilibration period
        equil_steps = int(len(self.magnetization_list) * 0.2)
        
        # Get magnetization data after equilibration
        m_values = np.array(self.magnetization_list[equil_steps:])
        
        # Calculate the mean magnetization
        mean_m = np.mean(m_values)
        
        # Calculate the normalized fluctuations
        m_fluctuations = m_values - mean_m
        
        # Calculate the autocorrelation function using numpy's correlate
        # Note: 'same' mode returns the central part of the correlation
        acf = np.correlate(m_fluctuations, m_fluctuations, mode='full')
        
        # Normalize and take the second half (positive time lags)
        acf = acf[len(m_fluctuations)-1:] / (len(m_fluctuations) * np.var(m_values))
        
        # Find where autocorrelation becomes negative or close to zero
        # This indicates poor statistics
        cut_idx = 0
        for i, val in enumerate(acf):
            if val < 0 or val < 0.01:  # Cut when autocorrelation becomes negative or very small
                cut_idx = i
                break
        
        # If we didn't find a suitable cutoff, use a reasonable fraction of the data
        if cut_idx == 0:
            cut_idx = len(acf) // 4
        
        # Integrate the autocorrelation function up to the cutoff
        # This is just a sum since we have discrete time steps
        tau = np.sum(acf[:cut_idx])
        
        # Convert from steps to sweeps
        tau = tau / (self.n**2)
        
        return tau

    def blocking_analysis(self, quantity_list, block_size_multiplier=16):
        """
        Perform blocking analysis to estimate errors in derived quantities.
        
        Args:
            quantity_list (list): List of measurements
            block_size_multiplier (int): Block size as a multiple of correlation time
        
        Returns:
            tuple: (mean, standard_error)
        """
        # Skip equilibration period
        equil_steps = int(len(quantity_list) * 0.2)
        data = np.array(quantity_list[equil_steps:])
        
        # Get correlation time (in steps)
        tau_steps = self.calculate_correlation_time() * self.n**2
        
        # Calculate block size (use at least 16 times the correlation time)
        block_size = max(int(block_size_multiplier * tau_steps), 100)
        
        # Ensure we have enough data for at least 5 blocks
        if len(data) < 5 * block_size:
            block_size = len(data) // 5
        
        # Calculate number of blocks
        num_blocks = len(data) // block_size
        
        # If too few blocks, adjust block size
        if num_blocks < 5:
            num_blocks = 5
            block_size = len(data) // num_blocks
        
        # Create blocks and calculate the quantity for each block
        block_values = []
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size
            block_data = data[start_idx:end_idx]
            block_values.append(np.mean(block_data))
        
        # Calculate mean and standard error across blocks
        mean_value = np.mean(block_values)
        std_error = np.std(block_values, ddof=1) / np.sqrt(num_blocks)
        
        return mean_value, std_error

    def plot_energy_magnetization(self):

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'XY model at temperature {round(T,2)}\nafter {self.frame/self.n**2} sweeps')
        ax = axes[0]
        #plt.subplot(1,2,1)
        ax.scatter(range(len(self.magnetization_list)), np.array(self.magnetization_list)/self.n**2)
        ax.set_ylabel('m', rotation='horizontal', labelpad = 6)
        ax.set_xlabel('Update step')
        ax.set_title(f'Magnetization per spin')
        ax.set_ylim(0, np.pi)
        ax.set_yticks([0, np.pi/2, np.pi])
        ax.set_yticklabels(['0', '$\pi/2$', '$\pi$'])
 
        ax = axes[1]
        plt.subplot(1,2,2)
        ax.scatter(range(len(self.energy_list)), np.array(self.energy_list)/self.n**2)
        ax.set_ylabel('e', rotation='horizontal')
        ax.set_xlabel('Update step')
        ax.set_title(f'Energy per spin')
        #ax.set_ylim(-1250, 1250)
        plt.tight_layout()
        plt.show()

    def analyze_thermal_quantities(self, temperatures=np.arange(0.00001, 2.5, 0.1), num_sweeps=200):
        """
        Analyze thermal quantities across a range of temperatures.
        
        Args:
            temperatures (list): List of temperatures to analyze
        
        Returns:
            dict: Dictionary containing analysis results
        """
        results = {
            'temperature': [],
            'correlation_time': [],
            'magnetization': {'mean': [], 'error': []},
            'energy': {'mean': [], 'error': []},
            'susceptibility': {'mean': [], 'error': []},
            'specific_heat': {'mean': [], 'error': []}
        }
        
        M_history = [[] for _ in range(len(temperatures))]
        M_squared_history = [[] for _ in range(len(temperatures))]
        E_history = [[] for _ in range(len(temperatures))]
        E_squared_history = [[] for _ in range(len(temperatures))]

        for i, temp in enumerate(temperatures):
            print(f"Analyzing temperature T = {temp}")
            # Set temperature
            self.temp = temp
            
            # Reset the system
            self.spins, self.angles = self.generate_state()
            self.current_state = self.initialize_grid()
            self.magnetization, self.magnetization_squared = self.calculate_magnetization()
            self.energy, self.energy_squared = self.calculate_total_energy()
            
            # Reset tracking lists
            self.magnetization_list = [self.magnetization]
            self.magnetization_squared_list = [self.magnetization_squared]
            self.energy_list = [self.energy]
            self.energy_squared_list = [self.energy_squared]
            
            # Run simulation (longer near critical point)
            sweeps = 2*num_sweeps if (0.8 <= temp <= 1.0) else num_sweeps
            self.run(sweeps, plot=False, plot_2d_graph=True)

            print(len(self.magnetization_list))

            # Store results
            M_history[i]         = np.array(self.magnetization_list, dtype=object)/self.n**2
            M_squared_history[i] = np.array(self.magnetization_squared_list, dtype=object)/self.n**2
            E_history[i]         = np.array(self.energy_list, dtype=object)/self.n**2
            E_squared_history[i] = np.array(self.energy_squared_list, dtype=object)/self.n**2

            # Calculate correlation time
            tau = self.calculate_correlation_time()
            
            # Calculate magnetization per spin with error
            m_mean, m_error = self.blocking_analysis([abs(m)/(self.n**2) for m in self.magnetization_list])
            
            # Calculate energy per spin with error
            e_mean, e_error = self.blocking_analysis([e/(self.n**2) for e in self.energy_list])
            
            # Calculate susceptibility
            chi = self.calculate_susceptibility()
            
            # Calculate specific heat
            c = self.calculate_specific_heat()
            
            # Estimating errors for derived quantities using block analysis
            # We'll create sequences of susceptibility and specific heat values from blocks
            equil_steps = int(len(self.magnetization_list) * 0.2)
            tau_steps = tau * self.n**2
            block_size = max(int(16 * tau_steps), 100) 
            num_blocks = (len(self.magnetization_list) - equil_steps) // block_size
            
            chi_values = []
            c_values = []
            
            for i in range(num_blocks):
                start_idx = equil_steps + i * block_size
                end_idx = equil_steps + (i + 1) * block_size
                
                # Calculate susceptibility for this block
                m_block = np.array(self.magnetization_list[start_idx:end_idx])
                m_squared_block = m_block**2
                avg_m_block = np.mean(m_block)
                avg_m_squared_block = np.mean(m_squared_block)
                chi_block = (1.0/self.temp) * (avg_m_squared_block - avg_m_block**2) / (self.n**2)
                chi_values.append(chi_block)
                
                # Calculate specific heat for this block
                e_block = np.array(self.energy_list[start_idx:end_idx]) / (self.n**2)
                e_squared_block = e_block**2
                avg_e_block = np.mean(e_block)
                avg_e_squared_block = np.mean(e_squared_block)
                c_block = (1.0 / (self.temp**2)) * (avg_e_squared_block - avg_e_block**2)
                c_values.append(c_block)
            
            # Calculate errors from block values
            chi_error = np.std(chi_values, ddof=1) / np.sqrt(num_blocks) if num_blocks > 1 else 0
            c_error = np.std(c_values, ddof=1) / np.sqrt(num_blocks) if num_blocks > 1 else 0
            
            # Store results
            results['temperature'].append(temp)
            results['correlation_time'].append(tau)
            results['magnetization']['mean'].append(m_mean)
            results['magnetization']['error'].append(m_error)
            results['energy']['mean'].append(e_mean)
            results['energy']['error'].append(e_error)
            results['susceptibility']['mean'].append(chi)
            results['susceptibility']['error'].append(chi_error)
            results['specific_heat']['mean'].append(c)
            results['specific_heat']['error'].append(c_error)
            
            print(f"T={temp}, τ={tau:.2f}, m={m_mean:.4f}±{m_error:.4f}, e={e_mean:.4f}±{e_error:.4f}")
            print(f"χ={chi:.4f}±{chi_error:.4f}, C={c:.4f}±{c_error:.4f}")

        M_history = np.array(M_history, dtype=object)
        M_squared_history = np.array(M_squared_history, dtype=object)
        E_history = np.array(E_history, dtype=object)
        E_squared_history = np.array(E_squared_history, dtype=object)

        #np.savetxt("magnetization_short.txt", M_history)
        #np.savetxt("magnetization_squared_short.txt", M_squared_history)
        #np.savetxt("energy_short.txt", E_history)
        #np.savetxt("energy_squared_short.txt", E_squared_history)

        np.save(f"magnetization_N{self.n}.npy", M_history, allow_pickle=True)
        np.save(f"magnetization_squared_N{self.n}.npy", M_squared_history, allow_pickle=True)
        np.save(f"energy_N{self.n}.npy", E_history, allow_pickle=True)
        np.save(f"energy_squared_N{self.n}.npy", E_squared_history, allow_pickle=True)

        return results

    def plot_results(self, results):
        """
        Plot analysis results.
        
        Args:
            results (dict): Results from analyze_thermal_quantities
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot correlation time
        axs[0, 0].errorbar(results['temperature'], results['correlation_time'], 
                          fmt='o-', capsize=3)
        axs[0, 0].set_xlabel('Temperature')
        axs[0, 0].set_ylabel('Correlation time (τ)')
        axs[0, 0].set_title('Correlation Time vs Temperature')
        axs[0, 0].grid(True)
        
        # Plot magnetization
        axs[0, 1].errorbar(results['temperature'], results['magnetization']['mean'], 
                          yerr=results['magnetization']['error'], fmt='o-', capsize=3)
        axs[0, 1].set_xlabel('Temperature')
        axs[0, 1].set_ylabel('Magnetization per spin (|m|)')
        axs[0, 1].set_title('Magnetization vs Temperature')
        axs[0, 1].grid(True)
        
        # Plot susceptibility
        axs[1, 0].errorbar(results['temperature'], results['susceptibility']['mean'], 
                          yerr=results['susceptibility']['error'], fmt='o-', capsize=3)
        axs[1, 0].set_xlabel('Temperature')
        axs[1, 0].set_ylabel('Magnetic Susceptibility (χ)')
        axs[1, 0].set_title('Magnetic Susceptibility vs Temperature')
        axs[1, 0].grid(True)
        
        # Plot specific heat
        axs[1, 1].errorbar(results['temperature'], results['specific_heat']['mean'], 
                          yerr=results['specific_heat']['error'], fmt='o-', capsize=3)
        axs[1, 1].set_xlabel('Temperature')
        axs[1, 1].set_ylabel('Specific Heat (C)')
        axs[1, 1].set_title('Specific Heat vs Temperature')
        axs[1, 1].grid(True)
        
        # Add a vertical line at the expected critical temperature
        for ax in axs.flat:
            ax.axvline(x=0.881, color='r', linestyle='--', alpha=0.7, label='T_c = 0.881')
        
        plt.tight_layout()
        plt.savefig('xy_model_analysis.png', dpi=300)
        plt.show()


    def detect_vortices(self):
        """
        Detect vortices and anti-vortices in the XY model.
        
        Returns:
            tuple: (vortex_positions, antivortex_positions)
        """
        vortex_positions = []
        antivortex_positions = []
        
        # Check each plaquette (square of 4 neighboring spins)
        for i in range(self.n):
            for j in range(self.n):
                # Get the four spins around the plaquette
                s1 = self.angles[i, j]
                s2 = self.angles[i, (j+1)%self.n]
                s3 = self.angles[(i+1)%self.n, (j+1)%self.n]
                s4 = self.angles[(i+1)%self.n, j]
                
                # Calculate angle differences (taking care of periodic boundary)
                dtheta1 = (s2 - s1 + np.pi) % (2*np.pi) - np.pi
                dtheta2 = (s3 - s2 + np.pi) % (2*np.pi) - np.pi
                dtheta3 = (s4 - s3 + np.pi) % (2*np.pi) - np.pi
                dtheta4 = (s1 - s4 + np.pi) % (2*np.pi) - np.pi
                
                # Sum up the differences to find winding number
                winding = (dtheta1 + dtheta2 + dtheta3 + dtheta4) / (2*np.pi)
                winding = round(winding)  # Should be close to an integer
                
                # Record position if we found a vortex or anti-vortex
                if winding == 1:
                    vortex_positions.append((i+0.5, j+0.5))
                elif winding == -1:
                    antivortex_positions.append((i+0.5, j+0.5))
        
        return vortex_positions, antivortex_positions

    def plot_with_vortices(self, show=False, save=False):
        """
        Plot the current state with vortices and anti-vortices highlighted.
        
        Args:
            show (bool): Whether to display the plot
            save (bool): Whether to save the plot
        """
        vortex_positions, antivortex_positions = self.detect_vortices()
        
        plt.figure(figsize=(10, 8))
        
        # Plot the angles
        plt.imshow(self.angles, cmap=custom_cmap)
        plt.xlim(-0.5, self.n-0.5)
        plt.ylim(-0.5, self.n-0.5)
        plt.clim(-np.pi, np.pi)
        
        # Mark vortices and anti-vortices
        vortex_y, vortex_x = zip(*vortex_positions) if vortex_positions else ([], [])
        antivortex_y, antivortex_x = zip(*antivortex_positions) if antivortex_positions else ([], [])
        
        plt.scatter(vortex_x, vortex_y, color='lime', s=100, marker='o', edgecolors='black', label='Vortex (+1)')
        plt.scatter(antivortex_x, antivortex_y, color='magenta', s=100, marker='o', edgecolors='black', label='Anti-vortex (-1)')
        
        # Add colorbar and labels
        clb = plt.colorbar(orientation="vertical")
        clb.set_label('Angle', rotation=90)
        clb.set_ticks([-np.pi, 0, np.pi], labels=['$-\pi$', '0', '$\pi$'])
        
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.title(f'XY Model at T={round(self.temp, 2)} with Vortices\n'
                  f'Vortices: {len(vortex_positions)}, Anti-vortices: {len(antivortex_positions)}')
        plt.legend(loc='upper right')
        
        if save:
            plt.savefig(f"{self.frame//self.n**2}.png", dpi=300)
        
        if show:
            plt.show()
        else:
            plt.close()

    def animate(self, name):
        """
        Create an animation from saved frames.
        
        Parameters:
        -----------
        name (str): Name of the output GIF file
        """
        files = glob.glob("*.png")
        n = []
        for i in files:
            try:
                n.append(int(i[:-4]))
            except: # If the name of the file is not a number
                continue
        sorted_files = [i for _, i in sorted(zip(n, files))]

        with imageio.get_writer(f"{name}.gif", mode='I', duration=0.1) as writer:
            for frame in sorted_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        
        # Clean up the directory 
        for frame in sorted_files:
            os.remove(frame)

