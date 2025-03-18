from simulation import Simulation, Atom

class Testing(Simulation):

  def verify_temperature_scaling(self):
    """
    Verify that temperature scaling is working correctly.
    
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


  def test_phases(self):
  """
  Test the system behavior at different phases (solid, liquid, gas).
  Creates visualizations to compare atomic motion.
  """
  phases = {
      "solid": [0.5, 0.9],    # Low temperature, high density
      "liquid": [1.0, 0.8],   # Medium temperature, medium density
      "gas": [3.0, 0.3]       # High temperature, low density
  }
  
  # Store original parameters
  original_temp = self.temperature
  original_density = self.density
  original_box_size = self.box_size
  
  # Test each phase
  for phase_name, (temp, dens) in phases.items():
    print(f"\nTesting {phase_name} phase (T={temp}, ρ={dens})...")
    
    # Set phase parameters
    self.temperature = temp
    self.density = dens
    self.box_size = (self.num_atoms / self.density) ** (1/3)
    
    # Reset atom positions to FCC lattice
    self.positions = self._initialize_fcc_lattice(self.density)
    self.velocities = self._initialize_velocities()
    
    # Create new atoms with these positions and velocities
    self.atoms = [Atom(element=self.element, temperature=self.temperature, 
                  position=self.positions[i], 
                  velocity=self.velocities[i]) for i in range(self.num_atoms)]
    
    # Equilibrate system
    print("Equilibrating...")
    self._equilibrate_velocities(num_steps=20)
    self.evolve_system(dt=0.001, t_end=0.5, plot3d=True)
    self.animate_system(f"{phase_name}")
      
  # Restore original parameters
  self.temperature = original_temp
  self.density = original_density
  self.box_size = original_box_size
  
  print("\nPhase testing completed!")
  
  
  # Restore original temperature
  self.temperature = original_temp
  self._equilibrate_velocities()
  
  return target_temps, actual_temps
