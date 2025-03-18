from simulation import Simulation, Atom

def main():
    print("Starting molecular dynamics simulation...")
    

    sim = Simulation(density=1.2, temperature=0.5, num_atoms=108, element="argon")
    print(sim)
    
    print("\nVerifying temperature scaling...")
    sim.verify_temperature_scaling()
    
    print("\nEquilibrating system...")
    sim.evolve_system(dt=0.005, t_end=0.5)
    
    print("\nCalculating pressure...")
    pressure, error = sim.get_pressure()
    print(f"Pressure: {pressure:.2f} ± {error:.2f} ε/σ³")
    
    print("\nCalculating radial distribution function...")
    sim.plot_radial_distribution()
    
    print("\nGenerating phase diagram and searching for triple point...")
    sim.plot_triple_point()
    sim.animate_system("18m")
    

if __name__ == "__main__":
    main()
