import numpy as np
import matplotlib.pyplot as plt

class Atom:
    def __init__(self, mass=6.6e-26, temperature=None):
        self.position = np.random.uniform(-1, 1, size=3)  # Initial position
        self.velocity = np.random.uniform(-1, 1, size=3)  # Initial velocity
        self.mass = mass  # Mass in kg
        self.force = np.zeros(3)  # Force accumulator
        
        if temperature:
            kb = 1.38e-23  # Boltzmann constant
            sigma_v = np.sqrt(kb * temperature / self.mass)
            self.velocity = np.random.normal(0, sigma_v, 3)  # Maxwell-Boltzmann distribution

    def compute_force(self, other):
        kb = 1.38e-23
        sigma = 3.405e-10  # m
        epsilon = 119.8 * kb  # J

        r_vec = self.position - other.position
        r_vec -= np.round(r_vec)  # Apply minimum image convention
        mag_r = np.linalg.norm(r_vec)

        if mag_r > 0:
            lj_force = (24 * epsilon / mag_r**2) * ((2 * (sigma / mag_r) ** 12) - ((sigma / mag_r) ** 6)) * r_vec / mag_r
            return -lj_force
        return np.zeros(3)

class Simulation:
    def __init__(self, num_atoms=10, box_size=2e-9, temperature=300):
        self.atoms = [Atom(temperature=temperature) for _ in range(num_atoms)]
        self.box_size = box_size  # Simulation box size
        self.time = 0

        for atom in self.atoms:
            atom.position *= self.box_size / 2  # Scale positions to fit within the box

    def apply_pbc(self, atom):
        atom.position = (atom.position + self.box_size / 2) % self.box_size - self.box_size / 2

    def compute_forces(self):
        for atom in self.atoms:
            atom.force = np.zeros(3)
        
        for i, atom in enumerate(self.atoms):
            for j, other in enumerate(self.atoms):
                if i < j:
                    force = atom.compute_force(other)
                    atom.force += force
                    other.force -= force

    def velocity_verlet(self, dt):
        for atom in self.atoms:
            atom.velocity += 0.5 * (atom.force / atom.mass) * dt
            atom.position += atom.velocity * dt
            self.apply_pbc(atom)
        
        self.compute_forces()
        
        for atom in self.atoms:
            atom.velocity += 0.5 * (atom.force / atom.mass) * dt
    
    def evolve_system(self, dt, t_end, plot=False):
        while self.time < t_end:
            self.compute_forces()
            self.velocity_verlet(dt)
            self.time += dt
            print(f"System evolved to {self.time:.3e} s")
            if plot:
                self.plot_system(save=True)

    def plot_system(self, save = False):
        positions = np.array([atom.position for atom in self.atoms])
        plt.figure()
        plt.scatter(positions[:, 0], positions[:, 1], c='red')
        plt.xlim(-self.box_size / 2, self.box_size / 2)
        plt.ylim(-self.box_size / 2, self.box_size / 2)
        plt.xlabel("Meters")
        plt.ylabel("Meters")
        plt.title(f"System at time {self.time:.3e} s")
        if save:
            plt.savefig(f"sys_{self.time}.png")
        plt.show()

# Example usage:
sim = Simulation(num_atoms=20, temperature=300)
sim.evolve_system(dt=1e-15, t_end=1e-12, plot=True)


import glob
import os
import imageio as imageio
from matplotlib.animation import FuncAnimation, PillowWriter


