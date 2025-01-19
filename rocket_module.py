import numpy as np
from engine_module import *
from trayectory_module import *
from aerodynamics_module import *
import csv

class rocket():

    def __init__(self, parameters):
        self.set_parameters(parameters) # Set all parameters as class attributes dynamically

        # Calculate mass and geometry properties based on initial parameters
        self.calculate_mass_properties()
        self.S = np.pi * (self.R + self.t_chamber) ** 2 

        dt_min = min(self.dt_engine, self.dt_trayectory)   # Smallest time step is limitant for both problems until engine shut down

        self.initialize_trajectory(dt_min)                 # Initialize trajectory module
        self.initialize_engine(dt_min)                     # Initialize engine module
        self.initialize_aerodynamics()

    def set_parameters(self, parameters):
        """Set all parameters as class attributes dynamically."""
        for key, value in parameters.items():
            setattr(self, key, value)
    
    def calculate_mass_properties(self):
        Mc = np.pi * ((self.R + self.t_chamber) ** 2 - self.R ** 2) * self.L * self.rho_c                   # Mass of the chamber
        Mcone = np.pi * self.R * self.R/ np.sin(np.deg2rad(self.alpha)) * self.t_cone * self.rho_cone       # Mass of the cone
        Mp0 = np.pi * (self.R ** 2 - self.R0 ** 2) * self.L * self.rho_pr                                   # Inital fuel mass

        self.Mi = Mc + Mcone + self.Mpl                                                                     # Inert mass (all without fuel)
        self.M0 = self.Mi + Mp0                                                                             # Total inital mass

    def initialize_trajectory(self, dt_min):
        """Initialize the trajectory module with the necessary configurations."""
        trayectory_geometry_parameters = TrayectoryGeometryParameters(M0=self.M0, S=self.S)
        trayectory_physical_constants = TrayectoryPhysicalConstants(g0=self.g0, Ra=self.Ra, Re=self.Re)
        trayectory_initial_conditions = TrayectoryInitialConditions(t0=self.t0, h0=self.h0, v0=self.v0)
        trayectory_simulation_configuration = TrayectorySimulationConfiguration(dt=dt_min, solver=self.solver_trayectory)
        
        self.rocket_trayectory = trayectory(
            trayectory_physical_constants,
            trayectory_initial_conditions,
            trayectory_geometry_parameters,
            trayectory_simulation_configuration
        )

        self.Pa0 = self.rocket_trayectory.Pa_t[-1]  # Initial atmospheric preassure for h0

    def initialize_engine(self, dt_min):
        """Initialize the engine module with the necessary configurations."""
        engine_geometry = EngineGeometryParameters(R=self.R, Rg=self.Rg, Rs=self.Rs, R0=self.R0, L=self.L)
        fuel_parameters = EngineFuelParameters(
            Tc=self.Tc, a=self.a, n=self.n, M_molar=self.M_molar, gamma=self.gamma,
            rho_pr=self.rho_pr, Rend=self.Rend
        )
        initial_conditions = EngineInitialConditions(t0=self.t0, Pa0=self.Pa0)
        simulation_configurations = EngineSimulationConfiguration(dt=dt_min, solver=self.solver_engine)
        
        self.rocket_engine = RocketEngine(
            engine_geometry,
            fuel_parameters,
            initial_conditions,
            simulation_configurations
        )

    def initialize_aerodynamics(self):
        """Initialize the aerodynamics module with the necessary configurations."""
        aerodynamics_geometry = AerodynamicsGeometryParameters(D=2*self.R,  alpha_cone=self.alpha, L_body=self.L, Rs=self.Rs)
        flow_constants = AerodynamicsFlowConstants(M_molar=self.M_molar_air,  gamma=self.gamma_air, mu=self.viscosity_air)
        self.rocket_aerodynamics = aerodynamics(aerodynamics_geometry, flow_constants)

    def simulation(self):
        """
        Performs a simulation of the rocket coupling the engine performance with the trayectory through the atmospheric preassure
        and the instaneous mass and thrust of the rocket. Both simulations are performed wiht the lowest step and when the engine 
        stops the trayectory step can be set to a higher value.
        """
        while True:
            # Rocket module 
            self.rocket_engine.solve_step(Pa=self.rocket_trayectory.Pa_t[-1])

            # Aerodynamics module
            engine_on = not self.rocket_engine.flag_stop_condition('engine_stop')
            _, rho_flow, T_flow = self.rocket_trayectory.atm_properties(self.rocket_trayectory.h_t[-1])
            cd0 = self.rocket_aerodynamics.calculate_Cd0(np.abs(self.rocket_trayectory.v_t[-1]), rho_flow, T_flow, engine_on)
            
            # Trayectory moddule
            self.rocket_trayectory.solve_step(dt=self.dt_trayectory if not engine_on else None, 
                                              E=self.rocket_engine.E_t[-1], Cd=cd0, M=self.rocket_engine.Mp_t[-1]+self.Mi)
            if self.rocket_trayectory.flag_stop_condition(self.stop_condition): break 
            if not engine_on and self.rocket_trayectory.h_t[-1] == 0: break #! check compatibility with other cases
        self.results_2_csv()


    def graphs(self):
        """
        Graphs the performance characteristics of the rocket.
        """
        self.rocket_engine.graphs()
        self.rocket_trayectory.graphs()

    def stress_analysis(self):
        """
        Calculates the circumferential stress in the chamber due to the gas presure in the worst case (Pa=0) and aplying a safety factor of 1.5.

        Returns:
            float: circumferential stress in MPa.
        """
        SF = 1.5
        Pc_max = np.max(self.rocket_engine.Pc_t)
        sigma_r = Pc_max * (self.R/self.t_chamber) * SF
        return sigma_r / 10 ** 6  #MPa
    
    def results(self):
        """
        Calculates some of the key performance parameters for optimization purposes.

        Returns:
            float, float, float: maximum heigth of the trayectory, total initial mass and circumferential stress in the chamber.
        """
        h_max = np.max(self.rocket_trayectory.h_t)
        M_total = self.M0
        sigma_r = self.stress_analysis()
        return {'h_max': h_max, 'M_total': M_total, 'sigma_r': sigma_r}

    def results_2_csv(self):
        with open('results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            results = self.results()
            
            # Escribir cada clave y valor en filas separadas con un espacio después de la coma
            for key, value in results.items():
                writer.writerow([key, f" {value}"])

if __name__ == '__main__':  
    parameters = {
    ## Design parameters
    'R': 2.25e-2, 
    'R0': 0.5e-2, 
    'Rg': 0.5e-2, 
    'Rs': 0.5e-2, 
    'L': 8e-2, 
    't_chamber': 0.002,
    't_cone': 0.002,
    'alpha': 20, # degrees
    'Mpl': 0.5,

    ## Constants
    'Tc': 1000,
    'M_molar': 41.98e-3,
    'M_molar_air': 28.97e-3,
    'gamma': 1.3,
    'gamma_air': 1.4,
    'viscosity_air': 1.82e-05,
    'rho_pr': 1800,
    'rho_cone': 2700,
    'rho_c': 2700,
    'Rend': 1 - 0.4237,
    'a': 6e-5,
    'n': 0.32,
    'Re': 6.37e6,
    'g0': 9.80665,
    'Ra': 287,

    ## Initial conditions
    'h0': 0,
    'v0': 0,
    't0': 0,
    'solver_engine':'RK4',
    'solver_trayectory':'Euler',
    'dt_engine': 5e-5,
    'dt_trayectory':100e-5,
    'stop_condition':'max_height'
    }

    r = rocket(parameters)
    r.simulation()
    r.graphs()
    r.results_2_csv()