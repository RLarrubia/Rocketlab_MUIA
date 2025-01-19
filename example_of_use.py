import rocket_module

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

r = rocket_module.rocket(parameters)

r.simulation()
r.graphs()
r.results_2_csv()

resultados = r.results()
h = resultados['h_max']
M = resultados['M_total']
sigma = resultados['sigma_r']
print(h)
