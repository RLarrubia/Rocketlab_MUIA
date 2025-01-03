import numpy as np
from scipy.optimize import minimize
from rocket_module import rocket

# Definir una función que valida y ajusta los parámetros
def validate_parameters(parameters):
    """
    Valida y ajusta los parámetros para cumplir con las restricciones físicas.

    Args:
        parameters (list): Lista con los valores de L, alpha, t_chamber, t_cone, Rg y R0.

    Returns:
        list: Lista de parámetros válidos.
    """
    L = max(parameters[0], 1e-6)  # Garantizar que L sea positivo
    alpha = max(parameters[1], 1e-6)  # Garantizar que alpha sea positivo
    t_chamber = max(parameters[2], 1e-6)  # Garantizar que t_chamber sea positivo
    t_cone = max(parameters[3], 1e-6)  # Garantizar que t_cone sea positivo
    Rg = max(parameters[4], 1e-6)  # Garantizar que Rg sea positivo
    R0 = max(parameters[5], 1e-6)  # Garantizar que R0 sea positivo
    return [L, alpha, t_chamber, t_cone, Rg, R0]

# Definir una función que llama al simulador y devuelve h_max
def evaluate_hmax(parameters):
    """
    Evalúa el h_max para un conjunto dado de parámetros.

    Args:
        parameters (list): Lista con los valores de L, alpha, t_chamber, t_cone, Rg y R0.

    Returns:
        float: El valor negativo de h_max (para minimizar).
    """
    parameters = validate_parameters(parameters)
    simulation_params = {
        'R': 0.0225,  # Fijo
        'R0': parameters[5],
        'Rg': parameters[4],
        'Rs': 0.006,  # Fijo
        'L': parameters[0],
        't_chamber': parameters[2],
        't_cone': parameters[3],
        'alpha': parameters[1],
        'Mpl': 2,  # Fijo
        # Constantes
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
        # Condiciones iniciales
        'h0': 0,
        'v0': 0,
        't0': 0,
        'solver_engine': 'RK4',
        'solver_trayectory': 'Euler',
        'dt_engine': 5e-5,
        'dt_trayectory': 1e-3,
        'stop_condition': 'max_height'
    }

    # Crear la instancia del cohete y simular
    try:
        r = rocket(simulation_params)
        r.simulation()
        result = r.results()
        return -result['h_max']  # Negativo porque estamos minimizando
    except Exception as e:
        print(f"Error en la simulación: {e}")
        return 1e6  # Penalización alta pero finita

# Definir los límites para los parámetros y calcular el gradiente
def optimize_hmax():
    """
    Optimiza los parámetros para maximizar h_max variando L, alpha, t_chamber, t_cone, Rg y R0.

    Returns:
        dict: Resultados de la optimización con los parámetros óptimos y h_max.
    """
    bounds = [
        (0.05, 0.5),   # L (m)
        (5, 30),       # alpha (grados)
        (0.001, 0.01), # t_chamber (m)
        (0.001, 0.01), # t_cone (m)
        (0.001, 0.05), # Rg (m)
        (0.001, 0.05)  # R0 (m)
    ]

    initial_guess = [
        0.08,  # L (Longitud inicial del cohete)
        20,    # alpha (Ángulo inicial del cono)
        0.002, # t_chamber (Espesor inicial de la cámara)
        0.002, # t_cone (Espesor inicial del cono)
        0.005, # Rg (Radio inicial del cuello de la boquilla)
        0.005  # R0 (Radio inicial del extremo del cuerpo)
    ]

    # Ejecutar la optimización con límite de iteraciones
    options = {
        'maxiter': 50,  # Limitar el número máximo de iteraciones
        'disp': True,   # Mostrar información sobre el progreso
    }

    result = minimize(evaluate_hmax, initial_guess, bounds=bounds, method='L-BFGS-B', options=options)

    if result.success:
        optimized_parameters = result.x
        h_max_optimized = -result.fun
        return {
            'optimized_parameters': optimized_parameters,
            'h_max': h_max_optimized
        }
    else:
        raise RuntimeError("La optimización no tuvo éxito.")

if __name__ == '__main__':
    try:
        results = optimize_hmax()
        print(f"Parámetros óptimos L: {results['optimized_parameters'][0]}, alpha: {results['optimized_parameters'][1]}, t_chamber: {results['optimized_parameters'][2]}, t_cone: {results['optimized_parameters'][3]}, Rg: {results['optimized_parameters'][4]}, R0: {results['optimized_parameters'][5]}")
        print(f"Altura máxima optimizada (h_max): {results['h_max']}")
    except RuntimeError as e:
        print(e)
