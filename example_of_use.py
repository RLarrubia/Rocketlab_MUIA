import numpy as np
from scipy.optimize import minimize
from rocket_module import rocket

# Definir una función que valida y ajusta los parámetros
def validate_parameters(parameters):
    """
    Valida y ajusta los parámetros para cumplir con las restricciones físicas.

    Args:
        parameters (list): Lista con los valores de R.

    Returns:
        list: Lista de parámetros válidos.
    """
    R = max(parameters[0], 1e-6)  # Garantizar que R sea positivo
    return [R]

# Definir una función que llama al simulador y devuelve h_max
def evaluate_hmax(parameters):
    """
    Evalúa el h_max para un conjunto dado de parámetros.

    Args:
        parameters (list): Lista con los valores de R.

    Returns:
        float: El valor negativo de h_max (para minimizar).
    """
    parameters = validate_parameters(parameters)
    simulation_params = {
        'R': parameters[0],
        'R0': 0.005,  # Fijo
        'Rg': 0.005,  # Fijo
        'Rs': 0.006,  # Fijo
        'L': 0.08,  # Fijo
        't_chamber': 0.002,  # Fijo
        't_cone': 0.002,  # Fijo
        'alpha': 20,  # Fijo
        'Mpl': 0.5,  # Fijo
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
    Optimiza los parámetros para maximizar h_max variando solo R.

    Returns:
        dict: Resultados de la optimización con los parámetros óptimos y h_max.
    """
    bounds = [
        (0.01, 0.1)  # R (m)
    ]

    initial_guess = [
        0.011  # R (Radio inicial del cuerpo principal)
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
        print(f"Parámetro óptimo R: {results['optimized_parameters'][0]}")
        print(f"Altura máxima optimizada (h_max): {results['h_max']}")
    except RuntimeError as e:
        print(e)