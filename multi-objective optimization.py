import numpy as np
from deap import base, creator, tools, algorithms
from rocket_module import rocket

# Clase cohete: función para calcular altura, masa y esfuerzo radiales
def evaluar_cohete(parametros):
    # Instancia de la clase ya implementada
    # Ejemplo de llamada: Rocket([largo_camara, radio_camara, diametro_garganta, diametro_salida, ...])
    cohete = rocket(parametros)

    # Calcular resultados
    resultados = cohete.results()
    h_max = resultados["h_max"]
    M_total = resultados["M_total"]
    sigma_r = resultados["sigma_r"]

    # Restricción: penalizar si el esfuerzo radial excede el límite permitido
    sigma_ult = 125  # Cambia esto por el valor límite adecuado
    if sigma_r > sigma_r:
        print ("Esfuerzo último superado, el cohete no ha soportado las cargas")
        print (f"El esfuerzo generado es de {sigma_r} MPa y supera los 125 MPa de esfuerzo último calculado")
    else:
        return h_max, M_total
    
# Definición del problema como minimización para DEAP
creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))  # Maximizar altura (+1.0), minimizar masa (-1.0)
creator.create("Individual", list, fitness=creator.FitnessMin)

# Rango de las variables de diseño
simulation_params = {
        'R': (0.005, 0.025),
        'R0': (0.005, 0.025),
        'Rg': (0.0025, 0.015),  
        'Rs': (0.005, 0.025),  
        'L': (0.05, 0.5),
        't_chamber': (0.002, 0.01), 
        't_cone': (0.002, 0.01), 
        'alpha': (10, 45), 
        'Mpl': 2,  
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

# Crear individuo y población
def crear_individuo():
    return [np.random.uniform(simulation_params[var][0], simulation_params[var][1]) for var in simulation_params]

toolbox = base.Toolbox()
toolbox.register("attr_float", crear_individuo)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Registro de funciones en el toolbox
toolbox.register("evaluate", evaluar_cohete)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[simulation_params[var][0] for var in simulation_params], 
                 up=[simulation_params[var][1] for var in simulation_params], eta=20.0, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Configuración del algoritmo genético
pop_size = 100
ngen = 200
cxpb = 0.7
mutpb = 0.2

# Generar población inicial
population = toolbox.population(n=pop_size)

# Optimización evolutiva
algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=pop_size*2, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                          stats=None, halloffame=None, verbose=True)

# Extraer frente de Pareto
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

# Mostrar resultados del frente de Pareto
for ind in pareto_front:
    print(f"Altura máxima: {-ind.fitness.values[0]:.2f}, Masa total: {ind.fitness.values[1]:.2f}, Parámetros: {ind}")
