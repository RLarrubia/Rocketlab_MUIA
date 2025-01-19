import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Definir límites
low = [0.01, 0.005, 0.005, 0.005, 0.05, 0.001, 0.001, 5, 2]
up = [0.1, 0.02, 0.02, 0.02, 0.5, 0.01, 0.01, 40, 2.001]

# Parámetros
SIGMA_ULT = 125  # Esfuerzo último (MPa)
MIN_MASS = 1e-3  # Masa mínima (kg)

# Evaluación rápida (ajustar según modelo físico)
def evaluate_rocket(individual):
    # Cálculos simulados (reemplazar con fórmulas reales si se tienen)
    h_max = individual[0] * 1000  # Altura máxima simulada (m)
    M_total = max(sum(individual) * 10, MIN_MASS)  # Masa total mínima
    sigma_r = individual[0] * 10 + individual[3] * 5  # Esfuerzo radial simulado (MPa)
    return h_max, M_total, sigma_r

# Penalización por restricciones
def apply_constraints(individual):
    Rg, Rs, R0, R = individual[2], individual[3], individual[1], individual[0]
    penalty = 0

    # Restricciones geométricas
    if Rs <= Rg: penalty += 1e6 * (Rg - Rs + 1e-6)
    if R0 >= R: penalty += 1e6 * (R0 - R + 1e-6)

    # Restricción de esfuerzo radial
    _, _, sigma_r = evaluate_rocket(individual)
    if sigma_r <= SIGMA_ULT: penalty += 1e6 * (SIGMA_ULT - sigma_r + 1e-6)

    return penalty

# Evaluación ponderada
def evaluate_weighted(individual, w_h=0.2, w_m=0.8):
    penalty = apply_constraints(individual)
    if penalty > 0: return penalty,
    h_max, M_total, _ = evaluate_rocket(individual)
    return -w_h * h_max + w_m * M_total,

# Configuración de DEAP
creator.create("FitnessSingle", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessSingle)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, low, up)
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 lambda: [np.random.uniform(l, u) for l, u in zip(low, up)])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=up, eta=20.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_weighted)

# Algoritmo evolutivo
if __name__ == '__main__':
    pop_size = 75  # Tamaño de la población
    n_gen = 25  # Número de generaciones

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_gen, 
                                       stats=stats, verbose=True)

    # Obtener los valores evaluados para graficar
h_values, m_values = [], []
for ind in pop:
    h, m, _ = evaluate_rocket(ind)
    h_values.append(h)
    m_values.append(m)

# Encontrar los puntos extremos
max_h_index = np.argmax(h_values)  # Índice del punto con máxima altura
min_m_index = np.argmin(m_values)  # Índice del punto con mínima masa

# Valores extremos
max_h_point = (m_values[max_h_index], h_values[max_h_index])
min_m_point = (m_values[min_m_index], h_values[min_m_index])

# Graficar resultados
plt.figure(figsize=(12, 8))

# Graficar puntos evaluados
plt.scatter(m_values, h_values, c="blue", label="Puntos evaluados", alpha=1.0)

# Graficar los puntos extremos
plt.scatter(*max_h_point, c="green", edgecolor="black", s=150, label="Máxima altura")
plt.scatter(*min_m_point, c="orange", edgecolor="black", s=150, label="Mínima masa")

# Detalles de la gráfica
plt.title("Altura máxima vs Masa total", fontsize=14)
plt.xlabel("Masa total (kg)", fontsize=12)
plt.ylabel("Altura máxima (m)", fontsize=12)
plt.legend(fontsize=12, loc="lower left")  # Mover la leyenda a la esquina inferior izquierda
plt.grid()
plt.savefig("extreme_points.png")  # Guardar gráfica
plt.show()

