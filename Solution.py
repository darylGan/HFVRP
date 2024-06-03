# -*- coding: utf-8 -*-
"""ModeFair Home Assessment

# Data
"""

#import libraries
import numpy as np
import pandas as pd

# read customer data
customers = pd.read_csv('/content/drive/MyDrive/Job/ModeFair/Customer.csv')
customers

# read vehicles data
vehicles = pd.read_csv('/content/drive/MyDrive/Job/ModeFair/Vehicle.csv')
vehicles

# depot location
depot = (4.4184, 114.0932)

# plot map
from matplotlib import pyplot as plt
customers.plot(kind='scatter', x='Latitude', y='Longitude')
plt.scatter(depot[0], depot[1], c='red', s=100, label='Depot')
plt.title('Vehicle Routing Problem Map')
plt.legend()
plt.grid()
plt.show()

# all coordinates as array
depot_arr = np.array(depot)
customer_arr = np.array(customers[['Latitude','Longitude']])
all_points = np.vstack([depot_arr, customer_arr])

# calculate distance matrix
from scipy.spatial import distance_matrix
dist_matrix = distance_matrix(all_points, all_points)
dist_matrix = dist_matrix*100   # multiply 100 to follow assumption formula

# distance matrix as df for better display
dist_matrix_df = pd.DataFrame(dist_matrix)
dist_matrix_df.rename(columns={0:'Depot'}, index={0:'Depot'}, inplace=True)
dist_matrix_df

"""# Functions used by both Algorithms"""

# create initial random solution
import random
def create_initial_solution():
  return random.sample(range(1, len(customers)+1), len(customers))

# evaluate total distance and total cost of a solution
def evaluate(solution):
  total_distance = 0
  total_cost = 0
  route = []    # current route being constructed
  routes = []   # list of constructed routes
  route_demand = 0
  vehicle_index = 0

  for customer_index in solution:
    demand = customers.iloc[customer_index - 1]['Demand']

    if route_demand + demand <= vehicles.iloc[vehicle_index]['Capacity']:
      route.append(customer_index)
      route_demand += demand
    else:   # over capacity
      routes.append((route, vehicle_index, route_demand))   # save route
      route = [customer_index]    # create new route with current customer
      route_demand = demand
      vehicle_index = (vehicle_index + 1) % len(vehicles)   #cycle through available vehicles

  if route:   # if there are remaining customers, append to routes
    routes.append((route, vehicle_index, route_demand))

  # calculate distance and cost of routes
  for route, vehicle_index, route_demand in routes:
    route_distance = dist_matrix[0][route[0]]
    for i in range(len(route)-1):
      route_distance += dist_matrix[route[i]][route[i+1]]
    route_distance += dist_matrix[route[-1]][0]

    total_distance += route_distance
    total_cost += route_distance * vehicles.iloc[vehicle_index]['Cost (RM per km)']
  return total_distance, total_cost

# print final solution
def print_solution(best_solution, customers, vehicles, dist_matrix, depot):
  total_distance = 0
  total_cost = 0
  route = []
  routes = []
  route_demand = 0
  vehicle_index = 0
  n = 1

  for customer_index in best_solution:
    demand = customers.iloc[customer_index - 1]['Demand']

    if route_demand + demand <= vehicles.iloc[vehicle_index]['Capacity']:
      route.append(customer_index)
      route_demand += demand
    else:
      routes.append((route, vehicle_index, route_demand))
      route = [customer_index]
      route_demand = demand
      vehicle_index = (vehicle_index + 1) % len(vehicles)

  if route:
    routes.append((route, vehicle_index, route_demand))

  # calculate and display distance and cost of routes
  for route, vehicle_index, route_demand in routes:
    route_distance = dist_matrix[0][route[0]]
    route_details = f"Depot -> C{route[0]} ({route_distance:.3f} km)"
    for i in range(len(route)-1):
      segment_distance = dist_matrix[route[i]][route[i+1]]
      route_distance += segment_distance
      route_details += f" -> C{route[i + 1]} ({segment_distance:.3f} km)"
    return_to_depot_distance = dist_matrix[route[-1]][0]
    route_distance += return_to_depot_distance
    route_details += f" -> Depot ({return_to_depot_distance:.3f} km)"

    total_distance += route_distance
    route_cost = route_distance * vehicles.iloc[vehicle_index]['Cost (RM per km)']
    total_cost += route_cost
    print(f"Vehicle {n} ({vehicles.iloc[vehicle_index]['Vehicle']}): \nRound Trip Distance: {route_distance:.3f} km, Cost: RM {route_cost:.2f}, Demand: {route_demand:.0f} \nRoute: {route_details}\n")
    n +=1

  print(f"Total Distance = {total_distance:.3f} km")
  print(f"Total Cost = RM {total_cost:.2f}\n")
  plot_routes(routes, customers, depot)

# plot final solution routes
def plot_routes(routes, customers, depot):
  depot_lat, depot_lon = depot
  plt.scatter(depot[0], depot[1], c='red', s=100, label='Depot')

  # plot every customer location
  for i, row in customers.iterrows():
    plt.scatter(row['Latitude'], row['Longitude'], c='blue', s=50, label='Customer' if i == 0 else "")   # only label once

  colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink']
  n = 0

  # plot every route
  for route, vehicle_index, route_demand in routes:
    # latitudes and longitudes of the route including depot
    route_lats = [depot_lat] + [customers.iloc[customer-1]['Latitude'] for customer in route] + [depot_lat]
    route_lons = [depot_lon] + [customers.iloc[customer-1]['Longitude'] for customer in route] + [depot_lon]

    color = colors[n]
    plt.plot(route_lats, route_lons, color=color, label=f'Route {n+1} (Vehicle {vehicles.iloc[vehicle_index]["Vehicle"]})')
    n += 1

    # annotate every customer location
    for j in range(len(route)):
      plt.text(route_lats[j+1], route_lons[j+1], f'C{route[j]}', fontsize=9, verticalalignment='bottom', horizontalalignment='right')

  plt.title('Vehicle Routing Problem - Routes')
  plt.xlabel('Latitude')
  plt.ylabel('Longitude')
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.grid()
  plt.show()

"""# Genetic Algorithm"""

# GA selection function
def select(population, fitnesses, k=3):
  selected = []
  for _ in range(len(population)):
    tournament = random.sample(list(zip(population, fitnesses)), k)   # randomly select k individuals from population (with replacement)
    selected.append(min(tournament, key=lambda x:x[1])[0])   # select individual with lowest fitness
  return selected

# GA crossover function
def crossover(parent1, parent2):
  size = len(parent1)
  cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))    # randomly select two crossover points
  child1 = [None] * size    # initialize children of same size
  child2 = [None] * size
  child1[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]    # copy from first parent
  child2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]
  fill_child(child1, parent2, cxpoint2, size)   # copy remaining from second parent
  fill_child(child2, parent1, cxpoint2, size)
  return child1, child2

def fill_child(child, parent, start, size):
  pos = start
  for i in range(size):
    if parent[i] not in child:    # checks if gene of parent is not in child, ensure that each gene is copied only once
      while child[pos] is not None:
        pos = (pos + 1) % size    # start from specified position and wrap around if necessary
      child[pos] = parent[i]

# GA mutation function
def mutate(solution, mutpb):   # mutate probability
  for i in range(len(solution)):
    if random.random() < mutpb:   # rand num 0~1
      j = random.randint(0, len(solution)-1)    # position of element to swap with current element
      solution[i], solution[j] = solution[j], solution[i]

# Genetic Algorithm
def genetic_algorithm(customers, vehicles, population_size, generations, cxpb, mutpb):
  population = [create_initial_solution() for _ in range(population_size)]
  fitnesses = [evaluate(indv) for indv in population]

  for gen in range(generations):
    selected = select(population, fitnesses)    # in each generation, individuals chosen
    next_population = []

    for i in range(0, len(selected), 2):
      if i + 1 < len(selected) and random.random() < cxpb:    # at least two individuals and probability met
        child1, child2 = crossover(selected[i], selected[i + 1])
      else:   # no crossover
        child1, child2 = selected[i], selected[i + 1] if i + 1 < len(selected) else selected[i]

      next_population.append(child1)
      next_population.append(child2)

    for indv in next_population:
      mutate(indv, mutpb)

    population = next_population
    fitnesses = [evaluate(indv) for indv in population]

  best_solution = min(zip(population, fitnesses), key=lambda x:x[1])[0]
  best_fitness = min(fitnesses, key=lambda x:x[1])
  return best_solution, best_fitness

# hyperparameter ranges to search
population_size_list = [50, 100, 150]
generations_list = [50, 100, 150]
cxpb_list = [0.6, 0.7, 0.8]
mutpb_list = [0.1, 0.2, 0.3]

# hyperparameter tuning for GA
def hyperparameter_tuning_ga(customers, vehicles, population_size_list, generations_list, cxpb_list, mutpb_list):
  best_hyperparams = None
  best_solution = None
  best_distance = float('inf')
  best_cost = float('inf')

  for population_size in population_size_list:
    for generations in generations_list:
      for cxpb in cxpb_list:
        for mutpb in mutpb_list:
          solution, (distance, cost) = genetic_algorithm(customers, vehicles, population_size, generations, cxpb, mutpb)
          if cost < best_cost:
            best_hyperparams = (population_size, generations, cxpb, mutpb)
            best_solution = solution
            best_distance = distance
            best_cost = cost
          print(f"population_size: {population_size}, generations: {generations}, cxpb: {cxpb}, mutpb: {mutpb} -> Total Distance: {distance:.3f} km, Total Cost: RM {cost:.2f}")

  return best_hyperparams, best_solution, best_distance, best_cost

best_hyperparams, best_solution, best_distance, best_cost = hyperparameter_tuning_ga(customers, vehicles, population_size_list, generations_list, cxpb_list, mutpb_list)
print(f"\nBest Hyperparameters: population_size = {best_hyperparams[0]}, generations = {best_hyperparams[1]}, cxpb = {best_hyperparams[2]}, mutpb = {best_hyperparams[3]}")

GA_solution = print_solution(best_solution, customers, vehicles, dist_matrix, depot)

"""# Variable Neighbourhood Search"""

# local search for better solution
def local_search(solution):
  best_solution = solution.copy()
  best_distance, best_cost = evaluate(best_solution)

  for i in range(len(solution)):
    for j in range(i + 1, len(solution)):   # each pair of elements only considered once
      new_solution = solution.copy()
      new_solution[i], new_solution[j] = new_solution[j], new_solution[i]   # swapping two consecutive customers
      new_distance, new_cost = evaluate(new_solution)

      if new_cost < best_cost:
        best_solution, best_distance, best_cost = new_solution, new_distance, new_cost
  return best_solution, best_distance, best_cost

# shake solution
def shake(solution, k):   # k = number of shakes
  new_solution = solution.copy()

  for _ in range(k):
    i, j = random.sample(range(len(solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]   # swapping two random customers
  return new_solution

# Variable Neighbourhood Search algorithm
def variable_neighbourhood_search(customers, vehicles, max_iterations, k_max):
  best_solution = create_initial_solution()
  best_distance, best_cost = evaluate(best_solution)

  iteration = 0
  while iteration < max_iterations:
    k = 1   # k = neighbourhood level
    while k <= k_max:
      new_solution = shake(best_solution, k)
      new_solution, new_distance, new_cost = local_search(new_solution)
      if new_cost < best_cost:
        best_solution, best_distance, best_cost = new_solution, new_distance, new_cost
        k = 1   # restart search beginning with new best solution
      else:
        k += 1
    iteration += 1
  return best_solution, best_distance, best_cost

# hyperparameter ranges to search
max_iterations_list = [50, 100, 150]
k_max_list = [3, 5, 7]

# hyperparameter tuning for VNS
def hyperparameter_tuning_vns(customers, vehicles, max_iterations_list, k_max_list):
  best_hyperparams = None
  best_solution = None
  best_distance = float('inf')
  best_cost = float('inf')

  for max_iterations in max_iterations_list:
    for k_max in k_max_list:
      solution, distance, cost = variable_neighbourhood_search(customers, vehicles, max_iterations, k_max)
      if cost < best_cost:
        best_hyperparams = (max_iterations, k_max)
        best_solution = solution
        best_distance = distance
        best_cost = cost
      print(f"max_iterations: {max_iterations}, k_max: {k_max} -> Total Distance: {distance:.3f} km, Total Cost: RM {cost:.2f}")

  return best_hyperparams, best_solution, best_distance, best_cost

best_hyperparams, best_solution, best_distance, best_cost = hyperparameter_tuning_vns(customers, vehicles, max_iterations_list, k_max_list)
print(f"\nBest Hyperparameters: max_iterations = {best_hyperparams[0]}, k_max = {best_hyperparams[1]}")

VNS_solution =  print_solution(best_solution, customers, vehicles, dist_matrix, depot)

"""# Add Customer"""

new_customer_data = {
  'Customer': [11],
  'Latitude': [4.3625],
  'Longitude': [114.1500],
  'Demand': [3] }

new_customers_df = pd.DataFrame(new_customer_data)
customers = pd.concat([customers, new_customers_df], ignore_index=True)

# coordinates
customer_arr = np.array(customers[['Latitude','Longitude']])
all_points = np.vstack([depot_arr, customer_arr])

# calculate distance matrix
dist_matrix = distance_matrix(all_points, all_points)
dist_matrix = dist_matrix*100

# Best Hyperparameters from GridSearch
max_iterations = 100
k_max = 3

solution, distance, cost = variable_neighbourhood_search(customers, vehicles, max_iterations, k_max)

VNS_solution =  print_solution(solution, customers, vehicles, dist_matrix, depot)