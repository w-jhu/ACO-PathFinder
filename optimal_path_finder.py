import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Waypoint distance with altitude
def find_distance(waypoint1, waypoint2):
    lat1, long1, alt1 = [waypoint1[key] for key in ('latitude', 'longitude', 'altitude')]
    lat2, long2, alt2 = [waypoint2[key] for key in ('latitude', 'longitude', 'altitude')]

    geo_distance = geodesic((lat1, long1), (lat2, long2)).feet
    alt_difference = alt1 - alt2
    distance = math.sqrt(geo_distance ** 2 + alt_difference ** 2)

    return distance

# Calculate x, y coordinates for plotting
def to_xy_coords(waypoints, min_lat, min_long):
    x_coords = []
    y_coords = []

    for waypoint in waypoints:
        lat = waypoint['latitude']
        long = waypoint['longitude']

        x = long - min_long
        y = lat - min_lat

        x_coords.append(x)
        y_coords.append(y)

    return x_coords, y_coords

# Ant colony optimization to solve traveling salesman problem
def aco_find_optimal_path(distances, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_waypoints = len(distances)
    pheromone = np.ones((n_waypoints, n_waypoints))
    best_path = None
    best_path_length = np.inf

    # Random pathfinding for each ant
    def ant_loop(distances, alpha, beta, n_waypoints, pheromone):
        visited = [False] * n_waypoints
        current_waypoint = 0
        visited[current_waypoint] = True
        path = [current_waypoint]
        path_length = 0

        # Ant visits every waypoint
        while False in visited:
            unvisited = np.where(np.logical_not(visited))[0]
            probabilities = np.zeros(len(unvisited))

            # Calculate probabilities for next waypoint based on pheromone level and distance
            for i, unvisited_waypoint in enumerate(unvisited):
                probabilities[i] = pheromone[current_waypoint, unvisited_waypoint]**alpha\
                    / distances[current_waypoint][unvisited_waypoint]**beta
                
            probabilities /= np.sum(probabilities)

            next_waypoint = np.random.choice(unvisited, p=probabilities)
            path.append(next_waypoint)
            path_length += distances[current_waypoint][next_waypoint]
            visited[next_waypoint] = True
            current_waypoint = next_waypoint

        # Add in distance for return trip to start    
        path_length += distances[current_waypoint][0]
        paths.append(path)
        path_lengths.append(path_length)       
    
    # Loops to send n_ants out in each of n_iterations
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []

        for ant in range(n_ants):
            ant_loop(distances, alpha, beta, n_waypoints, pheromone)

        # Prevents convergence towards suboptimal solution
        pheromone *= evaporation_rate

        # Updates optimal path and path length along with pheromone levels
        for path, path_length in zip(paths, path_lengths):
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

            for i in range(n_waypoints - 1):
                pheromone[path[i], path[i + 1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length

    # Append starting waypoint to the end of best path
    best_path.append(0)
    print("Shortest path length: ", best_path_length)
    print("Shortest path: ", best_path)
    return best_path_length, best_path

def main():
    cur_directory = os.path.dirname(os.path.abspath(__file__))
    coords_file_path = os.path.join(cur_directory, "data", "coords.json")

    # Reads in data
    with open(coords_file_path, "r") as json_file:
        coords = json.load(json_file)

    waypoints = coords.get("waypoints")
    waypoints_count = len(waypoints)
    
    # Calculates and stores distances between waypoints in 2D NumPy array
    distances = np.zeros((waypoints_count, waypoints_count))
    for i in range(waypoints_count):
        for j in range(waypoints_count):
            if i != j:
                distances[i][j] = find_distance(waypoints[i], waypoints[j])
    best_path_length, best_path = aco_find_optimal_path(distances, n_ants = 20, n_iterations = 250, alpha = 1, beta = 1, evaporation_rate = 0.5, Q = 1)

    flyzones = coords.get("flyZones")
    boundary_points = flyzones[0]['boundaryPoints']

    min_lat = float('inf')
    min_long = float('inf')

    # Find minimum latitude and longitude in boundary points
    for point in boundary_points:
        lat = point['latitude']
        long = point['longitude']

        if lat < min_lat:
            min_lat = lat

        if long < min_long:
            min_long = long

    x, y = to_xy_coords(waypoints, min_lat, min_long)
    path_x = [x[i] for i in best_path]
    path_y = [y[i] for i in best_path]

    # Plot waypoints and path, marking start as green
    plt.scatter(x, y, label='Waypoints', marker='o', color='blue', zorder=2)
    plt.scatter(x[0], y[0], label='Start', marker='o', color='green', zorder=3)
    plt.plot(path_x, path_y, label='Path', linestyle='-', color='red', zorder=1)
    plt.xticks([])
    plt.yticks([]) 
    plt.title('Map with Optimal Path')
    plt.show()

if __name__ == "__main__":
    main()