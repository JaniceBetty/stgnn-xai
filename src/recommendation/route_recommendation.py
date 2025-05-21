import networkx as nx
import numpy as np

def build_road_graph(adj_matrix, distance_matrix, node_ids):
    """
    Build a NetworkX graph from the adjacency and distance matrices.
    node_ids: list or dict mapping node index to sensor_id.
    """
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        G.add_node(node_ids[i])
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(node_ids[i], node_ids[j], distance=distance_matrix[i, j])
    return G

def get_predicted_speed(node_idx, time_idx, stgnn_prediction):
    """Return predicted speed for a node at a given time index."""
    return stgnn_prediction[node_idx, time_idx]

def recommend_route(G, node_ids_dict, start_sensor, end_sensor, time_idx, stgnn_prediction):
    """
    Recommend the best route from start_sensor to end_sensor based on congestion.
    node_ids_dict: dict mapping sensor_id to node index.
    """
    # Set edge weights based on congestion (travel time)
    for u, v, data in G.edges(data=True):
        idx_u = node_ids_dict[u]
        idx_v = node_ids_dict[v]
        # Use average predicted speed for the edge
        speed_u = get_predicted_speed(idx_u, time_idx, stgnn_prediction)
        speed_v = get_predicted_speed(idx_v, time_idx, stgnn_prediction)
        avg_speed = (speed_u + speed_v) / 2
        distance = data['distance']
        # Avoid division by zero or negative speeds
        if avg_speed > 1e-3:
            travel_time = distance / avg_speed
        else:
            travel_time = 1e6  # Penalize blocked/congested edges
        G[u][v]['weight'] = travel_time

    # Find shortest path by travel time
    route = nx.shortest_path(G, source=start_sensor, target=end_sensor, weight='weight')
    total_time = sum(G[route[i]][route[i+1]]['weight'] for i in range(len(route)-1))
    return route, total_time

# Example usage:
# adj_matrix = ... # shape [num_nodes, num_nodes]
# distance_matrix = ... # shape [num_nodes, num_nodes], in km
# node_ids = ... # list or dict mapping node index to sensor_id
# node_ids_dict = ... # dict mapping sensor_id to node index
# stgnn_prediction = ... # shape [num_nodes, num_timesteps], in km/h
# G = build_road_graph(adj_matrix, distance_matrix, node_ids)
# route, travel_time = recommend_route(G, node_ids_dict, start_sensor, end_sensor, time_idx, stgnn_prediction)