from typing import List

import matplotlib.pyplot as pyplt
from random import randint
import sys
import time
import heapq

def parse_file(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    vertices, edges = [], []
    for i, line in enumerate(lines):
        splitted_line = line.strip("\n\r").split(" ")
        if len(splitted_line) == 2:
            vertices.append((float(splitted_line[0]), float(splitted_line[1])))
        elif len(splitted_line) == 3 and i > 0:
            vertice_1, vertice_2 = int(splitted_line[0]), int(splitted_line[1])
            distance = int(splitted_line[2])  # dans les deux sens ou pas
            p1 = vertices[vertice_1]
            p2 = vertices[vertice_2]
            edges.append((vertice_1, vertice_2, distance, p1, p2))
        elif i > 0:
            raise Exception("unable to interpret line {}: ".format(i) + line)
    print("#E={}, #V={}, V index from {} to {}".format(len(edges), len(vertices),
                                                       min(min(_[0] for _ in edges), min(_[1] for _ in edges)),
                                                       max(max(_[0] for _ in edges), max(_[1] for _ in edges))))

    return vertices, edges


def plot_sample(vertices, edges, number):
    # plot vertices
    sample_1 = [vertices[randint(0, len(vertices) - 1)] for _ in range(0, int(number))]
    pyplt.plot([_[0] for _ in sample_1], [_[1] for _ in sample_1], ".")
    pyplt.show()

    # plot edges
    sample_2 = [edges[randint(0, len(edges) - 1)] for _ in range(0, int(number))]
    for edge in sample_2:
        pyplt.plot([_[0] for _ in edge[-2:]], [_[1] for _ in edge[-2:]], "b-")
    pyplt.show()


# Resolution Probleme

def vertices_degree(vertices, edges):
    """
    Enumerate the degree of each vertex
    :param vertices: table of vertices
    :param edges:  table of edges
    :return: table, for each vertex, its number of incident edges

    Complexity : O(edges size)
    """
    vertices_number = len(vertices)
    edges_number = len(edges)
    vertices_degree = vertices_number * [0]
    for i in range(edges_number):
        vertices_degree[edges[i][0]] += 1  # 1 edge = degree increases for the 2 incident vertices
        vertices_degree[edges[i][1]] += 1
    return vertices_degree


def odd_degree_vertices(vertices_degree):
    """
    Enumerate the vertices number with a odd degree
    :param vertices_degree: table, for each vertex, its number of incident edges
    :return: table, for each odd degree vertex, its number
    Complexity : O(vertices size)
    """
    vertices_number = len(vertices_degree)
    odd_degree_vertices = []
    for i in range(vertices_number):
        if vertices_degree[i] % 2 == 1:
            odd_degree_vertices.append(i)
    return odd_degree_vertices


def edges_of_each_vertex(vertices, edges, vertices_degree):
    """
    Sort the incident edges of each vertex
    :param vertices: table of vertices
    :param edges:  table of edges
    :param vertices_degree: table of vertices degree
    :return: table, incident edges number for each vertex
    Complexity: O(edges size)
    """
    vertices_number = len(vertices)
    edges_number = len(edges)
    edges_of_each_vertex = vertices_number * [0]
    count_occurence = vertices_number * [0]
    for i in range(vertices_number):
        edges_of_each_vertex[i] = vertices_degree[i] * [0]
    for j in range(edges_number):
        first_vertex = edges[j][0]
        second_vertex = edges[j][1]
        edges_of_each_vertex[first_vertex][
            count_occurence[first_vertex]] = j  # association of incident edges number to the vertex
        edges_of_each_vertex[second_vertex][count_occurence[second_vertex]] = j
        count_occurence[first_vertex] += 1
        count_occurence[second_vertex] += 1

    return edges_of_each_vertex


def dijkstra_vertex(vertex, edges_of_each_vertex, odd_degree_vertices, edges):
    """
    apply dijkstra's algorithm starting from parameter vertex.
    Calculate distance and predecessor edge between parameter vertex and all other odd degree vertices

    :param vertex: starting point for dijkstra's algorithm
    :param edges_of_each_vertex:
    :param odd_degree_vertices:
    :param edges:
    :return:
    Complexity:
    """
    # TODO estimate complexicity
    vertices_number = len(edges_of_each_vertex)
    # initialisation
    dijkstra_distance = vertices_number * [sys.maxsize]
    dijkstra_distance[vertex] = 0
    dijkstra_predecessor = vertices_number * [0]  # Verifier si pas d'effet de bord
    dijkstra_predecessor[vertex] = -1

    dijkstra_previous_edge = vertices_number * [0]
    dijkstra_previous_edge[vertex] = -1

    heap = []
    heapq.heappush(heap, [0, vertex])
    while len(heap) != 0:
        weight,v = heap[0]
        heapq.heappop(heap)
        for i in range(len(edges_of_each_vertex[v])):
            if edges[edges_of_each_vertex[v][i]][0] == v:
                adjacent_v = edges[edges_of_each_vertex[v][i]][1]
            else:
                adjacent_v = edges[edges_of_each_vertex[v][i]][0]
            weight_successor = weight + edges[edges_of_each_vertex[v][i]][2]
            if weight_successor < dijkstra_distance[adjacent_v]:
                dijkstra_distance[adjacent_v] = weight_successor
                dijkstra_predecessor[adjacent_v] = v   # = v ou bien : edges_of_each_vertex[v][i]
                dijkstra_previous_edge[adjacent_v]= edges_of_each_vertex[v][i]
                # TODO
                heapq.heappush(heap, [weight_successor,adjacent_v])


    return dijkstra_distance, dijkstra_predecessor, dijkstra_previous_edge


def dijkstra_vertices(edges_of_each_vertex, odd_degree_vertices, edges):
    """
    Apply dijkstra's algorithm to all odd degree vertices.

    :param edges_of_each_vertex:
    :param odd_degree_vertices:
    :param edges:
    :return: matrix of distance and predecessor, for all odd degree vertices
    Complexity:
    """
    # TODO estimate complexity
    vertices_number = len(edges_of_each_vertex)
    odd_degree_vertices_number = len(odd_degree_vertices)
    distance_matrix = vertices_number*[sys.maxsize]
    predecessor_matrix = vertices_number*[0]
    previous_edge_matrix = vertices_number*[0]
    # only odd degree vertices are interesting, no need to do dijkstra for all vertices
    for i in range(odd_degree_vertices_number):
        odd_degre_vertex = odd_degree_vertices[i]
        dijkstra_distance, dijkstra_predecessor, dijkstra_previous_edge = dijkstra_vertex(odd_degre_vertex, edges_of_each_vertex, odd_degree_vertices, edges)
        distance_matrix[odd_degre_vertex] = dijkstra_distance
        # problem : matrix should be mirror ?
        predecessor_matrix[odd_degre_vertex] = dijkstra_predecessor
        previous_edge_matrix[odd_degre_vertex] = dijkstra_previous_edge
    return distance_matrix, predecessor_matrix, previous_edge_matrix


def path(predecessor_matrix, previous_edge_matrix, edges, final_vertex, starting_vertex):
    edges_path = []
    vertices_path = []
    edges_path.append(edges[previous_edge_matrix[starting_vertex][final_vertex]])
    vertices_path.append(final_vertex)

    cursor = final_vertex

    while predecessor_matrix[starting_vertex][cursor] != -1:  # starting_vertex
        cursor = predecessor_matrix[starting_vertex][cursor]
        print("cursor", cursor)
        print("edge",previous_edge_matrix[starting_vertex][cursor])
        vertices_path.append(cursor)
        if predecessor_matrix[starting_vertex][cursor] != -1 :
            edges_path.append(edges[previous_edge_matrix[starting_vertex][cursor]])
    return vertices_path, edges_path


def complete_graph(odd_degree_vertices, distance_matrix):
    size = len(odd_degree_vertices)
    new_dist_matrix = size*[0]
    for i in range (size):
        line = size*[0]
        for j in range(size):
            line[j] = distance_matrix[odd_degree_vertices[i]][odd_degree_vertices[j]]
        new_dist_matrix[i] = line
    return new_dist_matrix


def minlist(list, treated_vertices, x):
    # TODO improve algorithm, currently polynomial
    size = len(list)
    min = sys.maxsize
    y = -1
    for i in range(size): # i!= index because the edge of distance=0 must not be selected
        if i != x and not treated_vertices[i] and list[i] < min:
            min = list[i]
            y = i
    if y == -1:
        print("ERROR, no edge selected")
    else:
        return y


def greedy_algorithm(odd_degree_vertices_dist_matrix):  # odd_degree_vertices_dist_matrix = new_dist_matrix
    size = len(odd_degree_vertices_dist_matrix)  # size is an even number
    # for i in range(size):
    #     print(odd_degree_vertices_dist_matrix[i])
    treated_vertices = size * [False]
    edges_size = size / 2
    edges_selected = []
    x = 0
    while len(edges_selected) != edges_size and x<size:   #and x < size
        if not treated_vertices[x]:
            y = minlist(odd_degree_vertices_dist_matrix[x], treated_vertices, x)
            edges_selected.append([x, y])
            treated_vertices[x] = True
            treated_vertices[y] = True
            # print(x, y, edges_selected)
        x += 1

    return edges_selected  # = perfect matching





def scheduling(distance_matrix, predecessor_matrix, odd_degree_vertices, edges):
    odd_degree_vertices_number = len(odd_degree_vertices)


    perfect_matching = [0]
    return perfect_matching




start = time.time()

vertices, edges = parse_file("paris_map.txt") # sys.argv[1]

print("\n ZONE DE TEST \n")

"""EXEMPLE 2"""

print("\n exemple 2")
verticesC =[0,1,2,3,4]
edgesC = [[0,1,3],[0,2,10],[1,2,2],[3,2,2],[1,3,4],[1,4,1],[4,3,7]]
print("verticesC", verticesC)
print("edgesC", edgesC)
vertices_degree_C = vertices_degree(verticesC,edgesC)
print("vertices_degree ", vertices_degree_C)
odd_degree_vertices_C = odd_degree_vertices(vertices_degree_C)
print("odd_degree_vertices_C ", odd_degree_vertices_C)
edges_of_each_vortex_C = edges_of_each_vertex(verticesC, edgesC, vertices_degree_C)
print("edges_of_each_vortex_C ", edges_of_each_vortex_C)
distance_matrix_C, predecessor_matrix_C, previous_edge_matrix_C = dijkstra_vertices(edges_of_each_vortex_C, odd_degree_vertices_C, edgesC)
print("distance_matrix_C ", distance_matrix_C)
print("predecessor_matrix_C", predecessor_matrix_C)
print("previous_edge_matrix_C", previous_edge_matrix_C)
new_dist_matrix_C = complete_graph(odd_degree_vertices_C, distance_matrix_C)
print("new_dist_matrix_C",new_dist_matrix_C)
perfect_matching_C = greedy_algorithm(new_dist_matrix_C)
print("perfect matching_C", perfect_matching_C)

# vertices_path_C34, edges_path_C34 = path(predecessor_matrix_C, previous_edge_matrix_C, edgesC, 2,3)
# print("vertices_path_C34", vertices_path_C34)
# print("edges_path_C34", edges_path_C34)

# vertices_path_C43, edges_path_C43 = path(predecessor_matrix_C, previous_edge_matrix_C, edgesC, 3,2)
# print("vertices_path_C43", vertices_path_C43)
# print("edges_path_C43", edges_path_C43)

vertices_path_C43bis, edges_path_C43bis = path(predecessor_matrix_C, previous_edge_matrix_C, edgesC, 3,2)
print("vertices_path_C43bis", vertices_path_C43bis)
print("edges_path_C43bis", edges_path_C43bis)


"""EXEMPLE 3"""
print("\n exemple 3")
verticesD = [0,1,2,3,4,5,6]
edgesD = [[0,2,3],[1,2,7],[1,5,1], [1,4,2],[4,5,4],[2,4,8],[2,6,6],[2,3,4],[3,6,1],[6,4,5],[4,3,3]]
print("verticesD", verticesD)
print("edgesD", edgesD)
vertices_degree_D = vertices_degree(verticesD, edgesD)
print("vertices_degree_D", vertices_degree_D)
odd_degree_vertices_D = odd_degree_vertices(vertices_degree_D)
print("odd_degree_vertices_D", odd_degree_vertices_D)
edges_of_each_vortex_D = edges_of_each_vertex(verticesD,edgesD,vertices_degree_D)
print("edges_of_each_vortex_D",edges_of_each_vortex_D)
distance_matrix_D, predecessor_matrix_D, previous_edge_matrix_D = dijkstra_vertices(edges_of_each_vortex_D,odd_degree_vertices_D,edgesD)
print("distance_matrix_D", distance_matrix_D)
print("predecessor_matrix_D", predecessor_matrix_D)
print("previous_edge_matrix_D", previous_edge_matrix_D)
new_dist_matrix_D = complete_graph(odd_degree_vertices_D, distance_matrix_D)
print("new_dist_matrix_D", new_dist_matrix_D)

perfect_matching_D = greedy_algorithm(new_dist_matrix_D)
print("perfect_matching_D", perfect_matching_D)

# vertices_path_D_31, edges_path_D31 = path(predecessor_matrix_D, previous_edge_matrix_D, edgesD, 3, 1)
# print("vertices_path_D_31", vertices_path_D_31)
# print("edges_path_D31", edges_path_D31)

vertices_path_D_31bis, edges_path_D31bis = path(predecessor_matrix_D, previous_edge_matrix_D, edgesD, 3, 1)
print("vertices_path_D_31bis", vertices_path_D_31bis)
print("edges_path_D31bis", edges_path_D31bis)

vertices_path_D60, edges_path_D60 = path(predecessor_matrix_D, previous_edge_matrix_D, edgesD, 6, 0)
print("vertices_path_D60", vertices_path_D60)
print("edges_path_D60", edges_path_D60)


vertices_path_D06, edges_path_D06 = path(predecessor_matrix_D, previous_edge_matrix_D, edgesD, 0, 6)
print("vertices_path_D06", vertices_path_D06)
print("edges_path_D06", edges_path_D06)


""" Exemple 4 """
print("\n exemple 4")


#TODO EXEMPLE PRINCIPALE
print("\n exemple principale")



# vertices_degree_A = vertices_degree(vertices, edges)
# print("vertices_degree : ", vertices_degree_A)
#
# odd_degree_vertices_A = odd_degree_vertices(vertices_degree_A)
# print("odd_degree_vertices : ", odd_degree_vertices_A)
#
# edges_of_each_vertex_A = edges_of_each_vertex(vertices, edges, vertices_degree_A)
# print("edges of each vertex : ", edges_of_each_vertex_A)
#
# print("len(odd_degree_vertices_A) : ", len(odd_degree_vertices_A))


# vertex_A = 14
# print(edges_of_each_vertex_A[14])
# print(edges[9668])
# print(edges[10572])
# print(edges[17549])
#
#
# dijkstra_distance_A, dijkstra_predecessor, dijkstra_previous_edge = dijkstra_vertex(vertex_A, edges_of_each_vertex_A, odd_degree_vertices_A,
#                                                             edges)
# print("\n taille dijkstra_distance1", len(dijkstra_distance_A))
# print("taille dijkstra_predecessor", len(dijkstra_predecessor))
# print(dijkstra_distance_A)
# print(dijkstra_predecessor)
# print("distance à vertex_A: ", dijkstra_distance_A[vertex_A])
# print("predecesseur à vertex A ? :", dijkstra_predecessor[vertex_A])


#DIJKSTRA COMPLET


# distance_matrix, predecessor_matrix = dijkstra_vertices(edges_of_each_vertex_A, odd_degree_vertices_A, edges)
# print("distance_matrix")
# print(distance_matrix)
# print("predecessor_matrix")
# print(predecessor_matrix)


end = time.time()
timelapse = end - start
print(timelapse)

print("fin")

# vertices, edges = parse_file("paris_map.txt") #sys.argv[1]
#plot_sample(vertices, edges, 50000) #sys.argv[2]

