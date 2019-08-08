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
    vertices_number = len(edges_of_each_vertex)
    # initialisation
    dijkstra_distance = vertices_number * [sys.maxsize]
    dijkstra_distance[vertex] = 0
    dijkstra_predecessor = vertices_number * [0]  # Verifier si pas d'effet de bord
    dijkstra_predecessor[vertex] = -1
    heap = []
    heapq.heappush(heap,[0, vertex])
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
                dijkstra_predecessor[adjacent_v] = v
                heapq.heappush(heap, [weight_successor,adjacent_v])


    return dijkstra_distance, dijkstra_predecessor


def dijkstra_vertices(edges_of_each_vertex, odd_degree_vertices, edges):
    vertices_number = len(edges_of_each_vertex)
    odd_degree_vertices_number = len(odd_degree_vertices)
    distance_matrix = vertices_number*[sys.maxsize]
    predecessor_matrix = vertices_number*[0]
    # only odd degree vertices are interesting, no need to do dijkstra for all vertices
    for i in range(odd_degree_vertices_number):
        odd_degre_vertex = odd_degree_vertices[i]
        dijkstra_distance, dijkstra_predecessor = dijkstra_vertex(odd_degre_vertex, edges_of_each_vertex, odd_degree_vertices, edges)
        distance_matrix[odd_degre_vertex] = dijkstra_distance
        # problem : matrix should be mirror ?
        predecessor_matrix[odd_degre_vertex] = dijkstra_predecessor
    return distance_matrix, predecessor_matrix



start = time.time()

vertices, edges = parse_file("paris_map.txt") # sys.argv[1]

print("\n ZONE DE TEST \n")
vertices_degree_A = vertices_degree(vertices, edges)
print("vertices_degree : ", vertices_degree_A)

odd_degree_vertices_A = odd_degree_vertices(vertices_degree_A)
print("odd_degree_vertices : ", odd_degree_vertices_A)

edges_of_each_vertex_A = edges_of_each_vertex(vertices, edges, vertices_degree_A)
print("edges of each vertex : ", edges_of_each_vertex_A)

print("len(odd_degree_vertices_A) : ", len(odd_degree_vertices_A))

# TODO AUTRE EXEMPLE
print("\n Nouvel exemple simple")
verticesC =[0,1,2,3,4]
edgesC = [[0,1,3],[0,2,10],[1,2,2],[3,2,2],[1,3,4],[1,4,1],[4,3,7]]
vertices_degree_C = vertices_degree(verticesC,edgesC)
print("vertices_degree ", vertices_degree_C)
odd_degree_vertices_C = odd_degree_vertices(vertices_degree_C)
print("odd_degree_vertices_C ", odd_degree_vertices_C)
edges_of_each_vortex_C = edges_of_each_vertex(verticesC,edgesC,vertices_degree_C)
print("edges_of_each_vortex_C ", edges_of_each_vortex_C)
distance_matrix_C, predecessor_matrix_C = dijkstra_vertices(edges_of_each_vortex_C,odd_degree_vertices_C,edgesC)
print("distance_matrix_C ", distance_matrix_C)
print("predecessor_matrix_C", predecessor_matrix_C)


#TODO EXEMPLE PRINCIPALE

vertex_A = 14
print(edges_of_each_vertex_A[14])
print(edges[9668])
print(edges[10572])
print(edges[17549])


dijkstra_distance_A, dijkstra_predecessor = dijkstra_vertex(vertex_A, edges_of_each_vertex_A, odd_degree_vertices_A,
                                                            edges)
print(dijkstra_distance_A)
print(dijkstra_predecessor)
time2b = time.time()





#DIJKSTRA COMPLET
time2a = time.time()

# distance_matrix, predecessor_matrix = dijkstra_vertices(edges_of_each_vertex_A, odd_degree_vertices_A, edges)
# print("distance_matrix")
# print(distance_matrix)
# print("predecessor_matrix")
# print(predecessor_matrix)

print("rapide: ", time2b-time2a)

# informations générales sur la modélisation du problème
# print("len(edges) : " , len(edges))
# print(" len(vertices) : ", len(vertices))
# print("\n vertices, nb of elements : ", len(vertices[0]))
# for i in range(2):
#     print(vertices[i])
# print("\n edges, nb of elements : ", len(edges[0]))
# for i in range(2):
#     print(edges[i])



# test dijkstra_vertex
end = time.time()
timelapse = end - start
print(timelapse)

print("fin")

# vertices, edges = parse_file("paris_map.txt") #sys.argv[1]
# plot_sample(vertices, edges, 50000) #sys.argv[2]
