from typing import List

import matplotlib.pyplot as pyplt
from random import randint
import sys
import time
import heapq
import copy

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


# Problem Solving

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
        # 1 edge = increased degree for the 2 incident vertices
        vertices_degree[edges[i][0]] += 1  
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
                heapq.heappush(heap, [weight_successor, adjacent_v])
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
    return edges_selected  # = perfect matching #Attention les indices de sommets sélectionnés sont ceux du graphe de sommets de degré impair uniquement
# il faut donc faire attention à ensuite convertir en ceux du graphe total original


def greedy_algorithm2(odd_degree_vertices_dist_matrix, odd_degree_vertices):  # odd_degree_vertices_dist_matrix = new_dist_matrix
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
            edges_selected.append([odd_degree_vertices[x], odd_degree_vertices[y]])
            treated_vertices[x] = True
            treated_vertices[y] = True
        x += 1
    return edges_selected  # = correct perfect matching, with index matching those in the general graph


def path(predecessor_matrix, previous_edge_matrix, edges, final_vertex, starting_vertex): #path() can't be applied with a even degree vertex
    # TODO is vertices_path useful for what the method is being used for ?
    edges_path = []
    vertices_path = []
    edges_path.append(edges[previous_edge_matrix[starting_vertex][final_vertex]])
    vertices_path.append(final_vertex)
    cursor = final_vertex
    while predecessor_matrix[starting_vertex][cursor] != -1:  # starting_vertex
        cursor = predecessor_matrix[starting_vertex][cursor]
        # print("cursor", cursor)
        # print("edge",previous_edge_matrix[starting_vertex][cursor])
        vertices_path.append(cursor)
        if predecessor_matrix[starting_vertex][cursor] != -1 :
            edges_path.append(edges[previous_edge_matrix[starting_vertex][cursor]])
    return vertices_path, edges_path



def graph_with_added_edges(perfect_matching, predecessor_matrix, previous_edge_matrix, edges):
    size = len(perfect_matching)
    new_edges = copy.copy(edges)
    for i in range(size):
        vertices_path, edges_path = path(predecessor_matrix, previous_edge_matrix, edges, perfect_matching[i][0], perfect_matching[i][1])
        for j in range(len(edges_path)):
            new_edges.append(edges_path[j])
    return new_edges


def totaldistance(new_edges):
    size = len(new_edges)
    d = 0
    for i in range(size):
        d += new_edges[i][2]
    return d


def available_edge_index(available_edges):  # available_edges tableau de boolean
    i = 0
    size = len(available_edges)
    bool = True
    while i< size and not available_edges[i]:
        i += 1
    if i == size:
        bool = False # "Toutes les arêtes ont été prises"
    return bool, i  # i = available edge index


def following_edge(vertex, edges, edges_of_each_vertex, available_edges):
    found = False
    i = 0
    size = len(edges_of_each_vertex[vertex])
    while i < size and not found:
        current_edge = edges_of_each_vertex[vertex][i]
        if available_edges[current_edge]:
            found = True
        else:
            i += 1

    return found, current_edge


def cycle(starting_vertex, edges, edges_of_each_vertex,available_edges):
    cycle = []
    current_vertex = starting_vertex
    boolean_edges, selected_edge = following_edge(current_vertex, edges, edges_of_each_vertex, available_edges)
    while boolean_edges:
        cycle.append(selected_edge)
        available_edges[selected_edge] = False
        if edges[selected_edge][0] == current_vertex:
            current_vertex = edges[selected_edge][1]
        else:
            current_vertex = edges[selected_edge][0]
        boolean_edges, selected_edge = following_edge(current_vertex, edges, edges_of_each_vertex, available_edges)
    return cycle, available_edges

def cycle_list(edges, edges_of_each_vertex):
    size = len(edges)
    available_edges = size * [True]
    cycle_list = []
    bool = True
    starting_vertex = 0
    compteur = 0 #TODO
    while bool and compteur < 5:
        cycle2, available_edges = cycle(starting_vertex, edges, edges_of_each_vertex, available_edges) #effet de bord sur available_edges
#        print("cycle2", cycle2)
#        print("available_edges", available_edges)
        cycle_list.append(cycle2)
#        print("cycle_list", cycle_list)
        bool, available_edge_index2 = available_edge_index(available_edges)
#        print("bool", bool)
#        print("available_edge_index2", available_edge_index2)
        if bool:
            starting_vertex = edges[available_edge_index2][0]
#            print("starting_vertex", starting_vertex)
        compteur += 1
#        print(cycle2, available_edges)
#        print(cycle_list)
    return cycle_list






#def scheduling(edges, edges_of_each_vertex): #edges = new_edges and edges_of_each_vertex = new_edges_of_each_vertex
#    cycle_list = cycle_list(edges, edges_of_each_vertex)
#    eulerian_cycle = eulerian_cycle(cycle_list)
#    return eulerian_cycle


def solving(vertices, edges):
    vertices_deg = vertices_degree(vertices, edges)
    odd_deg_vertices = odd_degree_vertices(vertices_deg)
    edges_per_vertex = edges_of_each_vertex(vertices_deg, edges, vertices_deg)
    distance_matrix, predecessor_matrix, previous_edge_matrix = dijkstra_vertices(edges_per_vertex, odd_deg_vertices, edges)
    new_dist_matrix = complete_graph(odd_deg_vertices, distance_matrix)
    correct_perfect_matching = greedy_algorithm2(new_dist_matrix, odd_deg_vertices)
    new_edges = graph_with_added_edges(correct_perfect_matching, predecessor_matrix, previous_edge_matrix, edges)
    new_vertice_deg = vertices_degree(vertices, new_edges)
    new_edges_of_each_vertex = edges_of_each_vertex(vertices, new_edges, new_vertice_deg)
    
    
    return new_edges_of_each_vertex #"odd", odd_deg_vertices,"edges", edges_per_vertex, "dist", distance_matrix, "pred", predecessor_matrix, "prev", previous_edge_matrix
    
    

start = time.time()


vertices, edges = parse_file("paris_map.txt") # sys.argv[1]

print("\n ZONE DE TEST \n")

"""EXEMPLE 2"""

print("\n exemple 2")
c_vertices =[0,1,2,3,4]
c_edges = [[0,1,3],[0,2,10],[1,2,2],[3,2,2],[1,3,4],[1,4,1],[4,3,7]]
c_vertices_degree = vertices_degree(c_vertices,c_edges)
c_odd__degree_vertices = odd_degree_vertices(c_vertices_degree)
c_edges_of_each_vortex = edges_of_each_vertex(c_vertices, c_edges, c_vertices_degree)
c_distance_matrix, c_predecessor_matrix, c_previous_edge_matrix = dijkstra_vertices(c_edges_of_each_vortex, c_odd__degree_vertices, c_edges)
c_new_dist_matrix = complete_graph(c_odd__degree_vertices, c_distance_matrix)
c_correct_perfect_matching = greedy_algorithm2(c_new_dist_matrix,c_odd__degree_vertices)
c_new_edges = graph_with_added_edges(c_correct_perfect_matching, c_predecessor_matrix, c_previous_edge_matrix, c_edges)
c_tot_dist = totaldistance(c_new_edges)
c_new_vertices_degree = vertices_degree(c_vertices, c_new_edges)
c_new_edges_of_each_vertex = edges_of_each_vertex(c_vertices, c_new_edges, c_new_vertices_degree)


c_solution= solving(c_vertices, c_edges)
print(c_solution)

# TODO ici on applique le problème à l'exemple C


#available_c_edges = len(c_edges) * [True]
#cycle_listC = []
#cycleC1, available_c_edges1 = cycle(0,c_new_edges, c_new_edges_of_each_vertex, available_c_edges)
#cycle_listC.append(cycleC1)
#boolC1, following_edgeC1 = available_edge_index(available_c_edges1)
#following_vertex_C1 = c_edges[following_edgeC1][0]
#print("cycleC1", cycleC1, available_c_edges1)
#print("cycle_listC", cycle_listC)
#print("boolC1", boolC1)
#print("follow i:1",following_edgeC1, following_vertex_C1)
#cycleC2, available_c_edges2 = cycle(following_vertex_C1, c_new_edges, c_new_edges_of_each_vertex, available_c_edges1)
#cycle_listC.append(cycleC2)
#boolC2, following_edgeC2 = available_edge_index(available_c_edges2)
#print("cycleC2", cycleC2, available_c_edges2)
#print("cycle_listC",cycle_listC)
#print("boolC2", boolC2)


#
#cycle_listC = cycle_list(c_edges, c_edges_of_each_vortex)
#print("cycle_listC", cycle_listC)


# ne marche pas comme il le faudrait




"""EXEMPLE 3"""
print("\n exemple 3")
d_vertices = [0,1,2,3,4,5,6]
d_edges = [[0,2,3],[1,2,7],[1,5,1], [1,4,2],[4,5,4],[2,4,8],[2,6,6],[2,3,4],[3,6,1],[6,4,5],[4,3,3]]


d_solution = solving(d_vertices, d_edges)
print(d_solution)



# TODO ici on applique le problème à l'exemple D
#available_d_edges = len(d_edges) * [True]
#cycleD1, available_d_edges1 = cycle(0,d_edges, new_edges_of_each_vertex_D, available_d_edges)
#print("d_edges[9][0]", d_edges[9][0])
#cycleD2, available_d_edges2 = cycle(6, d_edges, new_edges_of_each_vertex_D, available_d_edges1)
# print("cycleD1", cycleD1, available_d_edges1)
# print("cycleD2", cycleD2, available_d_edges2)



"""Exemple 4"""
print("\n Exemple 4")
#verticesE = [0,1,2,3,4,5,6,7]
#edgesE = [[0,1,1], [1,2,1], [2,3,1], [3,4,1], [2,5,1], [4,6,1], [6,7,1]]
#vertices_degree_E = vertices_degree(verticesE, edgesE)
#odd_degree_vertices_E = odd_degree_vertices(vertices_degree_E)
#edges_of_each_vortex_E = edges_of_each_vertex(verticesE,edgesE,vertices_degree_E)
#distance_matrix_E, predecessor_matrix_E, previous_edge_matrix_E = dijkstra_vertices(edges_of_each_vortex_E,odd_degree_vertices_E,edgesE)
#new_dist_matrix_E = complete_graph(odd_degree_vertices_E, distance_matrix_E)
#perfect_matching_E = greedy_algorithm(new_dist_matrix_E)
#correct_perfect_matching_E = greedy_algorithm2(new_dist_matrix_E, odd_degree_vertices_E)
#new_edges_E = graph_with_added_edges(correct_perfect_matching_E, predecessor_matrix_E, previous_edge_matrix_E, edgesE)
#totdistE = totaldistance(new_edges_E)
#new_vertices_degree_E = vertices_degree(verticesE, new_edges_E)
#new_edges_of_each_vertex_E = edges_of_each_vertex(verticesE, new_edges_E, new_vertices_degree_E)

#print("edgesE",len(edgesE), edgesE)
#print("odd_degree_vertices_E", odd_degree_vertices_E)
#print("perfect_matching_E", perfect_matching_E)
#print("correct_perfect_matching_E", correct_perfect_matching_E)
#print("new_edges_E",len(new_edges_E), new_edges_E)
#print("totdistE", totdistE)

# TODO ici on applique le problème à l'exemple E
#available_edgesE = len(edgesE) * [True]
#cycleE1, available_edgesE1 = cycle(0,edgesE, new_edges_of_each_vertex_E, available_edgesE)

# print("cycleE1", cycleE1, available_edgesE1)



""" Exemple Principal"""

#TODO EXEMPLE PRINCIPALE

print("\n exemple principale")

# vertices_degree_A = vertices_degree(vertices, edges)
# odd_degree_vertices_A = odd_degree_vertices(vertices_degree_A)
# edges_of_each_vertex_A = edges_of_each_vertex(vertices, edges, vertices_degree_A)
# distance_matrix_A, predecessor_matrix_A, previous_edge_matrix_A = dijkstra_vertices(edges_of_each_vertex_A, odd_degree_vertices_A, edges)
# new_dist_matrix_A = complete_graph(odd_degree_vertices_A, distance_matrix_A)
#
# correct_perfect_matching_A = greedy_algorithm2(new_dist_matrix_A, odd_degree_vertices_A)
#
# new_edges_A = graph_with_added_edges(correct_perfect_matching_A, predecessor_matrix_A,previous_edge_matrix_A,edges)
# totdist_A = totaldistance(new_edges_A)
#
# print("totdist_A", totdist_A)










# print("vertices_degree : ", vertices_degree_A)
# print("odd_degree_vertices : ", odd_degree_vertices_A)
#
# print("edges of each vertex : ", edges_of_each_vertex_A)
#
# print("len(odd_degree_vertices_A) : ", len(odd_degree_vertices_A))
#
# print("new_dist_matrix_A", new_dist_matrix_A)

# print("correct_perfect_matching_A", correct_perfect_matching_A)



# test sur sommet numero 14 :

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




end = time.time()
timelapse = end - start
print(timelapse)

print("fin")

# vertices, edges = parse_file("paris_map.txt") #sys.argv[1]
# plot_sample(vertices, edges, 50000) #sys.argv[2]

