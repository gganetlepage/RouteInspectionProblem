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


def vertices_of_a_edge_list(cycle, new_edges):
    size = len(cycle)
    vertices_list = []
    for i in range(size): #create a list indicating the vertices and the edge concerned
        edge_index = cycle[i]
        edge = new_edges[edge_index]
        vertices_list.append([edge[0], edge_index, i]) # i is the edge index in the cycle
        vertices_list.append([edge[1], edge_index, i])
    return vertices_list


""""
Next step: Create the vertices list associated to a cycle list
"""

def partition(vertices_list, low, high):
    pivot = vertices_list[high][0] # element used for comparaison
    i = low-1 # index used to swap
    for j in range(low, high):
        if vertices_list[j][0]< pivot:
            i += 1  
            vertices_list[i], vertices_list[j] = vertices_list[j], vertices_list[i]
    vertices_list[i+1], vertices_list[high] = vertices_list[high], vertices_list[i+1]
    return i+1 #index of pi


def quicksort(vertices_list, low, high):
    if high <= low:
        return "Error in indexes"
    else:
        
        pi = partition(vertices_list, low, high) # pi is partitioning index, verticesl_list[pi] is now at the right place
        quicksort(vertices_list, low, pi-1) # before pi
        quicksort(vertices_list, pi+1, high) # after pi
    
def reducing_list(vertices_list): # each vertex is occuring twice in the vertices_list, we only it once it fasten the method
    reduced_list = []
    size = len(vertices_list)
    steps = int(size / 2)
    for i in range(steps):
        reduced_list.append(vertices_list[2*i])
    return reduced_list
    

def vertices_cycle(cycle_list, new_edges):
    vertices_list = []
    for x in cycle_list:
        y = vertices_of_a_edge_list(x, new_edges)
        quicksort(y, 0, len(y)-1)
        reduced_list = reducing_list(y)
        vertices_list.append(reduced_list)
    return vertices_list
    
"""
Next step: with 2 cycles, being able to determine where to insert the first one into the second one
"""

def vertex_in_list(vertex, vertices_list): # is the vertex present in the list? return a bool
    present = False
    i = 0
    size = len(vertices_list)
    while not present and i < size:
        if vertices_list[i][0] == vertex: #[0] due to working with list of 2 elements
            present = True
        else:
            i += 1
    return present, i #bool and vertex index

def comparison_vertices_list(vertices_list1, vertices_list2):
    present = False
    i_list2 = 0 # index in list2 of a matching vertex with list1
    i = 0
    size = len(vertices_list1)
    while not present and i< size:
        present, i_list2 = vertex_in_list(vertices_list1[i][0], vertices_list2) #[0] due to working with list of 2 elements
        if not present:
            i+=1
    return present, i, i_list2 # bool and possible vertices indices 


def entering_or_outgoing(i, reduced_list, cycle_list ,new_edges): # True = entering 
    vertex = reduced_list[i][0]
#    edge = reduced_list[i][1]
    index_in_cycle_list = reduced_list[i][2]
#    edge_index = cycle_list[index_in_cycle_list]
    precedent_edge_index = cycle_list[index_in_cycle_list - 1]
    if new_edges[precedent_edge_index][0] == vertex or new_edges[precedent_edge_index][1] == vertex:
        return False # outgoing
    else:
        return True # entering


def real_order_of_cycle(precedent, cycle_list,edge_index_in_cycle): # edge_index_in_cycle = reduced_list[index][2]
    correct_order_cycle = []
    size = len(cycle_list)
    if precedent: # then the edge must be put at the ultimate end, the cycle first leaves the concertex vertex then go back into it
        for i in range(edge_index_in_cycle +1, size):
            correct_order_cycle.append(cycle_list[i])
        for i in range(edge_index_in_cycle+1):
            correct_order_cycle.append(cycle_list[i])
    else:
#        for i in range(edge_index_in_cycle, edge_index_in_cycle - size, -1): 
        for i in range(edge_index_in_cycle - size, edge_index_in_cycle): 
            correct_order_cycle.append(cycle_list[i])
    return correct_order_cycle

    
def list2_into_list1(cycle_list1, real_order_cycle_list2, precedent_list1, edge_index_in_cycle1):
    if precedent_list1: 
        # then list2 is added after the edge
        #It is not possible for the edge_index_in_cycle1 to be the last element of the cycle_list1, that would mean the list2 can also be added after the first element
        for x in real_order_cycle_list2:
            # list2 is added in reverses
            cycle_list1.insert(edge_index_in_cycle1+1, x)
    else:
        for x in real_order_cycle_list2:
            cycle_list1.insert(edge_index_in_cycle1, x)
    return cycle_list1
    

#delete list1? après la mathode list2_into_list1
        
    
# this method is not finished 
def add_list_into_list(cycle_list1, cycle_list2, vertices_list1, vertices_list2, new_edges, i_list1, i_list2):
#    vertices_list1 = vertices_cycle(cycle_list1, new_edges) vertices_list1 is already reduced_list1 
#    vertices_list2 = vertices_cycle(cycle_list2, new_edges) vertices_list2 is already reduced_list2
    edge_index_in_cycle1 = vertices_list1[i_list1][2] # TODO why not use directly i_list or i_list2 ?
    edge_index_in_cycle2 = vertices_list2[i_list2][2]
    precedent_list1 = entering_or_outgoing(i_list1, vertices_list1, cycle_list1, new_edges)
    precedent_list2 = entering_or_outgoing(i_list2, vertices_list2, cycle_list2, new_edges)
    # arbitrary choice to insert list2 into list1, as list1 is generally longer on tested cases
    correct_order_cycle_list2 = real_order_of_cycle(precedent_list2, cycle_list2, edge_index_in_cycle2)
    concatenated_list = list2_into_list1(cycle_list1, correct_order_cycle_list2, precedent_list1, edge_index_in_cycle1)
    return concatenated_list


   
    

#try to treat every case possible

def put_all_cycles_into_one3(cycle_list, vertices_list, new_edges):
    #vertices_list is reduced_list
    
    #loop WHILE here to iterate on every additional cycle
#    if len(vertices_list) >= 2:
    if len(cycle_list) >= 2:
        vertices_list1 = vertices_list[0]
        vertices_list2 = vertices_list[1]
        present, i_list1, i_list2 = comparison_vertices_list(vertices_list1, vertices_list2)
        if present:
            concatenated_list = add_list_into_list(cycle_list[0], cycle_list[1], vertices_list1, vertices_list2, new_edges, i_list1, i_list2)
            #cycle _list is updated by add_list_into_list()
            # so the second element of cycle_list needs to be deleted after that
            # but nothing has been done concerning vertices_list, which needs to be calculated again ?
            return concatenated_list
        else:
            return "there isn't a match there for the moment" # modification à faire: créer un compteur, itérer tant que l'on n'a pas present = true
    else:
        return cycle_list[0] #there is only one cycle or none
    

def put_all_cycles_into_one2(cycle_list, vertices_list, new_edges):
    #tentative récursive ?
    if len(cycle_list)<= 1:
        if len(cycle_list) == 1:
            return cycle_list[0]
        else:
            return cycle_list
    else:
        vertices_list1 = vertices_list[0] #NNONN, vertices_list[0] va être mise à jour
        count = 1
        count_error = 0 # TODO in case the function is not working properly 
        present, i_list1, i_list2 = comparison_vertices_list(vertices_list1, vertices_list[count])
        size = len(vertices_list)
        while not present and count < size and count_error <10:
            count += 1
            present, i_list1, i_list2 = comparison_vertices_list(vertices_list1, vertices_list[count])
            count_error += 1
        if present:
            concatenated_list = add_list_into_list(cycle_list[0], cycle_list[count], vertices_list1, vertices_list[count], new_edges, i_list1, i_list2)
            deleted_cycle = cycle_list[count]
            # vertices_list must be updated now ! does ne_edges needs to be too ?
            vertices_list = vertices_cycle(cycle_list, new_edges)
            return True
    return True


def put_all_cycles_into_one(cycle_list, vertices_list, new_edges):
    count_error = 0 # TODO
    while len(cycle_list) > 1 and  count_error < 10:
        count = 1
        count_error += 1
        present, i_list1, i_list2 = comparison_vertices_list(vertices_list[0], vertices_list[count])
        count_error2 = 0 #TODO
        while not present:
            count += 1
            count_error2 += 1
            present, i_list1, i_list2 = comparison_vertices_list(vertices_list[0], vertices_list[count])
        # a vertice is present on both cycles (fro cycle 0 and cycle number "count")
        
#        concatenated_list = add_list_into_list(cycle_list[0], cycle_list[count], vertices_list[0], vertices_list[count], new_edges, i_list1, i_list2)
        add_list_into_list(cycle_list[0], cycle_list[count], vertices_list[0], vertices_list[count], new_edges, i_list1, i_list2)
#        deleted_cycle = cycle_list.pop(count)
        cycle_list.pop(count)
        vertices_list = vertices_cycle(cycle_list, new_edges)
    return cycle_list[0]

        
            


      
            
    

"""
Reflexions :
    le return concatenated_list n'est pas utilie lors des différentes itérations, il sert pour le résultat final uniquement
    une fois la méthode appelée, cycle_list[0] est modifiée, et est constituée de son cycle intiale ainsi que du cycle de cycle_list[1]
    PAR CONTRE, pour reitérer la méthode, il faut actualiser vertices_list !!
    Peut-on le faire récursivement ?
    à la limte le faire avec une boucle tant que. Du genre :
        whle (len(cycle_list) >= 2 ):
            concatenated_list = put_all_cycles_into_one
            cycle_list[1].delete()  ????
            actualiser vertices_list # ATTENTION ici vertices_list correspond à quoi ? vertices_list classique, reduced_list ou autre ?
            
    Quelle différence entre add_list_into_list et put_all_cycles_into_one
    
    Gestion des cas où il n'y a pas de 
    
"""


    
    
         
    
    
    
    
#vetices_of_an_edge

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
    #working on list of edges:    
    cycle_list1 = cycle_list(new_edges, new_edges_of_each_vertex)
    vertices_list = vertices_cycle(cycle_list1, new_edges)
    # identifying matching vertices, DANGER how to behave if only one element in the list ?
    
    #concatenated_list = put_all_cycles_into_one(cycle_list1, vertices_list, new_edges)
    
    final_cycle = put_all_cycles_into_one(cycle_list1, vertices_list, new_edges)
    
    
    
#    if len(vertices_list) >= 2:
#        present, i, j = comparison_vertices_list(vertices_list[0], vertices_list[1])
#        if entering_or_outgoing(i, vertices_list[0],cycle_list1[0], new_edges):
#            return True
#        return vertices_list, present, i,j
#    
    
    

    return final_cycle
    
    

start = time.time()




vertices, edges = parse_file("paris_map.txt") # sys.argv[1]

print("\n ZONE DE TEST \n")


"""EXEMPLE 2"""

print("\n exemple 2")
c_vertices =[0,1,2,3,4]
c_edges = [[0,1,3],[0,2,10],[1,2,2],[3,2,2],[1,3,4],[1,4,1],[4,3,7]]

c_solution= solving(c_vertices, c_edges)
print(c_solution)


"""EXEMPLE 3"""
print("\n exemple 3")
d_vertices = [0,1,2,3,4,5,6]
d_edges = [[0,2,3],[1,2,7],[1,5,1], [1,4,2],[4,5,4],[2,4,8],[2,6,6],[2,3,4],[3,6,1],[6,4,5],[4,3,3]]

d_solution = solving(d_vertices, d_edges)
print(d_solution)



"""Exemple 4"""
print("\n Exemple 4")
e_vertices = [0,1,2,3,4,5,6,7]
e_edges = [[0,1,1], [1,2,1], [2,3,1], [3,4,1], [2,5,1], [4,6,1], [6,7,1]]
e_solution = solving(e_vertices, e_edges)
print(e_solution)






#vertices_degree_E = vertices_degree(e_vertices, e_edges)
#odd_degree_vertices_E = odd_degree_vertices(vertices_degree_E)
#edges_of_each_vortex_E = edges_of_each_vertex(e_vertices,e_edges,vertices_degree_E)
#distance_matrix_E, predecessor_matrix_E, previous_edge_matrix_E = dijkstra_vertices(edges_of_each_vortex_E,odd_degree_vertices_E,e_edges)
#new_dist_matrix_E = complete_graph(odd_degree_vertices_E, distance_matrix_E)
#perfect_matching_E = greedy_algorithm(new_dist_matrix_E)
#correct_perfect_matching_E = greedy_algorithm2(new_dist_matrix_E, odd_degree_vertices_E)
#new_edges_E = graph_with_added_edges(correct_perfect_matching_E, predecessor_matrix_E, previous_edge_matrix_E, e_edges)
#totdistE = totaldistance(new_edges_E)
#new_vertices_degree_E = vertices_degree(e_vertices, new_edges_E)
#new_edges_of_each_vertex_E = edges_of_each_vertex(e_vertices, new_edges_E, new_vertices_degree_E)

#print("e_edges",len(e_edges), e_edges)
#print("odd_degree_vertices_E", odd_degree_vertices_E)
#print("perfect_matching_E", perfect_matching_E)
#print("correct_perfect_matching_E", correct_perfect_matching_E)
#print("new_edges_E",len(new_edges_E), new_edges_E)
#print("totdistE", totdistE)

# TODO ici on applique le problème à l'exemple E
#available_e_edges = len(e_edges) * [True]
#cycleE1, available_e_edges1 = cycle(0,e_edges, new_edges_of_each_vertex_E, available_e_edges)

# print("cycleE1", cycleE1, available_e_edges1)



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

