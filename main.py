import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


def plot_sample(vertices, edges):
    fig, ax = plt.subplots(figsize=(40, 40))
    ax.plot([vertex[0] for vertex in vertices], [vertex[1] for vertex in vertices], "k.")
    # ax.plot(vertices[0][0], vertices[0][1], 'ro')
    for edge in edges:
        ax.plot([_[0] for _ in edge[-2:]], [_[1] for _ in edge[-2:]], linestyle='-', color='grey')
    plt.savefig("edgesPathGraph.png")
    plt.show()


def plot_animation(vertices, edges):
    fig, ax = plt.subplots(figsize=(40, 20))

    def init_func():
        for edge in edges:
            ax.plot([_[0] for _ in edge[-2:]], [_[1] for _ in edge[-2:]], linestyle='-', color='grey')
        # ax.plot([vertex[0] for vertex in vertices], [vertex[1] for vertex in vertices], "k.")
        ax.plot(vertices[0][0], vertices[0][1], 'ro')

    def update(frame):
        edge = edges[frame]
        line, = ax.plot([_[0] for _ in edge[-2:]], [_[1] for _ in edge[-2:]], linestyle='-', color='blue')
        return line,

    """class matplotlib.animation.FuncAnimation(fig, func,
     frames=None, init_func=None, fargs=None, save_count=None, *, cache_frame_data=True, **kwargs)"""
    animation = FuncAnimation(fig, update, frames=np.arange(0, len(edges), 1),
                              init_func=init_func(),
                              repeat=False)
    # TODO save not working. reported bug. Should be fixed with matplotlib 3.2.2 , currently 3.2.1
    # animation.save("animation.mp4", fps=30, writer='ffmpeg')  # dpi = 150
    # animation.save('animation.mp4')
    # plt.show()


def vertices_degree(vertices, edges):
    """Enumerate each vertex degree. Complexity: 0(edges)"""
    vertices_degree = len(vertices) * [0]
    for i in range(len(edges)):
        # 1 edge increases degree of the 2 incident vertices
        vertices_degree[edges[i][0]] += 1
        vertices_degree[edges[i][1]] += 1
    return vertices_degree


def odd_degree_vertices(vertices_degree):
    """
    :param vertices_degree: table, for each vertex, its number of incident edges
    :return: table, for each odd degree vertex, its index
    Enumerate odd degree vertex index. Complexity: 0(vertices)
    """
    odd_degree_vertices = []
    for i in range(len(vertices_degree)):
        if vertices_degree[i] % 2 == 1:
            odd_degree_vertices.append(i)
    return odd_degree_vertices


def edges_of_each_vertex(vertices, edges, vertices_degree):
    """
    :param vertices: table of vertices
    :param edges:  table of edges
    :param vertices_degree: table of vertices degree
    :return: table, incident edges number for each vertex
    Sort the incident edges of each vertex Complexity: O(edges)
    """
    vertices_size = len(vertices)
    edges_size = len(edges)
    edges_of_each_vertex = vertices_size * [0]
    count_occurrence = vertices_size * [0]
    for i in range(vertices_size):
        edges_of_each_vertex[i] = vertices_degree[i] * [0]
    for j in range(edges_size):
        first_vertex = edges[j][0]
        second_vertex = edges[j][1]
        edges_of_each_vertex[first_vertex][
            count_occurrence[first_vertex]] = j  # association of incident edges number to the vertex
        edges_of_each_vertex[second_vertex][count_occurrence[second_vertex]] = j
        count_occurrence[first_vertex] += 1
        count_occurrence[second_vertex] += 1

    return edges_of_each_vertex


def dijkstra_vertex(vertex, edges_of_each_vertex, odd_degree_vertices, edges):
    """
    :param vertex: starting point for dijkstra's algorithm
    :param edges_of_each_vertex:
    :param odd_degree_vertices:
    :param edges:
    :return:
    Apply dijkstra's algorithm starting from parameter vertex.
    Compute distance and predecessor edge between parameter vertex and all other odd degree vertices
    Complexity:  # TODO
    """
    vertices_size = len(edges_of_each_vertex)
    # initialization
    dijkstra_distance = vertices_size * [sys.maxsize]
    dijkstra_distance[vertex] = 0
    dijkstra_predecessor = vertices_size * [0]  # TODO check no edge effect
    dijkstra_predecessor[vertex] = -1

    dijkstra_previous_edge = vertices_size * [0]
    dijkstra_previous_edge[vertex] = -1

    heap = []
    heapq.heappush(heap, [0, vertex])
    while len(heap) != 0:
        weight, v = heap[0]
        heapq.heappop(heap)
        for i in range(len(edges_of_each_vertex[v])):
            if edges[edges_of_each_vertex[v][i]][0] == v:
                adjacent_v = edges[edges_of_each_vertex[v][i]][1]
            else:
                adjacent_v = edges[edges_of_each_vertex[v][i]][0]
            weight_successor = weight + edges[edges_of_each_vertex[v][i]][2]
            if weight_successor < dijkstra_distance[adjacent_v]:
                dijkstra_distance[adjacent_v] = weight_successor
                dijkstra_predecessor[adjacent_v] = v  # = v or edges_of_each_vertex[v][i] .why did I wrote that?
                dijkstra_previous_edge[adjacent_v] = edges_of_each_vertex[v][i]
                heapq.heappush(heap, [weight_successor, adjacent_v])
    return dijkstra_distance, dijkstra_predecessor, dijkstra_previous_edge


def dijkstra_vertices(edges_of_each_vertex, odd_degree_vertices, edges):
    """
    :param edges_of_each_vertex:
    :param odd_degree_vertices:
    :param edges:
    :return: matrix of distance and predecessor, for all odd degree vertices
    Apply dijkstra's algorithm to all odd degree vertices. Complexity: #TODO
    """

    vertices_number = len(edges_of_each_vertex)
    odd_degree_vertices_number = len(odd_degree_vertices)
    distance_matrix = vertices_number * [sys.maxsize]
    predecessor_matrix = vertices_number * [0]
    previous_edge_matrix = vertices_number * [0]
    # only odd degree vertices are interesting, no need to do dijkstra for all vertices
    for i in range(odd_degree_vertices_number):
        odd_degre_vertex = odd_degree_vertices[i]
        dijkstra_distance, dijkstra_predecessor, dijkstra_previous_edge = dijkstra_vertex(odd_degre_vertex,
                                                                                          edges_of_each_vertex,
                                                                                          odd_degree_vertices, edges)
        distance_matrix[odd_degre_vertex] = dijkstra_distance
        # problem : matrix should be mirror ?
        predecessor_matrix[odd_degre_vertex] = dijkstra_predecessor
        previous_edge_matrix[odd_degre_vertex] = dijkstra_previous_edge
    return distance_matrix, predecessor_matrix, previous_edge_matrix


def complete_graph(odd_degree_vertices, distance_matrix):
    size = len(odd_degree_vertices)
    new_dist_matrix = size * [0]
    for i in range(size):
        line = size * [0]
        for j in range(size):
            line[j] = distance_matrix[odd_degree_vertices[i]][odd_degree_vertices[j]]
        new_dist_matrix[i] = line
    return new_dist_matrix


def minlist(list, treated_vertices, x):
    # TODO improve algorithm, currently polynomial
    size = len(list)
    min = sys.maxsize
    y = -1
    for i in range(size):  # i!= index because the edge of distance=0 must not be selected
        if i != x and not treated_vertices[i] and list[i] < min:
            min = list[i]
            y = i
    if y == -1:
        print("ERROR, no edge selected")
    else:
        return y


"""
def greedy_algorithm(odd_degree_vertices_dist_matrix):  # odd_degree_vertices_dist_matrix = new_dist_matrix
    size = len(odd_degree_vertices_dist_matrix)  # size is an even number
    # for i in range(size):
    #     print(odd_degree_vertices_dist_matrix[i])
    treated_vertices = size * [False]
    edges_size = size / 2
    edges_selected = []
    x = 0
    while len(edges_selected) != edges_size and x < size:  # and x < size
        if not treated_vertices[x]:
            y = minlist(odd_degree_vertices_dist_matrix[x], treated_vertices, x)
            edges_selected.append([x, y])
            treated_vertices[x] = True
            treated_vertices[y] = True
            # print(x, y, edges_selected)
        x += 1
    return edges_selected  # = perfect matching #Attention les indices de sommets sélectionnés sont ceux du graphe de sommets de degré impair uniquement
"""


# il faut donc faire attention à ensuite convertir en ceux du graphe total original


def greedy_algorithm2(odd_degree_vertices_dist_matrix,
                      odd_degree_vertices):  # odd_degree_vertices_dist_matrix = new_dist_matrix
    size = len(odd_degree_vertices_dist_matrix)  # size is an even number
    treated_vertices = size * [False]
    edges_size = size / 2
    edges_selected = []
    x = 0
    while len(edges_selected) != edges_size and x < size:  # and x < size
        if not treated_vertices[x]:
            y = minlist(odd_degree_vertices_dist_matrix[x], treated_vertices, x)
            edges_selected.append([odd_degree_vertices[x], odd_degree_vertices[y]])
            treated_vertices[x] = True
            treated_vertices[y] = True
        x += 1
    return edges_selected  # = correct perfect matching, with index matching those in the general graph


def path(predecessor_matrix, previous_edge_matrix, edges, final_vertex,
         starting_vertex):  # path() can't be applied with a even degree vertex
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
        if predecessor_matrix[starting_vertex][cursor] != -1:
            edges_path.append(edges[previous_edge_matrix[starting_vertex][cursor]])
    return vertices_path, edges_path


def graph_with_added_edges(perfect_matching, predecessor_matrix, previous_edge_matrix, edges):
    size = len(perfect_matching)
    new_edges = copy.copy(edges)
    for i in range(size):
        vertices_path, edges_path = path(predecessor_matrix, previous_edge_matrix, edges, perfect_matching[i][0],
                                         perfect_matching[i][1])
        for j in range(len(edges_path)):
            new_edges.append(edges_path[j])
    return new_edges


def available_edge_index(available_edges):
    """:param available_edges: is_an_edge_available table
        :return: if a edge is available and index of the first one  """
    i = 0
    size = len(available_edges)
    is_an_edge_available = True
    while i < size and not available_edges[i]:
        i += 1
    if i == size:
        is_an_edge_available = False  # "All edges have been taken"
    return is_an_edge_available, i  # i = first available edge index


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


def cycle(starting_vertex, edges, edges_of_each_vertex, available_edges):
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
    compteur = 0  # TODO
    while bool and compteur < 5:
        cycle2, available_edges = cycle(starting_vertex, edges, edges_of_each_vertex,
                                        available_edges)  # effet de bord sur available_edges
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
    for i in range(size):  # create a list indicating the vertices and the edge concerned
        edge_index = cycle[i]
        edge = new_edges[edge_index]
        vertices_list.append([edge[0], edge_index, i])  # i is the edge index in the cycle
        vertices_list.append([edge[1], edge_index, i])
    return vertices_list


""""
Next step: Create the vertices list associated to a cycle list
"""


def partition(vertices_list, low, high):
    pivot = vertices_list[high][0]  # element used for comparaison
    i = low - 1  # index used to swap
    for j in range(low, high):
        if vertices_list[j][0] < pivot:
            i += 1
            vertices_list[i], vertices_list[j] = vertices_list[j], vertices_list[i]
    vertices_list[i + 1], vertices_list[high] = vertices_list[high], vertices_list[i + 1]
    return i + 1  # index of pi


def quicksort(vertices_list, low, high):
    if high <= low:
        return "Error in indexes"
    else:

        pi = partition(vertices_list, low,
                       high)  # pi is partitioning index, verticesl_list[pi] is now at the right place
        quicksort(vertices_list, low, pi - 1)  # before pi
        quicksort(vertices_list, pi + 1, high)  # after pi


def reducing_list(
        vertices_list):  # each vertex is occuring twice in the vertices_list, we only it once it fasten the method
    reduced_list = []
    size = len(vertices_list)
    steps = int(size / 2)
    for i in range(steps):
        reduced_list.append(vertices_list[2 * i])
    return reduced_list


def vertices_cycle(cycle_list, new_edges):
    vertices_list = []
    for x in cycle_list:
        y = vertices_of_a_edge_list(x, new_edges)
        quicksort(y, 0, len(y) - 1)
        reduced_list = reducing_list(y)
        vertices_list.append(reduced_list)
    return vertices_list


"""
Next step: with 2 cycles, being able to determine where to insert the first one into the second one
"""


def vertex_in_list(vertex, vertices_list):
    """is the vertex present in the list? return a bool"""
    present = False
    i = 0
    size = len(vertices_list)
    while not present and i < size:
        if vertices_list[i][0] == vertex:  # [0] due to working with list of 2 elements
            present = True
        else:
            i += 1
    return present, i  # bool and vertex index


def comparison_vertices_list(vertices_list1, vertices_list2):
    present = False
    i_list2 = 0  # index in list2 of a matching vertex with list1
    i = 0
    size = len(vertices_list1)
    while not present and i < size:
        present, i_list2 = vertex_in_list(vertices_list1[i][0],
                                          vertices_list2)  # [0] due to working with list of 2 elements
        if not present:
            i += 1
    return present, i, i_list2  # bool and possible vertices indices


def entering_or_outgoing(i, reduced_list, cycle_list, new_edges):  # True = entering
    vertex = reduced_list[i][0]
    #    edge = reduced_list[i][1]
    index_in_cycle_list = reduced_list[i][2]
    #    edge_index = cycle_list[index_in_cycle_list]
    precedent_edge_index = cycle_list[index_in_cycle_list - 1]
    if new_edges[precedent_edge_index][0] == vertex or new_edges[precedent_edge_index][1] == vertex:
        return False  # outgoing
    else:
        return True  # entering


def real_order_of_cycle(precedent, cycle_list, edge_index_in_cycle):  # edge_index_in_cycle = reduced_list[index][2]
    correct_order_cycle = []
    size = len(cycle_list)
    if precedent:
        # then the edge must be put at the ultimate end, the cycle first leaves the concertex vertex then go back into it
        for i in range(edge_index_in_cycle + 1, size):
            correct_order_cycle.append(cycle_list[i])
        for i in range(edge_index_in_cycle + 1):
            correct_order_cycle.append(cycle_list[i])
    else:
        for i in range(edge_index_in_cycle - size, edge_index_in_cycle):
            correct_order_cycle.append(cycle_list[i])
    return correct_order_cycle


def list2_into_list1(cycle_list1, real_order_cycle_list2, precedent_list1, edge_index_in_cycle1):
    if precedent_list1:
        # then list2 is added after the edge
        # It is not possible for the edge_index_in_cycle1 to be the last element of the cycle_list1, that would mean the list2 can also be added after the first element
        for x in real_order_cycle_list2:
            # list2 is added in reverses
            cycle_list1.insert(edge_index_in_cycle1 + 1, x)
    else:
        for x in real_order_cycle_list2:
            cycle_list1.insert(edge_index_in_cycle1, x)
    return cycle_list1


# delete list1? après la mathode list2_into_list1


# this method is not finished 
def add_list_into_list(cycle_list1, cycle_list2, vertices_list1, vertices_list2, new_edges, i_list1, i_list2):
    #    vertices_list1 = vertices_cycle(cycle_list1, new_edges) vertices_list1 is already reduced_list1
    #    vertices_list2 = vertices_cycle(cycle_list2, new_edges) vertices_list2 is already reduced_list2
    edge_index_in_cycle1 = vertices_list1[i_list1][2]  # TODO why not use directly i_list or i_list2 ?
    edge_index_in_cycle2 = vertices_list2[i_list2][2]
    precedent_list1 = entering_or_outgoing(i_list1, vertices_list1, cycle_list1, new_edges)
    precedent_list2 = entering_or_outgoing(i_list2, vertices_list2, cycle_list2, new_edges)
    # arbitrary choice to insert list2 into list1, as list1 is generally longer on tested cases
    correct_order_cycle_list2 = real_order_of_cycle(precedent_list2, cycle_list2, edge_index_in_cycle2)
    concatenated_list = list2_into_list1(cycle_list1, correct_order_cycle_list2, precedent_list1, edge_index_in_cycle1)
    return concatenated_list


# try to treat every case possible

def put_all_cycles_into_one3(cycle_list, vertices_list, new_edges):
    # vertices_list is reduced_list

    # loop WHILE here to iterate on every additional cycle
    #    if len(vertices_list) >= 2:
    if len(cycle_list) >= 2:
        vertices_list1 = vertices_list[0]
        vertices_list2 = vertices_list[1]
        present, i_list1, i_list2 = comparison_vertices_list(vertices_list1, vertices_list2)
        if present:
            concatenated_list = add_list_into_list(cycle_list[0], cycle_list[1], vertices_list1, vertices_list2,
                                                   new_edges, i_list1, i_list2)
            # cycle _list is updated by add_list_into_list()
            # so the second element of cycle_list needs to be deleted after that
            # but nothing has been done concerning vertices_list, which needs to be calculated again ?
            return concatenated_list
        else:
            return "there isn't a match there for the moment"  # modification à faire: créer un compteur, itérer tant que l'on n'a pas present = true
    else:
        return cycle_list[0]  # there is only one cycle or none


def put_all_cycles_into_one2(cycle_list, vertices_list, new_edges):
    # tentative récursive ?
    if len(cycle_list) <= 1:
        if len(cycle_list) == 1:
            return cycle_list[0]
        else:
            return cycle_list
    else:
        vertices_list1 = vertices_list[0]  # NNONN, vertices_list[0] va être mise à jour
        count = 1
        count_error = 0  # TODO in case the function is not working properly
        present, i_list1, i_list2 = comparison_vertices_list(vertices_list1, vertices_list[count])
        size = len(vertices_list)
        while not present and count < size and count_error < 10:
            count += 1
            present, i_list1, i_list2 = comparison_vertices_list(vertices_list1, vertices_list[count])
            count_error += 1
        if present:
            concatenated_list = add_list_into_list(cycle_list[0], cycle_list[count], vertices_list1,
                                                   vertices_list[count], new_edges, i_list1, i_list2)
            deleted_cycle = cycle_list[count]
            # vertices_list must be updated now ! does ne_edges needs to be too ?
            vertices_list = vertices_cycle(cycle_list, new_edges)
            return True
    return True


def put_all_cycles_into_one(cycle_list, vertices_list, new_edges):
    count_error = 0  # TODO
    while len(cycle_list) > 1 and count_error < 10:
        count = 1
        count_error += 1
        present, i_list1, i_list2 = comparison_vertices_list(vertices_list[0], vertices_list[count])
        count_error2 = 0  # TODO
        while not present:
            count += 1
            count_error2 += 1
            present, i_list1, i_list2 = comparison_vertices_list(vertices_list[0], vertices_list[count])
        # a vertex is present on both cycles (fro cycle 0 and cycle number "count")

        #        concatenated_list = add_list_into_list(cycle_list[0], cycle_list[count], vertices_list[0], vertices_list[count], new_edges, i_list1, i_list2)
        add_list_into_list(cycle_list[0], cycle_list[count], vertices_list[0], vertices_list[count], new_edges, i_list1,
                           i_list2)
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


def write_order(edgesOrder):
    with open('edgesOrder.txt', "w") as f:
        for i in range(len(edgesOrder)):
            f.write(str(edgesOrder[i]) + '\n')


def read_order(real_run=False):
    if real_run:
        with open('edgesOrder.txt', "r") as f:
            content = f.readlines()
            edgesOrder = []
            for i in range(len(content)):
                edgesOrder.append(int(content[i]))
            return edgesOrder
    else:
        with open('edgesOrderSolution.txt', "r") as f:
            content = f.readlines()
            edgesOrder = []
            for i in range(len(content)):
                edgesOrder.append(int(content[i]))
            return edgesOrder


def is_vertex_in_edge(vertex, edge):
    return vertex == edge[0] or vertex == edge[1]


def association_new_and_old_edges(old_edges, new_edges, edges_order):
    old_size = len(old_edges)
    new_size = len(new_edges)
    if new_size == old_size:
        return edges_order
    else:
        edges_association = []
        # initialization initial edges, no change
        for i in range(len(old_edges)):
            edges_association.append(i)
        # association of the old edges
        for i in range(old_size, new_size):
            for j in range(old_size):
                if new_edges[i] == old_edges[j]:
                    edges_association.append(j)
                    break
        # now association list is correct
        correct_edges_order = []
        for i in range(len(edges_order)):
            announced_index = edges_order[i]
            if announced_index >= old_size:
                # this is an artificially created edge
                correct_index = edges_association[announced_index]
                correct_edges_order.append(correct_index)
            else:
                # this is an initial edge
                correct_edges_order.append(announced_index)
        return correct_edges_order


# no need to extend vertices

def build_order(edges_order, vertices, edges):
    """
    :param edges_order:
    :param vertices:
    :param edges: full list of edges
    :return:
    """
    vertices_path = []
    edges_path = []
    # initialization:
    print(edges_order[0])
    current_edge = edges[edges_order[0]]
    current_vertex_index = current_edge[0]
    if is_vertex_in_edge(current_edge, edges[edges_order[1]]):
        real_first_vertex_index = current_edge[1]
        vertices_path.append(vertices[real_first_vertex_index])
        vertices_path.append(vertices[current_vertex_index])
    else:
        real_first_vertex_index = current_vertex_index
        current_vertex_index = current_edge[1]
        vertices_path.append(vertices[real_first_vertex_index])
        vertices_path.append(vertices[current_vertex_index])
    edges_path.append(current_edge)
    for i in range(1, len(edges_order)):
        current_edge = edges[edges_order[i]]
        if current_vertex_index == current_edge[0]:
            # then the new vertexIndex will be current_edge[1]
            current_vertex_index = current_edge[1]
        else:
            #  then the new vertexIndex will be current_edge[0]
            current_vertex_index = current_edge[0]
        vertices_path.append(vertices[current_vertex_index])
        edges_path.append(current_edge)
    return vertices_path, edges_path


def total_distance(edges):
    distance = 0
    for edge in edges:
        distance += edge[2]
    return distance


# TODO reformulate if elif else by a switch using dictionary
def solving(vertices, edges, read_file="solution", write_file=False, debug=False):
    """
    :param vertices:
    :param edges:
    :param debug: print each step
    :param read_file: "solution" = run edgesOrderSolution.txt, "no" compute entirely the problem, "current" = run current one, which is edgesOrder.txt
    :param write_file: only possible if read_file = "no", write edgesOrder.txt to save computing time for future executions
    :return:

    """
    t0 = time.time()
    if read_file != "no" and write_file:
        print('can not write if not doing a complete run')
        return 0, 0, 0
    if read_file == "no":
        print('doing a complete run')
        # get each vertex degree
        vertices_deg = vertices_degree(vertices, edges)
        # get odd degree vertices
        odd_deg_vertices = odd_degree_vertices(vertices_deg)
        # get edges of those odd degree vertices
        edges_per_vertex = edges_of_each_vertex(vertices_deg, edges, vertices_deg)
        # use dijkstra algorithm to get distance between odd degree vertices
        distance_matrix, predecessor_matrix, previous_edge_matrix = dijkstra_vertices(edges_per_vertex,
                                                                                      odd_deg_vertices,
                                                                                      edges)
        t1 = time.time()
        print('time to process up to dijkstra algorithm:', t1 - t0)  # 220sec
        new_dist_matrix = complete_graph(odd_deg_vertices, distance_matrix)
        correct_perfect_matching = greedy_algorithm2(new_dist_matrix, odd_deg_vertices)
        new_edges = graph_with_added_edges(correct_perfect_matching, predecessor_matrix, previous_edge_matrix,
                                           edges)
        new_vertices_deg = vertices_degree(vertices, new_edges)
        new_edges_of_each_vertex = edges_of_each_vertex(vertices, new_edges, new_vertices_deg)
        # working on list of edges:
        cycle_list1 = cycle_list(new_edges, new_edges_of_each_vertex)
        vertices_list = vertices_cycle(cycle_list1, new_edges)

        # identifying matching vertices, DANGER how to behave if only one element in the list ?

        edges_order = put_all_cycles_into_one(cycle_list1, vertices_list, new_edges)
        t2 = time.time()
        print(t2 - t1)
        real_edges_order = association_new_and_old_edges(edges, new_edges, edges_order)
        if write_file:
            write_order(real_edges_order)
            print('file wrote')

    elif read_file == 'current':
        print('reading current file')
        real_edges_order = read_order(True)
    else:
        # read 'solution' file
        print('reading solution file')
        real_edges_order = read_order()

    # edges_path, vertices_path = build_order(real_edges_order, vertices, new_edges)
    vertices_path, edges_path = build_order(real_edges_order, vertices, edges)
    dist = total_distance(edges_path)
    return edges_path, vertices_path, dist



"""EXEMPLE 2"""
c_vertices = [0, 1, 2, 3, 4]
c_edges = [[0, 1, 3], [0, 2, 10], [1, 2, 2], [3, 2, 2], [1, 3, 4], [1, 4, 1], [4, 3, 7]]

"""EXEMPLE 3"""
d_vertices = [0, 1, 2, 3, 4, 5, 6]
d_edges = [[0, 2, 3], [1, 2, 7], [1, 5, 1], [1, 4, 2], [4, 5, 4], [2, 4, 8], [2, 6, 6], [2, 3, 4], [3, 6, 1], [6, 4, 5],
           [4, 3, 3]]

"""Exemple 4"""
e_vertices = [0, 1, 2, 3, 4, 5, 6, 7]
e_edges = [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [2, 5, 1], [4, 6, 1], [6, 7, 1]]

""" Exemple Principal"""


def main():
    start = time.time()
    vertices, edges = parse_file("paris_map.txt")  # sys.argv[1]

    sol_edges_path, sol_vertices_path, dist = solving(vertices, edges, read_file="no", write_file=False)
    if sol_edges_path == 0 and sol_vertices_path == 0 and dist == 0:
        print('incorrect parameters given for solving method, won\'t plot ')
    else:
        print('total distance:',dist)
        plot_sample(sol_vertices_path, sol_edges_path)
        #TODO once matplotlib updates to 3.2.2, add plot_animation
        # https://github.com/matplotlib/matplotlib/issues/16965
        # plot_animation(sol_vertices_path, sol_edges_path)
        print('total processing time:', time.time()-start)


if __name__ == '__main__':
    main()
