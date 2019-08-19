import matplotlib.pyplot as pyplt
from random import randint
import sys

def parse_file(file_name):

    with open(file_name, "r") as f : lines = f.readlines()

    vertices, edges = [], []
    for i,line in enumerate(lines) :
        splitted_line = line.strip("\n\r").split(" ")
        if len(splitted_line) == 2 :
            vertices.append((float(splitted_line[0]), float(splitted_line[1])))
        elif len(splitted_line) == 3 and i > 0:
            vertice_1,vertice_2 = int(splitted_line[0]),int(splitted_line[1])
            distance = int(splitted_line[2]) # dans les deux sens ou pas
            p1 = vertices[vertice_1]
            p2 = vertices[vertice_2]
            edges.append((vertice_1,vertice_2,distance,p1,p2))
        elif i > 0 :
            raise Exception("unable to interpret line {}: ".format(i) + line)
    print("#E={}, #V={}, V index from {} to {}".format(len(edges), len(vertices), min(min( _[0] for _ in edges), min( _[1] for _ in edges)), 
          max(max( _[0] for _ in edges), max( _[1] for _ in edges))))

    return vertices, edges

def plot_sample(vertices, edges, number):

    # plot vertices
    sample_1 = [vertices[randint(0,len(vertices)-1)] for _ in range(0, int(number))]
    pyplt.plot([_[0] for _ in sample_1], [_[1] for _ in sample_1], ".")
    pyplt.show()

    # plot edges
    sample_2 = [edges[randint(0,len(edges)-1)] for _ in range(0, int(number))]
    for edge in sample_2:
        pyplt.plot( [_[0] for _ in edge[-2:]], [_[1] for _ in edge[-2:]], "b-")
    pyplt.show()

vertices, edges = parse_file(sys.argv[1])
plot_sample(vertices, edges, sys.argv[2])
