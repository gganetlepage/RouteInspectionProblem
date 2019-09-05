Route Inspection Problem
========================

Problem
-------
Route inspection problem:
The delivery person has to travel across all Paris streets.
Aim is to minimise total distance with an associated delivery person path

Environment
-----------
paris_map contains 2 datatypes.
* Vertex: introduced with its associated cartesian coordinates (x and y position)
* edge is introduced with its 2 incident vertices, plus distance

Methodology
-----------
First, graph modification:

* If the graph is eulerian, then the delivery person could pass by every edges and only once. Which is a optimal case.

* Graph surely isn't eulerian, so it needs to be changed to become one

* A graph is eulerian if and only if all vertices have a even degree. Identification of odd degree vertice, objective is that new edges are added to the global graph so all vertices become of even degree

* Construction of a new complet graph made with all odd degree vertices. Distance of each edge is calculated using Dijkstra algorithm.

* Selection of a good (total distance minimized) perfect matching using a greedy algorithm. The perfect matching allows edge vertex to increase its degree by 1, making them become all even degree vertices on the global graph.

* Addition of the perfect matching to the global graph. Graph becomes eulerian

Second, path determination:

* determination of cycles: As the delivery man goes from one vertex to another, he will probably get stuck at vertex with no unused edge left locally, but globally sections are not treated. So iterations are needed, several cycles are created

* cycles assembly into an unique one

Progress
--------

Currently, this project is at "determination of cycles" phase in the path determination section

Noticable point
---------------

This project is not done using OOP, I didn't know how to do it on Python at the time



