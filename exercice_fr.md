# Exercice
On souhaite déterminer le trajet de 10 véhicules afin de parcourir la ville de Paris le plus vite possible.
On part du principe que les voitures peuvent être placées n'importe où dans la ville.

# Consignes
- Délai de réponse de l'exercice: 1 semaine maximum.
- Les données d'entrée sont disponibles sur le fichier "paris_map.txt". 
Petit détail sur le format: il est composé de lignes à 2 éléments (les sommets) et d'autres à 3 éléments (les arêtes). 
Les sommets contiennent latitude et longitude du point, les arêtes contiennent le numéro du sommet de départ, d'arrivée et la distance entre les deux points.
Un modèle est fourni dans le code "main.py", dans la fonction de parsing.
- La fonction "main.py" contient également un graphique permettant de visualiser un echantillon des sommets et arêtes définies. 
Pour l'executer faire "python main.py paris_map.txt nb" avec nb le nombre de sommets et d'arêtes à visualiser.
La note "#E=17958, #V=11348, V index from 0 to 11347" donne le nombre de sommets et d'arêtes définies dans l'exemple.
- Ce problème se rapporte à celui du postier chinois. Chaque noeud représente un carrefour et chaque rue est un arc reliant des deux carrefours. 
L’objectif est de parcourir tous les arcs du graphe avec 10 voitures.
- N'hésitez pas à simplifier le problème autant que possible et fournir résultat qui fonctionne, mais sans avoir à trouver l'optimal (car assez long à obtenir).
Dans tous les cas, n'hésitez pas à me contacter si besoin.
- Ne pas hésiter à nous poser des questions si besoin (samir@padam.io).

Bon courage !
