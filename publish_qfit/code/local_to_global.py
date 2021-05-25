import numpy as np

path_lime = "../../comparison_methods/LIME/ranks/orange_skin_local_ranks.npy"
ran = np.load(path_lime)
ran = np.array(ran)
print(ran)
print(ran[0])

ll = np.argsort(ran)
print(ll)
print("num points:", len(ll))

#sum argsort and those will be the points

points = np.sum(ll, axis=0)

allpoints_argsort = np.argsort(points)
str_allpoints  = ",".join([str(a) for a in allpoints_argsort])
print(str_allpoints)

