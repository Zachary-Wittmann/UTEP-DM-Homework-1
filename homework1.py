import numpy as np
import csv
from sklearn.neighbors import NearestNeighbors as nn


def myManhattan(v1, v2):
    return sum(abs(v1-v2))


def myEuclid(v1, v2):
    return np.sqrt(sum((v1-v2)**2))


def myJaccard(v1, v2):
    minimums = [min(v1[i], v2[i]) for i in range(0, len(v1))]
    maximums = [max(v1[i], v2[i]) for i in range(0, len(v1))]
    return sum(minimums)/sum(maximums)


def myJaccardDis(v1, v2):
    return 1.0-myJaccard(v1, v2)


def myCosine(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.sqrt(sum(v1**2)) * np.sqrt(sum(v2**2))
    return numerator/denominator


def myTanimoto(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = sum(v1**2 + v2**2) - numerator
    return numerator/denominator


# My first attempt for coding Hamming Distance
def myHamming(v1, v2):
    differences = [i for i in range(len(v2)) if v2[i] != v1[i]]
    return len(differences)


with open('data-HW1.csv', mode='r') as file:
    heading = next(file)
    csvFile = csv.reader(file)
    ogData = [line for line in csvFile]

data = np.asarray(ogData, dtype=float)

refInd1 = 20
refInd2 = 96

vec1 = data[refInd1]
vec2 = data[refInd2]

print("Manhattan Distance: ", myManhattan(vec1, vec2))
print("Euclidean Distance: ", myEuclid(vec1, vec2))
print("Jaccard Coefficient: ", myJaccard(vec1, vec2))
print("Jaccard Dissimilarity: ", myJaccardDis(vec1, vec2))
print("Cosine Similarity: ", myCosine(vec1, vec2))
print("Tanimoto Similarity: ", myTanimoto(vec1, vec2))


refInd = 110

refVec = data[refInd]
k = 5


nbrs = nn(n_neighbors=k, algorithm='brute',
          metric=myJaccardDis)


fitted = nbrs.fit(data)

distances, indices = fitted.kneighbors([refVec])

print(k, " nearest neighbors and distances are:")

for i in range(k):
    print("Row", indices[0][i], "\nWith a distance of:", distances[0][i])
