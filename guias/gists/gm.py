import numpy as np

A = np.array([[1,1,0],[1,0,1], [0,1,1]])
Q, R = np.linalg.qr(A, 'complete')
print(Q, "\n", R)

A = np.array([[2, 3], [2, 1]])
e = np.linalg.eig(A) # e es una lista con dos elementos
print("Autovalores: ", e[0]) # El primer elemento es un array de autovalores
print("Autovectores:\n", e[1]) # El segundo elemento es una matriz con los autovectores como columnas.
