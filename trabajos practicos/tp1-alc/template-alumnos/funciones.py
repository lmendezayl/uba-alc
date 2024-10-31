import numpy as np
import scipy as sp

# Pendiente: Comentar ambas funciones

def calcularLU(A):

    n,m = A.shape
    if n!=m:
        return "La matríz debe ser cuadrada"
    
    L, U = [],[]
    P = np.eye(n)
    
    L = np.zeros((n,n))
    U = np.copy(A)

    for i in range(0,n):

        # Buscamos el id de la fila con el pivote de mayor módulo
        id_max = np.argmax(np.abs(U[i][i:])) + i

        # Si el pivote de mayor módulo es 0, la matríz era singular
        if (U[id_max,i] == 0):
            print ("La matríz no es inversible")
            return

        # Si es necesario swapear, swapeamos las columnas correspondientes
        if (id_max != i):
            temp = np.copy(P[i,:])
            P[i,:] = P[id_max,:]
            P[id_max,:] = temp
            U = P @ U
            L = P @ L
            
        # Seteo el elemento de la diagonal en 1
        L[i][i] = 1

        for j in range(i+1,n):
            # Defino el multiplicador correspondiente
            multiplicador = U[j][i] / U[i][i]

            # Lo asigno en L
            L[j][i] = multiplicador

            # Voy triangulando u
            U[j][i:] = U[j][i:] - multiplicador * U[i,i:]
    return L, U, P

""" Test:
C = np.asarray([[1,4,7],[2,5,8],[3,6,10]])
L, U, P = calcularLU(C)

"""



def inversaLU(L, U):
    Inv = []

    n=len(U)
    
    U_inv = sp.linalg.solve_triangular(U, np.eye(n))
    L_inv = sp.linalg.solve_triangular(L, np.eye(n), lower = True)

    assert np.allclose(U_inv ,np.linalg.inv(U))
    assert np.allclose(L_inv ,np.linalg.inv(L))
    
    Inv = U_inv @ L_inv
    
    return Inv
