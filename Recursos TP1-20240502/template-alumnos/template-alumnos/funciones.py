import numpy as np
import networkx as nx
import scipy as sp

def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline()) # cantidad de elementos del sistema
    m = int(f.readline()) # cantidad de links del sistema
    W = np.zeros(shape=(n,n)) #
    for _ in range(m):
        line = f.readline()
        i = int(line.split()[0]) - 1
        j = int(line.split()[1]) - 1
        W[j,i] = 1.0
    f.close()

    return W

def dibujarGrafo(W, print_ejes=True):

    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}

    N = W.shape[0]
    G = nx.DiGraph(W.T)

    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])

    nx.draw(G, pos=nx.spring_layout(G), **options)



def calcularGrado(W):
    '''
    Calcula el grado de cada pagina j de una matriz de conectividad W

    Input:
    W: matriz de conectividad entre paginas i y paginas j.

    Output:
    grados: vector n-dimensional cuyos elementos son los grados de las paginas
    '''
    npages = W.shape[0] # cantidad de paginas
    grados = np.zeros(npages) # array n-dim con los grados de cada pagina j

    for j in range(npages):
      for i in range(npages):
        grados[j] += W[i][j] # en el enunciado, grados[j] = c_j

    return grados

    # funcionando

def matrizPuntajes(W):
    '''
    Calcula la matriz de puntajes R = WD donde W es la matriz de conectividad, D es la matriz diagonal con grados 'normalizados'

    W: matriz de conectividad entre paginas i y paginas j.
    '''
    npages = W.shape[0]
    D = np.eye(npages) # genero D
    grados = calcularGrado(W) # genero array de grados

    for j in range(npages):
      if grados[j] == 0: # evitamos la division por 0
        D[j][j] = 0
      else:
        D[j][j] /= grados[j]

    R = W @ D

    return R

    # funcionando

def naveganteAleatorio(W, p):
    '''
    Calcula la probabilidad de salto entre paginas para γ (gamma) != 1

    Input:
    W: matriz de conectividad
    p: probabilidad de seguir un link de pagina

    Output:
    ez_T: matriz cuyas filas son iguales a z^T
    '''
    grados = calcularGrado(W) # defino vector de grados para calcular de z
    npages = W.shape[0]
    e = np.ones((npages,1)) # crea una matriz de unos de dimension (npages x 1) (vector columna)
    z = np.ones((npages,1))

    for j in range(npages): # cómputo de z
        if grados[j] == 0:
          z[j] /= npages
        else:
          z[j] = (1 - p)/npages

    ez_T = e@(z.T)
    return ez_T

def factLU(A):

    #Get the number of rows
    n = A.shape[0]
    m = A.shape[1]

    if m!=n:
        raise ValueError("La matriz operada en la descomposicion LU no es cuadrada.")
    
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    
    #Loop over rows
    for i in range(n):
            
        #Eliminate entries below i with row operations 
        #on U and reverse the row operations to 
        #manipulate L
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
    
    return L, U

def sort_rnk(arr):
  '''
  Ordena el array dado usando selection sort, es decir que recorre el arreglo buscando siempre el minimo y lo pone delante.
  El factor de ordenanza es el segundo elemento de la tupla.
  Complejidad: O(n^2)

  Input:
      arr: El arreglo a ordenar.
      el parametro de entrada es de tipo: [(index,score)]
  Output:
      Un nuevo arreglo ordenado.

  '''

  rnk = []
  while arr:
      max_score = arr[0][1]  # define el puntaje máximo como el puntaje del primer elemento de la lista
      max_index = 0
      for i in range(len(arr)):  # itera sobre los índices del arreglo
          if arr[i][1] > max_score:  # si el score es mayor actualiza index y score maximo
              max_score = arr[i][1]
              max_index = i
      rnk.append(arr[max_index][0])  # agrega el índice del puntaje máximo a la lista de rangos
      del arr[max_index]  # elimina el elemento con el puntaje máximo de la lista
  return rnk





def calcularRanking(M, p): # ingresa la matriz W de conectividad
    npages = M.shape[0]
    rnk = np.arange(0, npages) # ind{k] = i, la pagina k tienen el iesimo orden en la lista. estan ordenadas en base a la valoracion del ranking
    scr = np.zeros(npages) # scr[k] = alpha, la pagina k tiene un score de alpha. es el array R = WD

    R = matrizPuntajes(M)
    print("M: \n", M)

    A = np.eye(npages) - p*R
    print("A: \n", A)

    # solo vale decir que b = e si γ = 1 (γ = z^T * x)
    # faltaria implementarlo con el navegante aleatorio
    b = np.ones((npages, 1)) # vector columna e

    L, U = factLU(A)
    print("LU: \n", L@U == A)
    print("l: \n", L)
    print("u: \n", U)

    x = sp.linalg.solve_triangular(U,b) 
    y = sp.linalg.solve_triangular(L,x, lower=True)

    print("y(son los scores): \n", y)

    indexAndScore = []

    for i in range(npages):
       scr[i] += y[i] / npages
       indexAndScore.append((i, scr[i]))

    rnk = sort_rnk(indexAndScore)

    return rnk, scr

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)

    return output

