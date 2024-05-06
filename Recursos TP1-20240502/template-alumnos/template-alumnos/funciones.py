import numpy as np
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import time


def leer_archivo(Args_file_path):

    f = open(Args_file_path, 'r')
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
    '''Calcula el grado de cada pagina j de una matriz de conectividad W

    Args:
        W: matriz de conectividad entre paginas i y paginas j.

    Returns:
        grados: vector n-dimensional cuyos elementos son los grados de las paginas    '''
    npages = W.shape[0] # cantidad de paginas
    grados = np.zeros(npages) # array n-dim con los grados de cada pagina j
    for j in range(npages):
     grados[j] = sum(W[i][j] for i in range(npages)) 
     
    return grados

def matrizPuntajes(W):
    '''Calcula la matriz de puntajes R = WD donde W es la matriz de conectividad, D es la matriz diagonal con grados 'normalizados'

    Args:
        W: matriz de conectividad entre paginas i y paginas j.
        
    Returns: 
        R: matriz de puntajes'''
    npages = W.shape[0]
    D = np.eye(npages) # genero D
    grados = calcularGrado(W) # genero array de grados
    for j in range(npages):
        D[j][j] = 0 if grados[j] == 0 else D[j][j] / grados[j]

    return W@D

def naveganteAleatorio(W, p):
    '''Calcula la probabilidad de salto entre paginas para γ (gamma) != 1

    Args:
        W: matriz de conectividad
        p: probabilidad de seguir un link de pagina

    Returns:
        ez_T: matriz cuyas filas son iguales a z^T    '''
    
    grados = calcularGrado(W) # defino vector de grados para calcular de z
    npages = W.shape[0]
    e, z = np.ones((npages,1)), np.ones((npages,1)) # crea una matriz de unos de dimension (npages x 1) (vector columna)
    for j in range(npages): # cómputo de z
        z[j] = z[j]/npages if grados[j] == 0 else (1-p)/npages
    ez_T = e@(z.T)
    
    return ez_T

def factLU(A):
    """Descompone matriz A en LU

    Args:
        A (_type_): _description_

    Returns:
        _type_: _description_"""
    
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    if m!=n:
        print('Matriz no cuadrada')
        return
    matriz_factores = np.zeros((n,n))  # lista que almacena las matrices factores
    
    for j in range(n): # j = columna
        factores = np.zeros((n,n))  # creo una matriz identidad del tamaño de A
        for i in range(j+1, n): # i = fila
            factores[i, j] = - Ac[i, j] / Ac[j, j]  # calcular el coeficiente factores para cada fila i
            Ac[i, j:] += factores[i, j] * Ac[j, j:]  # aplicar la operación de eliminación gaussiana a la fila i
        matriz_factores += factores  # agregar la matriz factores a la lista.
        
    Ac -= matriz_factores
    L = np.tril(Ac,-1) + np.eye(A.shape[0])
    U = np.triu(Ac)

    return L, U


def sort_rnk(arr):
    '''Ordena el array dado usando selection sort, es decir que recorre el arreglo buscando siempre el minimo y lo pone delante.
    El factor de ordenanza es el segundo elemento de la tupla.
    Complejidad: O(n^2)

    Args:
        arr: El arreglo a ordenar.
        el parametro de entrada es de tipo: [(index,score)]
    Returns:
        Un nuevo arreglo ordenado.'''
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
    A = np.eye(npages) - p*R
    b = np.ones((npages, 1)) # vector columna e
    L, U = factLU(A) 
    x = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L,b, lower=True))
    norma = np.linalg.norm(x,1)
    scr = x/norma
    indexAndScore = []
    for i in range(npages):
       indexAndScore.append((i, scr[i]))
    rnk = sort_rnk(indexAndScore)
    
    return rnk, scr

def obtenerMaximoRankingScore(M, p):
    output = -np.inf
    # calculo el ranking y los scores
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)

    return output


#####################################
#                ANALISIS CUANTITATIVO DE DATOS               #
#####################################

p = 0.95

"""def generar_grafo_aleatorio(num_nodos, num_enlaces):
    G = nx.gnm_random_graph(num_nodos, num_enlaces)
    return nx.adjacency_matrix(G).toarray()

def medir_tiempos_de_ejecucion(tamaños_red, generar_grafo_func, p):
    
    tiempos_ejecucion = []

    for tamaño in tamaños_red:
        W = generar_grafo_func(tamaño, tamaño * 2)  # Por ejemplo, el doble de enlaces que de nodos
        tiempo = calcular_tiempo_ejecucion(W, p)
        tiempos_ejecucion.append(tiempo)

    return tiempos_ejecucion

# Definir rangos de tamaños de grafos y densidades
tamaños_grafo =np.arange(5,10,5)  
densidades = np.arange(0.1,0.5,0.1) 

# Lista para almacenar los tiempos de ejecución
tiempos_ejecucion = []

# Generar grafos aleatorios y medir tiempos de ejecución para cada combinación de tamaño y densidad
for tamaño in tamaños_grafo:
    for densidad in densidades:
        W = generar_grafo_aleatorio(tamaño, int(tamaño * densidad))
        tiempo = calcular_tiempo_ejecucion(W, p)
        tiempos_ejecucion.append(tiempo)

# Representación gráfica de los tiempos de ejecución
plt.figure(figsize=(40, 30))
for i in range(len(tamaños_grafo)):
    plt.plot(densidades, tiempos_ejecucion[i::len(tamaños_grafo)])
    plt.scatter(densidades, tiempos_ejecucion[i::len(tamaños_grafo)], label=f"Tamaño del Grafo: {tamaños_grafo[i]}")
plt.xlabel('Densidad del Grafo', size=15)
plt.ylabel('Tiempo de Ejecución (segundos)', size=15)
plt.title('Tiempo de Ejecución del Algoritmo de PageRank en función de la Densidad del Grafo')
plt.legend()
plt.grid(True, alpha=1)
plt.show()
 """
def tiempoEjecucion(W, p): 
    """Calcula el tiempo de ejecucion de obtenerMaximoRankingScore en segundos

    Args:
        W: matriz de conectividad
        p: parametro del navegante aleatorio

    Returns:
        tiempo_ejecucion
    """    # calcula el tiempo de ejecucion de obtenerMaximoRankingScore 
    inicio = time.time()
    obtenerMaximoRankingScore(W, p)
    fin = time.time()
    tiempo_ejecucion = fin - inicio 
    
    return tiempo_ejecucion

def tiempoEjecucionSize (n,p): 
    """Calcula el tiempo de ejecucion en base a la dimension de W

    Args:
        n: dimensión de W
        p: parametro del navegante aleatorio

    Returns:
        lista_tiempo: lista con los tiempos de ejecucion
        lista_size: lista de dimensiones paralelo a los tiempos de ejecucion
    """    
    i=2
    lista_tiempo=[]
    lista_size=[]
    while i <=n:
        W= np.random.choice([0, 1], size=(i,i))
        np.fill_diagonal(W, 0)
        tiempo_ejecucion= tiempoEjecucion(W, p)
        lista_tiempo.append(tiempo_ejecucion)
        lista_size.append(i)
        i+=1
    return lista_tiempo, lista_size

def tiempoEjecucionDensidad (n,p):
    W=  np.zeros((n, n))
    tiempo= tiempoEjecucion(W, p)
    tiempos= []
    nodos=[]
    conexiones = 0
    tiempos.append(tiempo)
    nodos.append(conexiones)
    for i in range (0,n):
        for j in range (0,n):
            if i!=j:
                W[i][j]=1
                tiempo= tiempoEjecucion(W, p)
                tiempos.append(tiempo)
                conexiones+=1
                nodos.append(conexiones)
    return tiempos, nodos

def graficarSize():
    tamaño1, tiempo1= tiempoEjecucionSize(100, 0.5)
    tamaño2, tiempo2= tiempoEjecucionSize(100, 0.25)

    plt.scatter(tamaño1,tiempo1, color='seagreen', label='p=0.5')
    plt.scatter(tamaño2, tiempo2, color='darkseagreen', label='p=0.25')
    plt.plot(tamaño1,tiempo1, color='darkgreen', linestyle='-')
    plt.plot(tamaño2, tiempo2, color='forestgreen', linestyle='-')
    plt.xlabel('dimensiones del grafo ')
    plt.ylabel('tiempo de ejecucion tardado [s]')
    plt.title('Tiempo de ejecucion del calculo del rankingpage segun el tamaño del grafo')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def graficarDensidad():
    
    tiempo, nodos= tiempoEjecucionDensidad (15,0.5)
    
    # Crear el gráfico de dispersión con múltiples conjuntos de datos
    plt.scatter(nodos,tiempo, color='palevioletred', label='tamaño=15*15,p=0.5')
    plt.xlabel('conexiones dentro del grafo ')
    plt.ylabel('tiempo de ejecucion tardado [s]')
    plt.title('Tiempo de ejecucion del calculo del rankingpage segun las conexiones entre paginas')
    plt.legend()
    plt.grid(True)
    plt.show()    


def graficarDensidad_2():
    
    tiempo4, nodos4= tiempoEjecucionDensidad (50,0.5)
    plt.scatter(nodos4, tiempo4, color='mediumvioletred', label='tamaño=50*50,p=0.5')
    
    # Añadir etiquetas y leyenda
    plt.xlabel('conexiones dentro del grafo ')
    plt.ylabel('tiempo de ejecucion tardado [s]')
    plt.title('Tiempo de ejecucion del calculo del rankingpage segun las conexiones entre paginas')
    plt.legend()
    plt.grid(True)
    plt.show() 
