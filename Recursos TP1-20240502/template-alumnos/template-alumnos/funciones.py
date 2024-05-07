import numpy as np
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import stats


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
    A = np.eye(npages) - p*R
    b = np.ones((npages, 1)) # vector columna e
    L, U = factLU(A)
    y = sp.linalg.solve_triangular(L,b, lower=True)
    x = sp.linalg.solve_triangular(U,y)
    indexAndScore = []
    norma = sp.linalg.norm(x,1)
    scr = x/norma

    for i in range(npages):
       indexAndScore.append((i, scr[i]))

    rnk = sort_rnk(indexAndScore)

    print(rnk, scr)
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

def tiempoEjecucion(W, p): 
    """Calcula el tiempo de ejecucion de obtenerMaximoRankingScore en segundos

    Args:
        W: matriz de conectividad
        p: parametro del navegante aleatorio

    Returns:
        tiempo_ejecucion
    """    # calcula el tiempo de ejecucion de obtenerMaximoRankingScore 
    inicio = time.time() # inicia cronometro
    obtenerMaximoRankingScore(W, p)
    fin = time.time() # termina cronometro
    tiempo_ejecucion = fin - inicio 
    
    return tiempo_ejecucion

def tiempoEjecucionSize (dimension,p): 
    """Calcula el tiempo de ejecucion en base a la dimension de W

    Args:
        dimension: dimensión de W
        p: parametro del navegante aleatorio

    Returns:
        lista_tiempo: lista con los tiempos de ejecucion
        lista_size: lista de dimensiones paralelo a los tiempos de ejecucion
    """    
    i=2 # hago que sea minimo 2x2
    lista_tiempo=[]
    lista_size=[]
    while i <= dimension:
        W= np.random.choice([0, 1], size=(i,i)) # creo W con 
        np.fill_diagonal(W, 0)
        tiempo_ejecucion= tiempoEjecucion(W, p)
        lista_tiempo.append(tiempo_ejecucion)
        lista_size.append(i)
        i+=1

    return lista_tiempo, lista_size

def tiempoEjecucionDensidad (densidad_maxima,p,size):
    """Calcula el tiempo de ejecucion en base a la densidad de W (densidad = links/nodos.)

    Args:
        densidad_maxima: densidad máxima de W que se grafica
        p: parametro del navegante aleatorio
        size: dimension de W (matriz cuadrada size x size)

    Returns:
        lista_tiempo: lista con los tiempos de ejecucion
        lista_densidad: lista de densidades paralelo a los tiempos de ejecucion
    """    
    lista_tiempo = []
    lista_densidad = []

    for n in range(50, densidad_maxima * 50+1,50): # se elige 50 porque es un valor suficientemente alto para ver datos pero suficientemente bajo como para no romper la compu. igual cumple por lo explicado en notas. el valor de 50 es indifirente.
        W = np.zeros((size, size), dtype=int)
        np.fill_diagonal(W, 0)
        indices = np.random.choice(size * size, n, replace=False) 
        indices = np.unravel_index(indices, W.shape) # This line converts a flat index or array of flat indices into a tuple of coordinate arrays. Since we're dealing with a 2D array W, we use W.shape to specify its shape. This function essentially converts the flat indices obtained in the previous step into 2D indices that correspond to the positions in the original array W.
        W[indices] = 1
        tiempo_ejecucion= tiempoEjecucion(W, p)
        lista_tiempo.append(tiempo_ejecucion)
        lista_densidad.append(n/50)
    
    return lista_tiempo, lista_densidad
    
# Notas: el tiempo de ejecucion se mantiene constante si el tamaño de la matriz W es fijo. La densidad de W no afecta en absoluto al tiempo de ejecucion pues vale lo mismo peor con un error minimo tanto para densidad igual a 1 como para densidad igual a 100. Veo si con densidad fija pero tamaño variable cambia la cosa (deberia).

def regresionLin(x, y):
    slope, intercept, r, p, std_err = stats.linregress(x, y)   
    return np.multiply(slope, x) + intercept # es necesario pues float * lista se arregla con np.multiply :D

def graficarSize():    
    for i in range(5):
        lista_tiempo, lista_size = tiempoEjecucionSize(100, 0.5)

    sns.set_style("darkgrid")
    plt.scatter(lista_size,lista_tiempo, color='seagreen', label='p=0.5')
    plt.plot(lista_size,lista_tiempo, color='darkgreen', linestyle='-')
    plt.xlabel('Dimensiones de W')
    plt.ylabel('Tiempo de Ejecucion (s)')
    plt.title('Tiempo de ejecucion segun la Dimension de W')
    plt.legend()
    plt.grid(True)
    plt.show()

def graficarDensidadRegresion():
    sns.set_style("darkgrid")
    for i in range(10):
        tiempo, densidad = tiempoEjecucionDensidad (50,0.5,50+i*10)
        plt.scatter(densidad,tiempo,s=8, label=f'{50+i*10}x{50+i*10}')
        regresion_lineal = regresionLin(densidad, tiempo)
        plt.plot(densidad, regresion_lineal)

    plt.xlabel('Densidad de W (cantidad de links/cantidad de nodos)')
    plt.ylabel('Tiempo de Ejecucion (s)')
    plt.title('Tiempo de Ejecucion en funcion a la Densidad de W')
    plt.legend(fontsize='7')
    plt.grid(True)
    plt.show()    

def graficarDensidadPlotError():
    promedios_tiempo = []
    errores_tiempo = []
    densidades = []

    for i in range(10):
        tiempo, densidad = tiempoEjecucionDensidad(50, 0.5, 50+i*10)
        promedios_tiempo.append(np.mean(tiempo))
        errores_tiempo.append(np.std(tiempo))
        densidades.append(50+i*10)

    sns.set_style("darkgrid")
    plt.errorbar(densidades, promedios_tiempo, yerr=errores_tiempo, fmt='o-', capsize=5)
    plt.xlabel('Densidad')
    plt.ylabel('Promedio de Tiempo')
    plt.title('Promedio de Tiempo en Función de la Densidad con Barras de Error')
    plt.grid(True)
    plt.show()

# nota: queda lindo pero cuando se pasa del umbral de 60 dimensiones es como que se va a la mierda, no se porque pero bueno, hecho está

######### ANALISIS CUALITATIVO #########

def calculoProbabilidadesRanking(W,p):
    probabilidades = []
    for i in range(p):
        exit
    ranking, scr =calcularRanking(W,p)
    
    return ranking, probabilidades

def graficarProbabilidadesRanking():
    """
    Crea un gráfico de dispersión con líneas de contorno para visualizar la relación entre
    el valor de p y las probabilidades de las páginas mejor rankeadas.

    Argumentos:
    - p_values: Una secuencia de valores de p.
    - probabilidades: Una secuencia de probabilidades de las páginas mejor rankeadas correspondientes a los valores de p.
    """
    valor_p = np.linspace(0.01, 0.99, 99)
    # Crear el gráfico de dispersión con líneas de contorno
    sns.set_style(style="darkgrid")
    plt.figure(figsize=(8, 6))
    sns.regplot(x=valor_p, y=probabilidades, color='blue', scatter_kws={'s': 50}, line_kws={'color': 'red', 'linewidth': 2})
    plt.xlabel('Valor de p')
    plt.ylabel('Probabilidad de las páginas mejor rankeadas')
    plt.title('Gráfico de dispersión con líneas de tendencia')
    plt.show()


