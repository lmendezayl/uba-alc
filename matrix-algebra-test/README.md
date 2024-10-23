
# **Instrucciones para la Entrega**

A continuación, se detallan los pasos para acceder al repositorio, trabajar en tu solución y enviarla para su revisión.

## **Paso 1: Hacer Fork del Repositorio**

1. Accede al repositorio central en GitHub: (este repo) https://github.com/matuteiglesias/matrix-algebra-test.git
2. En la esquina superior derecha, haz clic en el botón **Fork** para crear una copia de este repositorio en tu cuenta de GitHub.
3. Una vez que el repositorio esté en tu cuenta, verás que el nombre cambia a **tu_usuario/matrix-algebra-test**.

## **Paso 2: Clonar el Repositorio a Tu Computadora**

1. Abre una terminal en tu computadora y navega a la carpeta donde deseas trabajar.
2. Clona tu copia del repositorio a tu computadora local ejecutando el siguiente comando:
   ```bash
   git clone https://github.com/TU_USUARIO/matrix-algebra-exam.git
   ```
3. Ingresa en la carpeta clonada:
   ```bash
   cd matrix-algebra-exam
   ```

## **Paso 3: Trabajar en los Problemas**

1. En tu repositorio local, encontrarás los archivos con el código base (Test 01 ipynb).

2. Completa las funciones en cada uno de estos archivos de acuerdo con las instrucciones.

## **Paso 4: Crear una Carpeta para tu Entrega**

1. Dentro de la carpeta `entregas/`, crea una nueva carpeta con tu APELLIDO. Por ejemplo:
   ```bash
   mkdir entregas/TU_APELLIDO
   ```

2. Copia los archivos completados en esa carpeta:
   ```bash
   cp Test_01.ipynb entregas/TU_APELLIDO/
   ```

## **Paso 5: Hacer Commit y Push de los Cambios**

1. Añade los archivos de tu entrega a Git con el siguiente comando:
   ```bash
   git add entregas/TU_APELLIDO/*
   ```

2. Realiza un commit con un mensaje significativo:
   ```bash
   git commit -m "Entrego soluciones del examen"
   ```

3. Envía tus cambios a tu repositorio en GitHub:
   ```bash
   git push origin main
   ```

## **Paso 6: Crear un Pull Request (PR)**

1. Ve a tu repositorio en GitHub (tu copia del examen).
2. Verás un botón que dice **Compare & pull request** o **Nuevo pull request**. Haz clic en él.
3. En el título del pull request, pon algo como **"Entrega de TU_USUARIO"**.
4. Envíalo. ¡Tu entrega ha sido completada!

## **Nota Importante**

- No modifiques otros archivos fuera de tu carpeta de entrega.
- Si necesitas hacer cambios adicionales, puedes hacer más commits y actualizar el Pull Request.

