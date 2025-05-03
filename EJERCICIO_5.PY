# Importamos las bibliotecas necesarias:
import numpy as np  # Biblioteca para cálculos matemáticos (números, arreglos, etc.)
import matplotlib.pyplot as plt  # Biblioteca para hacer gráficos
import seaborn as sns  # Biblioteca para mejorar el estilo de los gráficos

# Establecemos un estilo visual bonito para los gráficos
sns.set(style="whitegrid")

# Definimos la ecuación diferencial que vamos a resolver: dy/dx = x / y
def f(x, y):
    return x / y  # Esta función representa la derivada de y con respecto a x

# Definimos la derivada parcial de la función anterior, necesaria para el método de Taylor de segundo orden
def df_dx(x, y):
    return (1 / y) - (x**2 / y**3)  # Esta fórmula es matemática, no es necesario memorizarla

# Definimos la solución exacta del problema (conocida previamente)
# y(x) = sqrt(x^2 + 3) es una solución conocida de la ecuación diferencial
def y_exact(x):
    return np.sqrt(x**2 + 3)

# Definimos los valores iniciales:
x0 = 1     # Valor inicial de x
y0 = 2     # Valor inicial de y (cuando x=1, y=2)
h = 0.5    # Tamaño del paso (intervalo entre puntos)
N = 5      # Número de pasos (cuántas veces repetimos los cálculos)

# Creamos arreglos (vectores) para guardar los valores de x y las soluciones con cada método
x_vals = np.arange(x0, x0 + (N+1)*h, h)  # Crea los valores de x desde x0 hasta el último paso
exact_vals = y_exact(x_vals)            # Calcula los valores exactos para cada x
euler_vals = np.zeros(N+1)              # Arreglo para los valores calculados con el método de Euler
heun_vals = np.zeros(N+1)               # Arreglo para los valores con el método de Heun
rk4_vals = np.zeros(N+1)                # Arreglo para el método de Runge-Kutta de orden 4
taylor_vals = np.zeros(N+1)             # Arreglo para el método de Taylor orden 2

# Asignamos la condición inicial a todos los métodos
euler_vals[0] = y0
heun_vals[0] = y0
rk4_vals[0] = y0
taylor_vals[0] = y0

# Empezamos a aplicar los métodos numéricos
for n in range(N):
    x_n = x_vals[n]  # Valor actual de x en el paso n

    # Método de Euler
    # Este método estima el siguiente valor usando una pendiente constante
    euler_vals[n+1] = euler_vals[n] + h * f(x_n, euler_vals[n])

    # Método de Heun (mejora del método de Euler)
    pred = heun_vals[n] + h * f(x_n, heun_vals[n])  # Predicción con Euler
    corr = heun_vals[n] + (h/2) * (f(x_n, heun_vals[n]) + f(x_n + h, pred))  # Promedio de pendientes
    heun_vals[n+1] = corr  # Asignamos el valor corregido

    # Método de Runge-Kutta de orden 4 (muy preciso)
    # Calcula un promedio ponderado de varias pendientes
    k1 = f(x_n, rk4_vals[n])
    k2 = f(x_n + h/2, rk4_vals[n] + (h/2) * k1)
    k3 = f(x_n + h/2, rk4_vals[n] + (h/2) * k2)
    k4 = f(x_n + h, rk4_vals[n] + h * k3)
    rk4_vals[n+1] = rk4_vals[n] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    # Método de Taylor de segundo orden
    # Usa derivadas adicionales para mejorar la aproximación
    taylor_vals[n+1] = taylor_vals[n] + h * f(x_n, taylor_vals[n]) + (h**2 / 2) * df_dx(x_n, taylor_vals[n])

# Creamos un gráfico con dos partes: una curva y una tabla
fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1]})
fig.suptitle(r'$\frac{dy}{dx} = \frac{x}{y}$ con $y(1)=2$', fontsize=16, fontweight='bold')  # Título principal

# Parte 1: Graficar las soluciones
axs[0].plot(x_vals, exact_vals, 'k-', linewidth=2.5, label='Exacta')         # Línea negra: solución exacta
axs[0].plot(x_vals, euler_vals, 'o--', color='red', label='Euler')           # Círculos rojos
axs[0].plot(x_vals, heun_vals, 's-.', color='green', label='Heun')           # Cuadrados verdes
axs[0].plot(x_vals, rk4_vals, '^-', color='blue', label='Runge-Kutta 4')     # Triángulos azules
axs[0].plot(x_vals, taylor_vals, 'D:', color='purple', label='Taylor orden 2')  # Rombos morados

# Añadimos detalles al gráfico
axs[0].set_title('Soluciones Aproximadas', fontsize=14)  # Título del gráfico
axs[0].set_xlabel('x', fontsize=12)                      # Etiqueta eje x
axs[0].set_ylabel('y(x)', fontsize=12)                   # Etiqueta eje y
axs[0].legend()                                          # Muestra la leyenda (nombres de curvas)
axs[0].grid(True)                                        # Muestra una cuadrícula

# Parte 2: Mostrar una tabla comparando los valores
axs[1].axis('off')  # Ocultamos el gráfico, porque solo queremos mostrar la tabla

# Preparamos los datos de la tabla
table_data = [
    [f"{x_vals[i]:.1f}", 
     f"{exact_vals[i]:.4f}",
     f"{euler_vals[i+1]:.4f}", 
     f"{heun_vals[i+1]:.4f}", 
     f"{rk4_vals[i+1]:.4f}", 
     f"{taylor_vals[i+1]:.4f}"] for i in range(N)
]
col_labels = ["x", "Exacta", "Euler", "Heun", "RK4", "Taylor"]  # Encabezados

# Creamos la tabla dentro del gráfico
table = axs[1].table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.scale(1.2, 1.5)  # Ajustamos el tamaño
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(col_labels))))

# Damos estilo al encabezado de la tabla
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Primera fila (encabezado)
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')  # Color oscuro
    else:
        cell.set_facecolor('#f2f2f2')  # Color claro para el resto

axs[1].set_title("Tabla de Valores", fontweight="bold", fontsize=14)

# Ajustamos el diseño para que todo se vea bien
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Mostramos todo en pantalla
plt.show()
