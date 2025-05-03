# Importamos bibliotecas necesarias:
import numpy as np                      # Biblioteca para cálculos numéricos (como usar raíces cuadradas, potencias, arreglos, etc.)
import matplotlib.pyplot as plt        # Biblioteca para hacer gráficos (dibujar funciones)
import seaborn as sns                  # Biblioteca para mejorar el estilo de los gráficos

# Ajustamos el estilo visual de los gráficos
sns.set(style="whitegrid")             # Ponemos un fondo blanco con líneas suaves de cuadrícula

# Definimos la ecuación diferencial que queremos resolver:
# dy/dx = x^2 / (y + 0.5)
def f(x, y):
    return x**2 / (y + 0.5)

# Calculamos la derivada total con respecto a x, necesaria para el método de Taylor de segundo orden
def df_dx(x, y):
    dy_dx = f(x, y)                    # Llamamos a la función original para obtener dy/dx
    # Aplicamos una fórmula derivada matemáticamente para obtener df/dx
    df = (2 * x * (y + 0.5) - x**2 * dy_dx) / (y + 0.5)**2
    return df

# Esta es la solución exacta de la ecuación, que se obtuvo usando matemáticas analíticas
def y_exact(x):
    return (-1 + np.sqrt((8 * x**3 + 19) / 3)) / 2

# ---------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------

x0 = 1        # Valor inicial de x
y0 = 1        # Valor inicial de y
h = 1         # Tamaño del paso (cuánto avanza x en cada iteración)
N = 5         # Número de pasos que se van a calcular

# Creamos una lista con los valores de x que se van a usar: [1, 2, 3, 4, 5, 6]
x_vals = np.arange(x0, x0 + (N+1)*h, h)

# Calculamos los valores exactos de y para cada x
exact_vals = y_exact(x_vals)

# Creamos arreglos para guardar los resultados de cada método
euler_vals = np.zeros(N+1)
heun_vals = np.zeros(N+1)
rk4_vals = np.zeros(N+1)
taylor_vals = np.zeros(N+1)

# Asignamos la condición inicial y0 al primer valor de cada método
euler_vals[0] = heun_vals[0] = rk4_vals[0] = taylor_vals[0] = y0

# ---------------------------
# CÁLCULOS NUMÉRICOS
# ---------------------------

# Iteramos desde el paso 0 hasta el paso N-1
for n in range(N):
    x_n = x_vals[n]  # Valor actual de x

    # Método de Euler (simple)
    euler_vals[n+1] = euler_vals[n] + h * f(x_n, euler_vals[n])

    # Método de Heun (una mejora del método de Euler)
    pred = heun_vals[n] + h * f(x_n, heun_vals[n])  # Predicción con Euler
    corr = heun_vals[n] + (h/2) * (f(x_n, heun_vals[n]) + f(x_n + h, pred))  # Corrección promediando
    heun_vals[n+1] = corr

    # Método de Runge-Kutta de orden 4 (muy preciso)
    k1 = f(x_n, rk4_vals[n])
    k2 = f(x_n + h/2, rk4_vals[n] + (h/2) * k1)
    k3 = f(x_n + h/2, rk4_vals[n] + (h/2) * k2)
    k4 = f(x_n + h, rk4_vals[n] + h * k3)
    rk4_vals[n+1] = rk4_vals[n] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    # Método de Taylor de segundo orden
    taylor_vals[n+1] = taylor_vals[n] + h * f(x_n, taylor_vals[n]) + (h**2 / 2) * df_dx(x_n, taylor_vals[n])

# ---------------------------
# VISUALIZACIÓN DE RESULTADOS
# ---------------------------

# Creamos una figura con dos gráficas una al lado de la otra
fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1]})
fig.suptitle(r'Comparación de Métodos Numéricos para $\frac{dy}{dx} = \frac{x^2}{y + 0.5}$', fontsize=16, fontweight='bold')

# PRIMER GRÁFICO: Soluciones
axs[0].plot(x_vals, exact_vals, 'k-', linewidth=2.5, label='Exacta')            # Línea negra: solución exacta
axs[0].plot(x_vals, euler_vals, 'o--', color='red', label='Euler')              # Círculos rojos: método de Euler
axs[0].plot(x_vals, heun_vals, 's-.', color='green', label='Heun')              # Cuadrados verdes: Heun
axs[0].plot(x_vals, rk4_vals, '^-', color='blue', label='Runge-Kutta 4')        # Triángulos azules: RK4
axs[0].plot(x_vals, taylor_vals, 'D:', color='purple', label='Taylor orden 2')  # Rombos morados: Taylor

axs[0].set_title('Soluciones Aproximadas', fontsize=14)
axs[0].set_xlabel('x', fontsize=12)
axs[0].set_ylabel('y(x)', fontsize=12)
axs[0].legend()
axs[0].grid(True)

# SEGUNDA GRÁFICA: Tabla de resultados

axs[1].axis('off')  # Quitamos los ejes para mostrar una tabla en su lugar

# Creamos los datos para la tabla: una fila por cada paso
table_data = [
    [f"{x_vals[i]:.0f}", 
     f"{exact_vals[i]:.4f}",
     f"{euler_vals[i+1]:.4f}", 
     f"{heun_vals[i+1]:.4f}", 
     f"{rk4_vals[i+1]:.4f}", 
     f"{taylor_vals[i+1]:.4f}"] for i in range(N)
]

# Etiquetas de las columnas
col_labels = ["x", "Exacta", "Euler", "Heun", "RK4", "Taylor"]

# Insertamos la tabla en la gráfica
table = axs[1].table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')

# Ajustes visuales de la tabla
table.scale(1.2, 1.5)                    # Escalamos tamaño de celdas
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(col_labels))))

# Estilizamos el encabezado
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Fila del encabezado
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')  # Fondo azul oscuro
    else:
        cell.set_facecolor('#f2f2f2')  # Fondo gris claro para las filas

axs[1].set_title("Tabla de Valores", fontweight="bold", fontsize=14)

# Ajustamos el espacio de la figura
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()  # Mostramos todo en pantalla
