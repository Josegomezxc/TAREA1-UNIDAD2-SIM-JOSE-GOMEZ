# Importamos librerías necesarias:
import numpy as np                   # Sirve para hacer cálculos numéricos, como trabajar con arrays (listas de números).
import matplotlib.pyplot as plt      # Sirve para crear gráficos.
import seaborn as sns                # Sirve para mejorar el estilo visual de los gráficos.

# Establecemos un estilo visual bonito y claro para los gráficos.
sns.set(style="whitegrid")

# Definimos la ecuación diferencial: dh/dt = t * sqrt(h)
# Esta función representa la derivada de h respecto al tiempo t.
def f(t, h):
    return t * np.sqrt(h)

# Calculamos la derivada de f con respecto al tiempo, para el método de Taylor de segundo orden.
def df_dt(t, h):
    return np.sqrt(h) + (t / (2 * np.sqrt(h))) * f(t, h)

# Solución exacta conocida de la ecuación: h(t) = (t²/4 + 2)²
def h_exact(t):
    return (t**2 / 4 + 2)**2

# Establecemos los valores iniciales:
h0 = 4.0     # Altura inicial
t0 = 0       # Tiempo inicial
N = 7        # Número de pasos que queremos calcular
h = 1        # Tamaño del paso (intervalo de tiempo entre cálculos)

# Creamos arreglos (listas de números) para guardar los valores de tiempo y resultados de cada método.
t_vals = np.arange(t0, t0 + (N+1)*h, h)   # Lista de tiempos desde t0 hasta t0 + N*h
euler_vals = np.zeros(N+1)               # Resultados del método de Euler
heun_vals = np.zeros(N+1)                # Resultados del método de Heun
rk4_vals = np.zeros(N+1)                 # Resultados del método Runge-Kutta 4
taylor_vals = np.zeros(N+1)              # Resultados del método de Taylor de orden 2
exact_vals = h_exact(t_vals)             # Solución exacta para todos los tiempos

# Asignamos la condición inicial a todos los métodos (en el tiempo t=0, h=4)
euler_vals[0] = heun_vals[0] = rk4_vals[0] = taylor_vals[0] = h0

# Empezamos a calcular los valores de h para cada paso, usando cada método numérico
for n in range(N):
    t_n = t_vals[n]   # Tiempo actual

    # Método de Euler: cálculo directo con la fórmula de la derivada
    euler_vals[n+1] = euler_vals[n] + h * f(t_n, euler_vals[n])

    # Método de Heun: usa una predicción y luego una corrección
    pred = heun_vals[n] + h * f(t_n, heun_vals[n])                             # Predicción (como Euler)
    corr = heun_vals[n] + (h/2) * (f(t_n, heun_vals[n]) + f(t_n + h, pred))    # Corrección promediando pendientes
    heun_vals[n+1] = corr

    # Método de Runge-Kutta 4: más preciso, combina varias pendientes
    k1 = f(t_n, rk4_vals[n])
    k2 = f(t_n + h/2, rk4_vals[n] + (h/2) * k1)
    k3 = f(t_n + h/2, rk4_vals[n] + (h/2) * k2)
    k4 = f(t_n + h, rk4_vals[n] + h * k3)
    rk4_vals[n+1] = rk4_vals[n] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

    # Método de Taylor de segundo orden: usa la derivada y su derivada
    taylor_vals[n+1] = taylor_vals[n] + h * f(t_n, taylor_vals[n]) + (h**2 / 2) * df_dt(t_n, taylor_vals[n])

# Empezamos a crear la figura con 2 gráficos uno al lado del otro
fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2.5, 1]})
fig.suptitle('Comparación de Métodos Numéricos para $\\frac{dh}{dt} = t\\sqrt{h}$', fontsize=16, fontweight='bold')

# Primer gráfico: muestra las curvas generadas por cada método
axs[0].plot(t_vals, exact_vals, 'k-', linewidth=2.5, label='Exacta')             # Línea negra: solución exacta
axs[0].plot(t_vals, euler_vals, 'o--', color='red', label='Euler')               # Círculos rojos: Euler
axs[0].plot(t_vals, heun_vals, 's-.', color='green', label='Heun')               # Cuadrados verdes: Heun
axs[0].plot(t_vals, rk4_vals, '^-', color='blue', label='Runge-Kutta 4')         # Triángulos azules: RK4
axs[0].plot(t_vals, taylor_vals, 'D:', color='purple', label='Taylor orden 2')   # Rombos morados: Taylor

axs[0].set_title('Soluciones Aproximadas', fontsize=14)
axs[0].set_xlabel('Tiempo $t$', fontsize=12)
axs[0].set_ylabel('Altura $h(t)$', fontsize=12)
axs[0].legend()   # Muestra la leyenda (nombres de las curvas)
axs[0].grid(True) # Muestra cuadrícula en el gráfico

# Segundo gráfico: tabla de resultados
axs[1].axis('off')  # Quitamos los ejes

# Creamos los datos para la tabla: cada fila representa un tiempo y los resultados de cada método
table_data = [
    [f"{t_vals[i]:.0f}", 
     f"{exact_vals[i]:.4f}",
     f"{euler_vals[i+1]:.4f}", 
     f"{heun_vals[i+1]:.4f}", 
     f"{rk4_vals[i+1]:.4f}", 
     f"{taylor_vals[i+1]:.4f}"] for i in range(N)
]

# Nombres de las columnas
col_labels = ["t", "Exacta", "Euler", "Heun", "RK4", "Taylor"]

# Creamos la tabla en el gráfico derecho
table = axs[1].table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
table.scale(1.2, 1.5)                    # Ajustamos el tamaño
table.auto_set_font_size(False)         
table.set_fontsize(10)                  # Tamaño del texto
table.auto_set_column_width(col=list(range(len(col_labels))))  # Ajuste automático del ancho

# Estilo especial para la primera fila (encabezado)
for key, cell in table.get_celld().items():
    if key[0] == 0:  # Si es la primera fila
        cell.set_text_props(weight='bold', color='white')    # Texto blanco en negrita
        cell.set_facecolor('#40466e')                        # Fondo azul oscuro
    else:
        cell.set_facecolor('#f2f2f2')                        # Fondo gris claro para los demás

axs[1].set_title("Tabla de Valores", fontweight="bold", fontsize=14)

# Ajustamos el diseño para que todo quepa bien
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()  # Mostramos la figura completa en pantalla
