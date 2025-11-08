import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Función Objetivo
def f(x):
    return (np.exp(x[0]) + 1)*(x[1]**2 + 1) - np.sin(x[0] + x[1]**2) - x[0]

# Vector Gradiente
def grad_f(x):
    
    df_dx = np.exp(x[0])*(x[1]**2 + 1) - np.cos(x[0] + x[1]**2) - 1
    df_dy = 2*x[1]*(np.exp(x[0]) + 1) - 2*x[1]*np.cos(x[0] + x[1]**2)
    return np.array([df_dx, df_dy])

# Hessiano 
def hess_f(x):
    h11 = np.exp(x[0])*(x[1]**2 + 1) + np.sin(x[0] + x[1]**2)
    h22 = 2*(np.exp(x[0]) + 1) - 2*np.cos(x[0] + x[1]**2) + 4*x[1]**2*np.sin(x[0] + x[1]**2)
    h12 = 2*x[1]*np.exp(x[0]) + 2*x[1]*np.sin(x[0] + x[1]**2)
    return np.array([[h11, h12], [h12, h22]])


# Método de Región de Confianza
def trust_region_method(x0, grad_f, hess_f, delta0=1.0, eta=0.15, tol=1e-6, max_iter=1000):
    
    # x0: punto inicial
    # grad_f: función que calcula el gradiente
    # hess_f: función que calcula el Hessiano
    # delta0: radio inicial de la región de confianza
    # eta: umbral para aceptar el paso (0 < eta < 1)
    # tol: tolerancia para criterio de parada
    # max_iter: número máximo de iteraciones

    x = np.array(x0, dtype=float)
    delta = delta0 #radio region de confianza

    history = [x.copy()] #almacenar historial de puntos


    for _ in range(max_iter):
        # 1- Calcular gradiente y Hessiana
        g = grad_f(x)
        B = hess_f(x)

        if np.linalg.norm(g) < tol: # verificar gradiente es suficientemente pequeño
            break
        # 2- Resolver el subproblema Region COnfianza
        try:
            # Intentar calcular paso de Newton
            p_newton = -np.linalg.solve(B, g)

            # Verificar paso de Newton está dentro de la Region de COnfianza
            if np.linalg.norm(p_newton) <= delta:
                p = p_newton
            else:
                #calcular paso Cauchy => dirección máximo descenso
                p_cauchy = - (np.dot(g, g) / (np.dot(g, np.dot(B, g)) + 1e-12)) * g 

                # Verificar si paso Cauchy dentro de la region de Confiaza
                if np.linalg.norm(p_cauchy) > delta:
                    p = -delta * g / np.linalg.norm(g)
                else:
                    p = delta * p_newton / np.linalg.norm(p_newton) #USAR paso Cauchy truncado al borde de la REgion
        except np.linalg.LinAlgError:
            # Hessiano Singular => usar dirección de descenso más pronunciado
            p = -delta * g / np.linalg.norm(g)

        # 3- Evaluar calidad del Paso    
        f_actual = f(x)
        # valor real funcion en el nuevo paso
        f_new = f(x + p)


        # Aproximacion Taylor 2do Orden (Modelo Cuadrático)
        m_new = f_actual + np.dot(g, p) + 0.5*np.dot(p, np.dot(B, p))
        # Razon de Reducción: reducción real / reduccion predicha
        rho = (f_actual - f_new) / (f_actual - m_new + 1e-12)
        
        # 4- Ajustar radio region de confianza
        if rho < 0.25:
            delta *= 0.25 # Reducir la region de cofianza
        elif rho > 0.75:
            delta = min(2*delta, 10) # expandir region de confianza
        
        # 5- Decición de aceptar o no el PASO
        if rho > eta:
            x = x + p # PASO ACEPTADO => moverse al nuevo punto
            history.append(x.copy())
        # PASO MUY GRANDE => PARAR    
        if np.linalg.norm(p) < tol:
            break
    return x, history


# Método ARC (Regularización Adaptativa Cúbica)
def arc_method(x0, grad_f, hess_f, sigma0=1.0, tol=1e-6, max_iter=1000):
    
    # x0: punto inicial
    # grad_f: función que calcula el gradiente
    # hess_f: función que calcula el Hessiano  
    # sigma0: parámetro de regularización inicial
    # tol: tolerancia para criterio de parada
    # max_iter: número máximo de iteraciones

    x = np.array(x0, dtype=float)
    sigma = sigma0 # Parametro Regulacion Cubica
    history = [x.copy()] # Hiatorial puntos


    for _ in range(max_iter):
        # 1- Calcular gradiente & Hessiano en el punto Actual
        g = grad_f(x)
        B = hess_f(x)

        # Verificar si el gradiente es suficientemente pequeño
        if np.linalg.norm(g) < tol:
            break
        # 2- Resolver Subproblema regularizador Cubico
        try:
            # Calcular paso => (B + sigma*I)p = -g
            p = -np.linalg.solve(B + sigma*np.eye(len(x)), g)
        except np.linalg.LinAlgError:
            p = -g / (np.linalg.norm(g) + 1e-12) # Problemas Numericos => usar direccion descenso mas pronunciado
        
        # 3- Evaluar calidad del paso
        f_actual = f(x)
        f_new = f(x + p)  # valor real de la funcion en el nuevo punto 
        p_norm = np.linalg.norm(p)

        # Taylor 2do orden + térmico cúbico (Modelo Cúbico)
        m_new = f_actual + np.dot(g, p) + 0.5*np.dot(p, np.dot(B, p)) + (sigma/3)*p_norm**3

        # Calcular razón de reducción => reducción real / reducción predicha
        rho = (f_actual - f_new) / (f_actual - m_new + 1e-12)

        # 4- Ajustar parametro regularizacion
        if rho < 0.25:
            sigma *= 2 # aumentar regularizacion => pasos más conservadores
        elif rho > 0.75:
            sigma = max(sigma/2, 1e-8) # disminuir regularizacion
        
        # 5- Decision acpetar o NO el paso-
        if rho > 1e-4:
            # Paso aceptado => moverse al nuevo punto
            x = x + p
            history.append(x.copy())

        # paso muy pequeño => PARAR     
        if np.linalg.norm(p) < tol:
            break
    return x, history


# Ejecución Métodos
np.random.seed(42)
x0 = np.random.uniform(-2, 2, size=2)
print("Punto inicial:", x0)
print("f(x0) =", f(x0))

x_tr, hist_tr = trust_region_method(x0, grad_f, hess_f)
x_arc, hist_arc = arc_method(x0, grad_f, hess_f)

print("Óptimo Región de Confianza:", x_tr, "f(x) =", f(x_tr))
print("Óptimo ARC:", x_arc, "f(x) =", f(x_arc))


# Crear malla vectorizada
x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = np.vectorize(lambda x, y: f([x, y]))(X, Y)

# Historiales como arrays
hist_tr_arr = np.array(hist_tr)
hist_arc_arr = np.array(hist_arc)
f_tr = np.array([f(x) for x in hist_tr])
f_arc = np.array([f(x) for x in hist_arc])



# GRAFICOS 2D - Primero solo los gráficos

# Crear figura solo para gráficos
fig_graphs = plt.figure(figsize=(12, 5))

# 1. Vista 2D 
ax1 = fig_graphs.add_subplot(121)
# Contornos con más niveles y colormap invertido
contour = ax1.contour(X, Y, Z, 30, cmap=cm.coolwarm_r, alpha=0.7)
ax1.clabel(contour, inline=True, fontsize=8)

# Trayectorias 2D
ax1.plot(hist_tr_arr[:,0], hist_tr_arr[:,1], 'o-', color='red', linewidth=2, 
         markersize=6, markerfacecolor='darkred', markeredgecolor='black', 
         markeredgewidth=0.5, label='Región de Confianza')
ax1.plot(hist_arc_arr[:,0], hist_arc_arr[:,1], 's-', color='blue', linewidth=2, 
         markersize=6, markerfacecolor='darkblue', markeredgecolor='black', 
         markeredgewidth=0.5, label='ARC')

# Puntos importantes
ax1.scatter([x0[0]], [x0[1]], color='green', s=150, marker='*', 
           edgecolor='black', linewidth=1, label='Punto Inicial', zorder=5)
ax1.scatter([x_tr[0]], [x_tr[1]], color='red', s=150, marker='*', 
           edgecolor='black', linewidth=1, label='Óptimo TR', zorder=5)
ax1.scatter([x_arc[0]], [x_arc[1]], color='blue', s=150, marker='*', 
           edgecolor='black', linewidth=1, label='Óptimo ARC', zorder=5)

ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('y', fontsize=12, fontweight='bold')
ax1.set_title('Trayectorias de Optimización', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.7)
ax1.set_aspect('equal')

# 2. Gráfico de convergencia
ax2 = fig_graphs.add_subplot(122)
f_min = min(f_tr[-1], f_arc[-1])

ax2.semilogy(np.arange(len(f_tr)), f_tr - f_min, 'o-', color='red', linewidth=2.5,
             markersize=6, markerfacecolor='darkred', markeredgecolor='black',
             markeredgewidth=0.5, label='Región de Confianza')
ax2.semilogy(np.arange(len(f_arc)), f_arc - f_min, 's-', color='blue', linewidth=2.5,
             markersize=6, markerfacecolor='darkblue', markeredgecolor='black',
             markeredgewidth=0.5, label='ARC')

ax2.set_xlabel('Iteración', fontsize=12, fontweight='bold')
ax2.set_ylabel('f(x) - f*', fontsize=12, fontweight='bold')
ax2.set_title('Convergencia de la Función Objetivo', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.7)

# Ajustar layout y mostrar gráficos
plt.tight_layout()
plt.show()

# TABLA VISUAL - Figura separada
fig_table = plt.figure(figsize=(10, 4))
ax_table = fig_table.add_subplot(111)
ax_table.axis('tight')
ax_table.axis('off')

# Calcular métricas adicionales
norm_grad_tr = np.linalg.norm(grad_f(x_tr))
norm_grad_arc = np.linalg.norm(grad_f(x_arc))
path_length_tr = np.sum(np.linalg.norm(np.diff(hist_tr_arr, axis=0), axis=1))
path_length_arc = np.sum(np.linalg.norm(np.diff(hist_arc_arr, axis=0), axis=1))

# Datos para la tabla
table_data = [
    ['Métrica', 'Región de Confianza', 'ARC'],
    ['Iteraciones', f'{len(hist_tr)}', f'{len(hist_arc)}'],
    ['f(x) final', f'{f_tr[-1]:.6e}', f'{f_arc[-1]:.6e}'],
    ['||∇f(x)|| final', f'{norm_grad_tr:.2e}', f'{norm_grad_arc:.2e}'],
    ['Longitud trayectoria', f'{path_length_tr:.4f}', f'{path_length_arc:.4f}'],
    ['Punto óptimo', f'({x_tr[0]:.4f}, {x_tr[1]:.4f})', f'({x_arc[0]:.4f}, {x_arc[1]:.4f})']
]

# Crear tabla
table = ax_table.table(cellText=table_data, 
                      cellLoc='center', 
                      loc='center',
                      colWidths=[0.3, 0.35, 0.35])

# Estilo de la tabla
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Colores y estilo de celdas
for i, key in enumerate(table.get_celld().keys()):
    cell = table.get_celld()[key]
    if key[0] == 0:  # Encabezado
        cell.set_facecolor('#4B8BBE')
        cell.set_text_props(weight='bold', color='white', size=12)
    elif key[0] % 2 == 1:  # Filas impares
        cell.set_facecolor('#F9F9F9')
    else:  # Filas pares
        cell.set_facecolor('#FFFFFF')
    
    # Negrita para la primera columna (nombres de métricas)
    if key[1] == 0 and key[0] > 0:
        cell.set_text_props(weight='bold')

# Título de la tabla
ax_table.set_title('COMPARACIÓN DE ALGORITMOS DE OPTIMIZACIÓN', 
                  fontsize=16, fontweight='bold', pad=20)

# Ajustar layout y mostrar tabla
plt.tight_layout()
plt.show()

# Información adicional de convergencia
print(f"\n--- Estadísticas de Convergencia ---")
print(f"Punto inicial: ({x0[0]:.4f}, {x0[1]:.4f}) | f(x0) = {f(x0):.6e}")
print(f"Región de Confianza: {len(hist_tr)} iteraciones, f final = {f_tr[-1]:.6e}")
print(f"ARC: {len(hist_arc)} iteraciones, f final = {f_arc[-1]:.6e}")
print(f"Diferencia entre métodos: {abs(f_tr[-1] - f_arc[-1]):.2e}")

# Análisis comparativo
print(f"\n--- Análisis Comparativo ---")
if abs(f_tr[-1] - f_arc[-1]) < 1e-8:
    print("✅ Ambos métodos convergen esencialmente al mismo valor óptimo")
elif f_tr[-1] < f_arc[-1]:
    print("🏆 Región de Confianza encuentra una solución ligeramente mejor")
else:
    print("🏆 ARC encuentra una solución ligeramente mejor")

if len(hist_tr) < len(hist_arc):
    print(f"⚡ Región de Confianza es {len(hist_arc)/len(hist_tr):.1f}x más rápido")
else:
    print(f"⚡ ARC es {len(hist_tr)/len(hist_arc):.1f}x más rápido")