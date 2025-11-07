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



# GRAFICOS

# Crear figura con subplots
fig = plt.figure(figsize=(18, 6))

# 1. Gráfico 3D (mejorado)
ax1 = fig.add_subplot(131, projection='3d')
# Superficie con colormap invertido (como en el documento)
surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm_r, alpha=0.8, 
                       linewidth=0, antialiased=True, rstride=2, cstride=2)

# Trayectorias más destacadas
ax1.plot(hist_tr_arr[:,0], hist_tr_arr[:,1], f_tr, 'o-', 
         color='red', linewidth=2.5, markersize=6, markerfacecolor='darkred', 
         markeredgecolor='black', markeredgewidth=0.5, label='Región de Confianza')

ax1.plot(hist_arc_arr[:,0], hist_arc_arr[:,1], f_arc, 's-', 
         color='blue', linewidth=2.5, markersize=6, markerfacecolor='darkblue', 
         markeredgecolor='black', markeredgewidth=0.5, label='ARC')

# Puntos importantes más grandes
ax1.scatter([x0[0]], [x0[1]], [f(x0)], color='green', s=150, 
           marker='*', edgecolor='black', linewidth=1, label='Punto Inicial', zorder=5)
ax1.scatter([x_tr[0]], [x_tr[1]], [f(x_tr)], color='red', s=150, 
           marker='*', edgecolor='black', linewidth=1, label='Óptimo TR', zorder=5)
ax1.scatter([x_arc[0]], [x_arc[1]], [f(x_arc)], color='blue', s=150, 
           marker='*', edgecolor='black', linewidth=1, label='Óptimo ARC', zorder=5)

ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('y', fontsize=12, fontweight='bold')
ax1.set_zlabel('f(x,y)', fontsize=12, fontweight='bold')
ax1.set_title('Superficie 3D y Trayectorias de Optimización', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Vista 2D (contornos)
ax2 = fig.add_subplot(132)
# Contornos con más niveles y colormap invertido
contour = ax2.contour(X, Y, Z, 30, cmap=cm.coolwarm_r, alpha=0.7)
ax2.clabel(contour, inline=True, fontsize=8)

# Trayectorias 2D
ax2.plot(hist_tr_arr[:,0], hist_tr_arr[:,1], 'o-', color='red', linewidth=2, 
         markersize=6, markerfacecolor='darkred', markeredgecolor='black', 
         markeredgewidth=0.5, label='Región de Confianza')
ax2.plot(hist_arc_arr[:,0], hist_arc_arr[:,1], 's-', color='blue', linewidth=2, 
         markersize=6, markerfacecolor='darkblue', markeredgecolor='black', 
         markeredgewidth=0.5, label='ARC')

# Puntos importantes
ax2.scatter([x0[0]], [x0[1]], color='green', s=150, marker='*', 
           edgecolor='black', linewidth=1, label='Punto Inicial', zorder=5)
ax2.scatter([x_tr[0]], [x_tr[1]], color='red', s=150, marker='*', 
           edgecolor='black', linewidth=1, label='Óptimo TR', zorder=5)
ax2.scatter([x_arc[0]], [x_arc[1]], color='blue', s=150, marker='*', 
           edgecolor='black', linewidth=1, label='Óptimo ARC', zorder=5)

ax2.set_xlabel('x', fontsize=12, fontweight='bold')
ax2.set_ylabel('y', fontsize=12, fontweight='bold')
ax2.set_title('Trayectorias (Vista Superior)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.7)
ax2.set_aspect('equal')

# 3. Gráfico de convergencia (mejorado)
ax3 = fig.add_subplot(133)
f_min = min(f_tr[-1], f_arc[-1])

# Líneas más gruesas y marcadores
ax3.semilogy(np.arange(len(f_tr)), f_tr - f_min, 'o-', color='red', linewidth=2.5,
             markersize=6, markerfacecolor='darkred', markeredgecolor='black',
             markeredgewidth=0.5, label='Región de Confianza')
ax3.semilogy(np.arange(len(f_arc)), f_arc - f_min, 's-', color='blue', linewidth=2.5,
             markersize=6, markerfacecolor='darkblue', markeredgecolor='black',
             markeredgewidth=0.5, label='ARC')

ax3.set_xlabel('Iteración', fontsize=12, fontweight='bold')
ax3.set_ylabel('f(x) - f*', fontsize=12, fontweight='bold')
ax3.set_title('Convergencia de la Función Objetivo', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.7)

# Ajustar layout y mostrar
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Información adicional de convergencia
# ---------------------------------------------------

print(f"\n--- Estadísticas de Convergencia ---")
print(f"Región de Confianza: {len(hist_tr)} iteraciones, f final = {f_tr[-1]:.6e}")
print(f"ARC: {len(hist_arc)} iteraciones, f final = {f_arc[-1]:.6e}")
print(f"Diferencia entre métodos: {abs(f_tr[-1] - f_arc[-1]):.2e}")