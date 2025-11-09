import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
    x = np.array(x0, dtype=float)
    delta = delta0 # radio region de confianza
    history = [x.copy()] # almacenar historial de puntos

    for _ in range(max_iter):
        g = grad_f(x)
        B = hess_f(x)

        if np.linalg.norm(g) < tol:
            break
        try:
            p_newton = -np.linalg.solve(B, g)
            if np.linalg.norm(p_newton) <= delta:
                p = p_newton
            else:
                p_cauchy = - (np.dot(g, g) / (np.dot(g, np.dot(B, g)) + 1e-12)) * g 
                if np.linalg.norm(p_cauchy) > delta:
                    p = -delta * g / np.linalg.norm(g)
                else:
                    p = delta * p_newton / np.linalg.norm(p_newton)
        except np.linalg.LinAlgError:
            p = -delta * g / np.linalg.norm(g)

        f_actual = f(x)
        f_new = f(x + p)
        m_new = f_actual + np.dot(g, p) + 0.5*np.dot(p, np.dot(B, p))
        rho = (f_actual - f_new) / (f_actual - m_new + 1e-12)

        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75:
            delta = min(2*delta, 10)

        if rho > eta:
            x = x + p
            history.append(x.copy())

        if np.linalg.norm(p) < tol:
            break
    return x, history


# Método ARC (Regularización Adaptativa Cúbica)
def arc_method(x0, grad_f, hess_f, sigma0=1.0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    sigma = sigma0 # Parametro Regulacion Cubica
    history = [x.copy()] # Historial de puntos

    for _ in range(max_iter):
        g = grad_f(x)
        B = hess_f(x)

        if np.linalg.norm(g) < tol:
            break
        try:
            p = -np.linalg.solve(B + sigma*np.eye(len(x)), g)
        except np.linalg.LinAlgError:
            p = -g / (np.linalg.norm(g) + 1e-12)
        
        f_actual = f(x)
        f_new = f(x + p)
        p_norm = np.linalg.norm(p)
        m_new = f_actual + np.dot(g, p) + 0.5*np.dot(p, np.dot(B, p)) + (sigma/3)*p_norm**3
        rho = (f_actual - f_new) / (f_actual - m_new + 1e-12)

        if rho < 0.25:
            sigma *= 2
        elif rho > 0.75:
            sigma = max(sigma/2, 1e-8)
        
        if rho > 1e-4:
            x = x + p
            history.append(x.copy())

        if np.linalg.norm(p) < tol:
            break
    return x, history


# ========================
# PRUEBAS EN TRES RANGOS
# ========================
rangos = [[-2, 2], [-10, 10], [-100, 100]]

for r in rangos:
    np.random.seed(42)
    x0 = np.random.uniform(r[0], r[1], size=2)
    print(f"\n===== RANGO {r} =====")
    print("Punto inicial:", x0)
    print("f(x0) =", f(x0))

    x_tr, hist_tr = trust_region_method(x0, grad_f, hess_f)
    x_arc, hist_arc = arc_method(x0, grad_f, hess_f)

    print("Óptimo Región de Confianza:", x_tr, "f(x) =", f(x_tr))
    print("Óptimo ARC:", x_arc, "f(x) =", f(x_arc))

    # Crear malla vectorizada
    x_range = np.linspace(r[0], r[1], 200)
    y_range = np.linspace(r[0], r[1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.vectorize(lambda x, y: f([x, y]))(X, Y)

    hist_tr_arr = np.array(hist_tr)
    hist_arc_arr = np.array(hist_arc)
    f_tr = np.array([f(x) for x in hist_tr])
    f_arc = np.array([f(x) for x in hist_arc])

    # GRAFICOS 2D
    fig_graphs = plt.figure(figsize=(12, 5))
    fig_graphs.suptitle(f"RANGO {r}", fontsize=16, fontweight='bold')

    # 1. Vista 2D 
    ax1 = fig_graphs.add_subplot(121)
    contour = ax1.contour(X, Y, Z, 30, cmap=cm.coolwarm_r, alpha=0.7)
    ax1.clabel(contour, inline=True, fontsize=8)

    ax1.plot(hist_tr_arr[:,0], hist_tr_arr[:,1], 'o-', color='red', linewidth=2, 
            markersize=6, markerfacecolor='darkred', markeredgecolor='black', 
            markeredgewidth=0.5, label='Región de Confianza')
    ax1.plot(hist_arc_arr[:,0], hist_arc_arr[:,1], 's-', color='blue', linewidth=2, 
            markersize=6, markerfacecolor='darkblue', markeredgecolor='black', 
            markeredgewidth=0.5, label='ARC')

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

    plt.tight_layout()
    plt.show()
