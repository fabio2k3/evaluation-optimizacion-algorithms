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



# EJECUCIÓN EN TRES RANGOS

rangos = [(-2, 2), (-10, 10), (-100, 100)]
resultados = []

for rmin, rmax in rangos:
    np.random.seed(42)
    x0 = np.random.uniform(rmin, rmax, size=2)
    x_tr, hist_tr = trust_region_method(x0, grad_f, hess_f)
    x_arc, hist_arc = arc_method(x0, grad_f, hess_f)
    resultados.append({
        "rango": f"[{rmin}, {rmax}]",
        "x0": x0,
        "x_tr": x_tr,
        "x_arc": x_arc,
        "hist_tr": hist_tr,
        "hist_arc": hist_arc,
        "f_tr": np.array([f(x) for x in hist_tr]),
        "f_arc": np.array([f(x) for x in hist_arc])
    })

# *** EXPORTACION A JSON ***
import json
import os
import numpy as np
from datetime import datetime

# Función para convertir arrays de numpy a tipos nativos de Python para JSON
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    return obj

# Exportar resultados a JSON
def export_results_to_json(resultados, filename=None):
    # Si no se especifica filename, crear uno automáticamente en la raíz
    if filename is None:
        # Subir dos niveles desde Codes/ para llegar a la raíz del proyecto
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = os.path.join(project_root, "Result.json")
    
    # Preparar datos en formato de tabla para fácil comparación
    tabla_comparativa = []
    
    for res in resultados:
        # Datos para TR
        tabla_comparativa.append({
            "rango": res["rango"],
            "metodo": "Región de Confianza",
            "iteraciones": len(res["hist_tr"]),
            "f_x_final": float(res["f_tr"][-1]),
            "norma_gradiente_final": float(np.linalg.norm(grad_f(res["x_tr"]))),
            "punto_optimo": convert_to_serializable(res["x_tr"])
        })
        
        # Datos para ARC
        tabla_comparativa.append({
            "rango": res["rango"],
            "metodo": "ARC",
            "iteraciones": len(res["hist_arc"]),
            "f_x_final": float(res["f_arc"][-1]),
            "norma_gradiente_final": float(np.linalg.norm(grad_f(res["x_arc"]))),
            "punto_optimo": convert_to_serializable(res["x_arc"])
        })
    
    # Crear estructura completa del resultado
    export_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "funcion_objetivo": "f(x) = (exp(x) + 1)*(y² + 1) - sin(x + y²) - x",
            "algoritmos": ["Trust Region", "ARC"],
            "rangos_evaluados": [str(rango) for rango in rangos],
            "total_resultados": len(tabla_comparativa)
        },
        "tabla_comparativa": tabla_comparativa,
        "datos_completos": {
            "resultados_detallados": [
                {
                    "rango": res["rango"],
                    "punto_inicial": convert_to_serializable(res["x0"]),
                    "trust_region": {
                        "punto_optimo": convert_to_serializable(res["x_tr"]),
                        "historial_iteraciones": convert_to_serializable(np.array(res["hist_tr"])),
                        "valores_funcion": convert_to_serializable(res["f_tr"]),
                        "iteraciones_totales": len(res["hist_tr"]),
                        "valor_final": float(res["f_tr"][-1]),
                        "norma_gradiente_final": float(np.linalg.norm(grad_f(res["x_tr"])))
                    },
                    "arc": {
                        "punto_optimo": convert_to_serializable(res["x_arc"]),
                        "historial_iteraciones": convert_to_serializable(np.array(res["hist_arc"])),
                        "valores_funcion": convert_to_serializable(res["f_arc"]),
                        "iteraciones_totales": len(res["hist_arc"]),
                        "valor_final": float(res["f_arc"][-1]),
                        "norma_gradiente_final": float(np.linalg.norm(grad_f(res["x_arc"])))
                    }
                }
                for res in resultados
            ]
        }
    }
    
    # Exportar a JSON
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=convert_to_serializable)
        
        # Verificación
        file_path = os.path.abspath(filename)
        print(f"✅ Resultados exportados exitosamente a: {file_path}")
        print(f"📊 Estructura del archivo:")
        print(f"   - Metadatos de ejecución")
        print(f"   - Tabla comparativa ({len(tabla_comparativa)} registros)")
        print(f"   - Datos completos ({len(resultados)} rangos con información detallada)")
        
        # Verificar que el archivo se creó
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"📏 Tamaño del archivo: {file_size} bytes")
        else:
            print(f"❌ El archivo no se creó en la ruta esperada")
            
    except Exception as e:
        print(f"❌ Error al exportar resultados: {e}")
        import traceback
        traceback.print_exc()

# LLAMAR LA FUNCIÓN DE EXPORTACIÓN DESDE Algorithms.py
# Opción 1: Ruta manual explícita
# export_results_to_json(resultados, "../../Result.json")

# Opción 2: Detección automática (RECOMENDADA)
export_results_to_json(resultados)
# *** FIN JSON ***


# GRAFICAR [-2,2]

res = resultados[0]
x0, x_tr, x_arc = res["x0"], res["x_tr"], res["x_arc"]
hist_tr_arr = np.array(res["hist_tr"])
hist_arc_arr = np.array(res["hist_arc"])
f_tr, f_arc = res["f_tr"], res["f_arc"]

# Crear malla vectorizada
x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = np.vectorize(lambda x, y: f([x, y]))(X, Y)

# GRAFICOS 2D 

fig_graphs = plt.figure(figsize=(12, 5))

# Vista 2D 
ax1 = fig_graphs.add_subplot(121)
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
ax1.set_title('Trayectorias de Optimización [-2,2]', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.7)
ax1.set_aspect('equal')

# Gráfico de convergencia
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
ax2.set_title('Convergencia de la Función Objetivo [-2,2]', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.7)

plt.tight_layout()
plt.show()


# TABLA COMPARATIVA (3 RANGOS)

fig_table = plt.figure(figsize=(11, 4))
ax_table = fig_table.add_subplot(111)
ax_table.axis('tight')
ax_table.axis('off')

# Datos de la tabla
table_data = [['Rango', 'Método', 'Iteraciones', 'f(x) final', '||∇f(x)|| final', 'Punto óptimo']]

for res in resultados:
    for metodo, hist, fx, xopt in [
        ('Región de Confianza', res['hist_tr'], res['f_tr'], res['x_tr']),
        ('ARC', res['hist_arc'], res['f_arc'], res['x_arc'])
    ]:
        table_data.append([
            res['rango'],
            metodo,
            len(hist),
            f"{fx[-1]:.6e}",
            f"{np.linalg.norm(grad_f(xopt)):.2e}",
            f"({xopt[0]:.4f}, {xopt[1]:.4f})"
        ])

# Crear tabla
table = ax_table.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

for i, key in enumerate(table.get_celld().keys()):
    cell = table.get_celld()[key]
    if key[0] == 0:
        cell.set_facecolor('#4B8BBE')
        cell.set_text_props(weight='bold', color='white')

ax_table.set_title('COMPARACIÓN DE ALGORITMOS EN DIFERENTES RANGOS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
