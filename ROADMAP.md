### V2.1 Corrección, validación y UX técnica

Objetivo: resultados más predecibles y configuración robusta.

1. Validación consistente de configuraciones
   • Generaliza la validación a todos los modos. Ahora solo `validate_hybrid_config` hace chequeos serios. Extrae un validador común que verifique: suma de pesos por bloque, índices, existencia de claves y forma de tensores cuando hay locks. Ganchos en `XLFusion.py` antes de llamar a la fusión.  

Criterio de aceptación: una configuración inválida nunca llega a `merge_*`; los mensajes de error indican bloque, índice y causa.

2. Arreglar semántica para N modelos en CLI
   • `prompt_block_merge` y `prompt_crossattn_boost` no deben asignar “1 − w” a todos los modelos no cero. Cambia a un input vectorial por modelo o a normalización automática de la fila. Actualiza GUI Hybrid y Legacy para denotar distribución correcta por bloque.  

Criterio: con 3 modelos puedes definir boosts y multiplicadores por bloque y la suma se normaliza a 1.0 por bloque.

3. Modo “plan” o preflight
   • En CLI y GUI, añade un paso que calcule memoria estimada, número de claves afectadas por bloque y advertencias de compatibilidad antes de ejecutar. Ya usas estimación de memoria dentro de `merge_*`; expón ese cálculo y llámalo en la vista previa.  

Criterio: el usuario ve memoria estimada, conteo por bloque, locks efectivos y avisos antes de pulsar “Iniciar fusión”.

4. Metadatos fortalecidos
   • En `save_merge_results` agrega huellas BLAKE2 de cada checkpoint y LoRA, versión de torch, parámetros exactos por bloque y device. Es el sitio adecuado para ampliar lo que luego rehidratas vía YAML. 

Criterio: `metadata.txt` y el metadato embebido incluyen hashes de entrada y configuración completa reproducible.

5. Progreso determinista
   • En GUI reemplaza la barra indeterminada por progreso real usando `len(base_keys)` del backbone y cuenta de claves procesadas en los bucles de `merge_*`. Ya iteras clave a clave con tqdm. Propaga ticks a la cola de log.  

---

### V2.2 Rendimiento y escalabilidad

Objetivo: acelerar sin romper memoria.

1. E/S paralela segura
   • Las lecturas desde `safe_open` son I/O bound. Paraleliza la extracción por clave entre modelos con un pool de hilos o prefetch de tensores en colas. Mantén la suma en el hilo principal para no fragmentar memoria. 

Criterio: mejora del 15 a 30 % en throughput en discos NVMe en pruebas con 2 y 3 modelos.

2. Lote por bloque
   • En Hybrid y PerRes, procesa claves agrupadas por bloque para minimizar cambios de manejador. Ya tienes `get_block_assignment` y stats por bloque. Ordena las listas antes de iterar. 

---

### V2.3 Extensibilidad y limpieza interna

Objetivo: desacoplar y permitir ampliar.

1. Motor de plantillas único
   • Ahora hay dos sistemas de plantillas: uno simple en `templates.py` y otro con evaluación segura en el batch via `interpolate_params`. Unifica el segundo como servicio reusable y úsalo en CLI y GUI cuando el usuario seleccione una plantilla.  

2. Plugins de mapeo de bloques
   • `blocks.get_block_assignment` gobierna todo. Define un registro de patrones para admitir arquitecturas o particionados alternativos sin tocar `merge.py` ni `analyzer.py`.  

---

### V2.4 Calidad de fusión y análisis avanzado

Objetivo: que las decisiones de mezcla se apoyen en métricas útiles.

1. Analítica cuantitativa ampliada
   • En `analyzer.py` amplía el muestreo de similitud y añade métricas por submódulo y capa. Registra histogramas de cosenos y L2 por bloque y un score de “coherencia estructural” independiente de estilo. 

2. Autoajuste de pesos
   • Implementa una búsqueda de cuadrícula o bayesiana sobre 2 o 3 grados de libertad por bloque, con la métrica anterior como objetivo y restricciones de dominancia máxima por modelo. Devuelve una plantilla Hybrid recomendada. 

3. Informe de previsión en GUI
   • Integra `FusionPredictor` en la vista previa. Muestra dominancia prevista por bloque y alerta de baja diversidad.  

---

### V2.5 Reproducibilidad, auditoría y seguridad

Objetivo: trazabilidad fuerte y ejecución segura.

1. Auditoría exhaustiva
   • `BatchProcessor` ya crea un log y YAML. Añade registro de hashes de entrada, huellas del entorno y un inventario de claves realmente tomadas del backbone vs. sustituidas. 

2. Esquema de configuración validado
   • Define un esquema pydantic para los YAML de batch y para los presets de GUI. Valida antes de ejecutar en todos los modos. 

3. Modo “sólo UNet” opcional en todos los modos
   • En Legacy ya existe `only_unet=True`. Expónlo en CLI y GUI y añade la opción de incluir VAE y text encoder cuando el usuario quiera. Cambios mínimos en `stream_weighted_merge_from_paths` y en orquestación.  

---

### V2.6 Ampliaciones funcionales de alto impacto

Objetivo: personalización real y casos avanzados.

1. Álgebra de checkpoints
   • Añade operaciones A + α(B − C) en Legacy y Hybrid. Nueva bandera CLI y controles GUI. Se implementa reusando el streaming: carga clave de B y C, calcula delta y añade a A según pesos del bloque. 

Criterio: prueba unitaria con tensores sintéticos que verifica que el resultado coincide con la aritmética.

2. Soporte LoRA ampliado
   • `lora.py` solo cubre UNet. Extiende mapeos para text encoders cuando existan claves LoRA correspondientes y añade validación de forma. Reporte de aplicados y omitidos por submódulo. 

3. Plantillas guiadas “objetivo” en GUI
   • Inserta las plantillas de `templates.py` en el paso de configuración con explicación y parámetros editables, usando el motor unificado del punto V2.3.  
