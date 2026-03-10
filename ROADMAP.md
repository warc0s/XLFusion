# XLFusion Roadmap

Estado actual: `main` ya cubre V2.1 con CLI, GUI, batch, analisis, metadatos reproducibles, validacion comun y preflight. Este roadmap recoge solo mejoras futuras que todavia aportan valor real a la aplicacion.

## Principios para siguientes versiones

- Priorizar fiabilidad antes que mas modos de fusion.
- Mejorar memoria y tiempos sin sacrificar reproducibilidad.
- Unificar experiencia entre CLI, GUI y batch.
- Convertir el analisis en ayuda real para tomar decisiones de mezcla.

## V2.2 Flujo experto y alto rendimiento

Objetivo: hacer que XLFusion escale mejor en memoria, sea mas rapido y permita reutilizar configuraciones complejas sin friccion.

### 1. Modo low-memory tensor a tensor

- Introducir un cargador estilo `safe_open` tensor a tensor para rutas de memoria limitada.
- Preservar dtype en disco y convertir solo durante acumulacion cuando haga falta.
- Exponer esta ruta como opcion clara en CLI, GUI y batch.

Criterio de aceptacion:
- Fusionar modelos grandes consume menos RAM que la ruta actual y sigue generando resultados equivalentes dentro de tolerancia numerica.

### 2. Pre-carga y ordenacion por bloque

- Procesar claves agrupadas por bloque para reducir overhead de acceso.
- Evaluar prefetch ligero entre modelos cuando el almacenamiento lo permita.
- Medir mejoras con benchmarks sobre 2, 3 y 4 checkpoints.

Criterio de aceptacion:
- Documentar mejora medible en throughput frente a la ruta actual en escenarios repetibles.

### 3. Progreso ligero fuera de TTY

- Mantener `tqdm` donde aporta valor, pero ofrecer una salida de progreso simple para logs, entornos no interactivos y empaquetados.
- Evitar ruido excesivo en batch y facilitar cancelacion/observabilidad.

Criterio de aceptacion:
- La aplicacion informa progreso util tanto en terminal interactiva como en logs o GUI sin duplicar salidas.

### 4. Presets reutilizables de verdad

- Unificar `templates`, presets de GUI y configuraciones batch en un solo sistema.
- Permitir guardar una configuracion creada en GUI o CLI como preset reutilizable.
- Permitir cargar presets desde metadata previa.

Criterio de aceptacion:
- Una fusion configurada en GUI puede guardarse y relanzarse por batch sin editar a mano.

### 5. Recuperacion desde metadatos

- Añadir comando para reconstruir una ejecucion a partir de `metadata/meta_*`.
- Rehidratar configuracion exacta, entradas y nombre de salida propuesto.
- Señalar diferencias si faltan modelos o LoRAs originales.

Criterio de aceptacion:
- Desde una carpeta de metadata se puede relanzar la fusion o generar un YAML equivalente.

### 6. UX de configuracion mas segura

- Mejorar formularios de GUI para pesos por bloque, locks y LoRAs con validacion inline.
- En CLI, simplificar prompts complejos y mostrar defaults mas legibles.
- Añadir confirmacion final con resumen antes de ejecutar.

Criterio de aceptacion:
- Configurar una fusion compleja de 3 o 4 modelos requiere menos pasos y menos errores de entrada.

## V2.3 Calidad de fusion y analisis accionable

Objetivo: pasar de "ver datos" a "tomar mejores decisiones de mezcla".

### 1. Analisis por submodulo y capa

- Ampliar el analizador con metricas por submodulo, histogramas y resumen por zonas del modelo.
- Separar mejor estructura, semantica y estilo para que la recomendacion no sea solo un score global.

Criterio de aceptacion:
- El informe ayuda a entender que modelo domina composicion, detalle y estilo.

### 2. Recomendador de pesos y bloques

- Generar sugerencias iniciales de `hybrid_config`, `assignments` y backbone a partir del analisis.
- Ofrecer perfiles como `balanced`, `style transfer`, `detail recovery` o `prompt fidelity`.

Criterio de aceptacion:
- El usuario puede partir de una propuesta razonable sin configurar todo a mano.

### 3. Alertas de compatibilidad antes de fusionar

- Detectar diferencias potencialmente peligrosas antes de ejecutar: shapes incompatibles, modelos demasiado alejados, locks incoherentes o combinaciones con bajo valor esperado.
- Integrar estas alertas en preflight y GUI.

Criterio de aceptacion:
- Las combinaciones de alto riesgo se detectan antes de gastar tiempo y memoria en una fusion fallida o mediocre.

### 4. Algebra de checkpoints

- Añadir operaciones tipo `A + alpha(B - C)` y variantes compatibles con `legacy` y `hybrid`.
- Reutilizar el motor streaming para evitar disparar memoria.
- Exponerlo como modo avanzado, no como sustituto del flujo principal.

Criterio de aceptacion:
- Hay tests sinteticos que verifican la aritmetica tensorial y la salida queda auditada en metadata.

### 5. Soporte LoRA ampliado

- Extender el horneado de LoRAs mas alla de UNet cuando existan claves compatibles para text encoders.
- Añadir mejor validacion de shape y un reporte por submodulo aplicado/omitido.

Criterio de aceptacion:
- El usuario sabe exactamente que partes de la LoRA se aplicaron y cuales no.

### 6. Exponer mezcla no solo UNet

- Convertir `only_unet` en una opcion visible y soportada en todos los modos.
- Permitir incluir o excluir VAE y text encoder de forma explicita cuando tenga sentido.

Criterio de aceptacion:
- La configuracion deja claro que componentes se mezclan y esa decision queda guardada en metadata.

## V2.4 Plataforma y evolucion interna

Objetivo: facilitar que el proyecto siga creciendo sin repetir logica ni abrir regresiones.

### 1. API interna mas clara

- Delimitar mejor capas de `config`, `merge`, `workflow`, `analysis` y GUI.
- Reducir duplicacion entre CLI interactiva, batch y GUI.
- Formalizar tipos de configuracion compartidos.

### 2. Cobertura de tests orientada a regresiones reales

- Priorizar tests sobre validacion, cancelacion, metadata reproducible, modos low-memory y algebra de checkpoints.
- Añadir fixtures sinteticos para comparar salidas entre distintas rutas de ejecucion.

### 3. Arquitecturas futuras sin tocar el core

- Preparar un registro de mapeos de bloques para soportar otras particiones o arquitecturas derivadas sin modificar el motor principal.
- Mantener SDXL como ruta principal, pero evitar acoplamientos innecesarios.

## Prioridades recomendadas

1. V2.2 Flujo experto y alto rendimiento
2. V2.3 Calidad de fusion y analisis accionable
3. V2.4 Plataforma y evolucion interna
