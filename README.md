# XLFusion V2.0

Fusionador profesional de checkpoints SDXL con CLI y GUI, orientado a reproducibilidad, control por bloques y workflows por lotes. Incluye tres modos de fusión (Legacy, PerRes y Hybrid), horneado de LoRAs, análisis avanzado y metadatos completos.

## Novedades V2

- GUI nativa (Tk) con asistente paso a paso y vista previa por bloques.
- Modo Hybrid: asignación por bloque + mezcla ponderada.
- Procesamiento por lotes (batch) con validación, logs y YAML reproducible.
- Análisis (diff, compatibilidad, predicción, recomendaciones).
- Metadatos enriquecidos: hashes BLAKE2 de entradas, YAML autocontenible y versionado automático.

## Modos de fusión

- Legacy (ponderado clásico)
  - Pesos globales por modelo, control opcional por bloques gruesos `down/mid/up` y multiplicadores por grupos `down_0_1`, `down_2_3`, `mid`, `up_0_1`, `up_2_3`.
  - Cross-attention boost (mejora adherencia al prompt).
  - Horneado de LoRAs en el resultado.
  - Proceso streaming optimizado en memoria.

- PerRes (por resolución)
  - Asignación 100% por grupo de bloques: `down_0_1`, `down_2_3`, `mid`, `up_0_1`, `up_2_3`.
  - Locks opcionales de cross-attention (`down/mid/up`).

- Hybrid (PerRes + mezcla)
  - Por bloque defines pesos por modelo (suma ≈ 1.0), con soporte de locks.
  - Ideal para transferir estilo en `up_*` y preservar composición en `down_*`.

Compatibilidad técnica:
- Modelos SDXL derivados (NoobAI, Illustrious, Pony, etc.).
- Entradas/Salidas `.safetensors` compatibles A1111/ComfyUI.

## Requisitos e instalación

Requiere Python 3.10+ y los paquetes de `requirements.txt`:

```
pip install -r requirements.txt
```

Paquetes principales: `torch`, `safetensors`, `PyYAML`, `numpy`, `tqdm`, `psutil`.

Nota: Si empaquetas la GUI en Windows, instala además `pyinstaller` y usa `scripts/build_gui_exe.py`.

## Estructura de carpetas

```
XLFusion/
├── XLFusion.py                 # Entrada CLI principal
├── gui_app.py                  # Interfaz gráfica (V2)
├── config.yaml                 # Configuración centralizada
├── Utils/                      # Módulos internos (merge, lora, batch, analyzer, ...)
├── models/                     # Checkpoints de entrada (.safetensors)
├── loras/                      # LoRAs (.safetensors) opcionales
├── output/                     # Resultados fusionados
├── metadata/                   # Metadatos y auditorías
├── scripts/                    # Utilidades (batch, smoke, build exe)
└── tests/                      # Pruebas unitarias
```

## Uso rápido (CLI interactivo)

1) Coloca modelos `.safetensors` en `models/` y (opcional) LoRAs en `loras/`.
2) Ejecuta:

```
python XLFusion.py
```

3) Selecciona modelos, el modo de fusión y ajusta la configuración.
4) El resultado se guarda en `output/` y la auditoría en `metadata/`.

## GUI (Asistente gráfico)

Lanza la GUI con:

```
python XLFusion.py --gui
```

Características de la GUI:
- Biblioteca de modelos con tamaño, selección múltiple y orden.
- Configuración guiada por modo (Legacy, PerRes, Hybrid) y LoRAs.
- Vista previa por bloques con pesos/asignaciones.
- Progreso real y cancelación segura.

## Procesamiento por lotes (Batch)

Permite definir múltiples trabajos en un `YAML` y procesarlos secuencialmente con validación y logs.

- Ejecutar batch:

```
python XLFusion.py --batch batch_config_example.yaml
```

- Validar únicamente:

```
python XLFusion.py --batch batch_config_example.yaml --validate-only
```

Consulta `batch_config_example.yaml` y `tests/test_batch_full.yaml` para ejemplos de:
- Legacy con `weights`, `block_multipliers`, `crossattn_boosts` y `loras`.
- PerRes con `assignments` y `attn2_locks`.
- Hybrid con `hybrid_config` y `attn2_locks`.

También hay plantillas en `Utils/templates.py` y sección `templates` en el YAML de ejemplo.

Atajos:
- `scripts/run_batch.sh <config.yaml>`
- `scripts/run_batch_validate.sh <config.yaml>`

## Modo análisis (V1.3)

Herramientas para entender diferencias, compatibilidad y predecir características de fusión.

Ejemplos:

```
# Comparación entre dos modelos (por índice mostrado)
python XLFusion.py --analyze --compare 0 1

# Recomendaciones para objetivo concreto
python XLFusion.py --analyze --recommend balanced

# Exportar informe a JSON
python XLFusion.py --analyze --compare 0 1 --export-analysis report.json
```

Métricas clave: similitud coseno por bloque, cambios relativos, avisos de arquitectura, puntuación de compatibilidad y recomendaciones.

## Configuración (`config.yaml`)

Se centraliza el nombre de salida, versionado, rutas y defaults.

- `model_output`:
  - `base_name`: prefijo del archivo resultante (p.ej. `XLFusion_V1.safetensors`).
  - `version_prefix`: `V`, `v`, `Ver`, etc.
  - `file_extension`: siempre `.safetensors`.
  - `output_dir`, `metadata_dir`, `auto_version`.

- `directories`:
  - `models`, `loras`, `output`, `metadata`.

- `merge_defaults`:
  - `legacy`: multiplicadores y `cross_attention_boost` por defecto.
  - `perres`: `cross_attention_locks` por defecto.
  - `hybrid`: normalización automática, pesos mínimos, locks por defecto.

- `app`:
  - `tool_name`, `version` (incluido en metadatos).

## Salida y metadatos

- Modelos: `XLFusion_V{n}.safetensors` en `output/`.
- Metadatos: carpeta `metadata/meta_{n}/` con:
  - `metadata.txt`: resumen humano-legible, hashes BLAKE2 de entradas (modelos y LoRAs) y kwargs exactos.
  - `batch_config.yaml`: configuración reproducible del trabajo para rehacer el resultado.

Los metadatos también se incrustan en el propio `.safetensors`.

## Buenas prácticas y rendimiento

- Revisa el aviso de memoria estimada antes de grandes fusiones.
- En Legacy, normaliza pesos; con `block_multipliers` y `crossattn_boosts` puedes afinar comportamiento.
- Con PerRes/Hybrid, usa `attn2_locks` para consistencia del texto.
- El modo streaming evita cargar todo en GPU/CPU simultáneamente.

## Pruebas y smoke test

Este repo incluye pruebas unitarias en `tests/` y un smoke test automatizado que genera modelos sintéticos y limpia residuos:

```
scripts/smoke_test.sh
```

El script crea modelos de prueba, ejecuta un batch con 4 jobs y elimina los artefactos temporales al finalizar.

## Roadmap

Consulta `ROADMAP.md` para objetivos V2.1+ (validación reforzada, preflight de memoria/compatibilidad, mejoras de rendimiento y extensibilidad).

## Créditos y contacto

- Portfolio: https://warcos.dev/
- LinkedIn: https://www.linkedin.com/in/marcosgarest/
