# XLFusion - Hoja de Ruta

> **Work in Progress**
> Esta hoja de ruta está sujeta a cambios según disponibilidad de tiempo libre y feedback de la comunidad.

## Versiones Actuales

### V1.0 (Actual)
- Modo Legacy: Fusión ponderada con control por bloques
- Modo PerRes: Asignación por resolución con control preciso
- Sistema de configuración YAML centralizado
- Soporte para LoRAs (modo Legacy)
- Versionado automático
- Metadatos y logs de auditoría

## Roadmap Futuro

### V1.1 - Modo Híbrido
**Objetivo**: Fusionar las capacidades de Legacy y PerRes en un solo modo unificado

**Características planeadas:**
- **Modo Híbrido**: Combinar ponderación (Legacy) con asignación por resolución (PerRes)
- **Control granular**: Aplicar diferentes pesos a diferentes bloques de resolución
- **Compatibilidad total**: Mantener soporte para configuraciones Legacy y PerRes existentes
- **UI mejorada**: Interfaz más intuitiva para configurar el modo híbrido

**Beneficios:**
- Máxima flexibilidad en la fusión de modelos
- Mejor control sobre aspectos específicos (composición, detalles, estilo)
- Aprovecha lo mejor de ambos modos actuales

---

### V1.2 - Fusión por Lotes (Batch)
**Objetivo**: Permitir fusiones automatizadas mediante configuración JSON/YAML

**Características planeadas:**
- **Configuración JSON/YAML**: Definir múltiples fusiones en un archivo
- **Procesamiento por lotes**: Ejecutar fusiones sin interacción CLI
- **Templates predefinidos**: Configuraciones comunes para diferentes tipos de fusión
- **Validación de configuración**: Verificar archivos de configuración antes de procesar
- **Progreso detallado**: Barra de progreso y logs para fusiones múltiples

**Formato de configuración ejemplo:**
```yaml
batch_jobs:
  - name: "Artistic_Mix"
    mode: "hybrid"
    models: ["model_a.safetensors", "model_b.safetensors"]
    weights: [0.7, 0.3]
    resolution_assignments:
      down_0_1: "model_a"
      down_2_3: "model_b"

  - name: "Style_Transfer"
    mode: "perres"
    models: ["base.safetensors", "style.safetensors"]
    # ... más configuración
```

**Beneficios:**
- Automatización de flujos de trabajo repetitivos
- Reproducibilidad exacta de fusiones
- Procesamiento masivo sin supervisión
- Integración con pipelines de ML

---

### V1.3 - Análisis Avanzado
**Objetivo**: Herramientas de análisis y comparación de modelos

**Características planeadas:**
- **Análisis de diferencias**: Comparar modelos y detectar cambios significativos
- **Predicción de resultados**: Estimación de características del modelo fusionado
- **Visualización de estructura**: Mapas de calor de diferencias entre modelos
- **Métricas de compatibilidad**: Puntuación de qué tan bien se fusionarán dos modelos
- **Recomendaciones inteligentes**: Sugerencias automáticas de configuración

---

### V2.0 - Interfaz Gráfica (.exe)
**Objetivo**: Aplicación ejecutable para usuarios no técnicos (Windows)

**Características planeadas:**
- **Ejecutable independiente**: Aplicación .exe sin dependencias externas
- **GUI nativa**: Interfaz gráfica intuitiva para Windows
- **Vista previa visual**: Comparación visual de configuraciones
- **Asistente guiado**: Wizard paso a paso para nuevos usuarios
- **Gestión de modelos**: Biblioteca de modelos con metadatos
- **Instalación simple**: Un solo archivo ejecutable, sin setup complejo

**Nota**: El ejecutable será exclusivo para Windows. Usuarios de Linux/macOS continuarán usando el script Python directamente.

---

### V2.1 - Funciones Experimentales
**Objetivo**: Técnicas avanzadas de fusión

**Características planeadas:**
- **Fusión diferencial**: Técnicas basadas en diferencias entre modelos
- **Fusión por capas semánticas**: Control basado en el significado de las capas
- **Optimización automática**: Búsqueda automática de mejores configuraciones
- **Fusión condicional**: Diferentes configuraciones según el prompt

---

## Prioridades de Desarrollo

### Alta Prioridad
1. **V1.1 - Modo Híbrido**: Máximo impacto, complejidad media
2. **V1.2 - Fusión por Lotes**: Alta demanda de automatización

### Media Prioridad
3. **V1.3 - Análisis Avanzado**: Funcionalidad diferenciadora
4. **V2.0 - Interfaz Gráfica**: Mejora de usabilidad

### Baja Prioridad
5. **V2.1 - Experimental**: Investigación y desarrollo a largo plazo

## Contribuciones de la Comunidad

### Cómo Contribuir
- **Issues**: Reportar bugs y solicitar características
- **Pull Requests**: Implementaciones de funcionalidades
- **Testing**: Pruebas con diferentes modelos y configuraciones
- **Documentación**: Mejoras en guías y tutoriales


---

## Notas Finales

Esta hoja de ruta refleja la visión a largo plazo de XLFusion, pero está sujeta a:

- **Disponibilidad de tiempo**: Desarrollo realizado en tiempo libre
- **Feedback de usuarios**: Prioridades basadas en necesidades reales
- **Evolución tecnológica**: Adaptación a nuevas técnicas y formatos
- **Recursos de desarrollo**: Limitaciones de tiempo y capacidad

**Tienes sugerencias o quieres contribuir?**
Abre un issue en el repositorio o contáctame a través de [LinkedIn](https://www.linkedin.com/in/marcosgarest/).

---

*Última actualización: Septiembre 2025*