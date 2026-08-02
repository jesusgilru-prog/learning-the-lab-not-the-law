# Prior art del criterio de identificabilidad por diseño

**Fecha:** 2026-07-30. Búsqueda hecha tras la ronda 3 del equipo, que coincidió en que
esto es el punto que decide si un revisor lo lee como aportación o como reetiquetado de
resultados clásicos.

**Conclusión: cada pieza existe por separado y hay que citarla explícitamente. La
composición —evaluar el rango del diseño realizado en el espacio Π, por fuente, como
paso previo a auditar confusión de dominio en regresión simbólica— no la he encontrado
publicada.** El álgebra no se reclama; se reclama dónde se inserta.

---

## 1. Referencias obligatorias, verificadas

| Pieza | Referencia | Qué es y por qué obliga |
|---|---|---|
| Diseño de experimentos en espacio Π | **Albrecht, Nachtsheim, Albrecht & Cook (2013)**, *Experimental Design for Engineering Dimensional Analysis*, Technometrics 55(3):257–270, DOI 10.1080/00401706.2012.746207 | El pariente más cercano en dirección **directa**: diseñar experimentos nuevos sobre grupos adimensionales. Nosotros vamos al revés: auditar diseños ajenos ya ejecutados. |
| Correlación espuria en diseños Π | **Comment: Spurious Correlation and Other Observations on Experimental Design for Engineering Dimensional Analysis**, Technometrics 55(3), DOI 10.1080/00401706.2013.778792 | Discute correlaciones espurias entre grupos Π inducidas por el diseño. **Es lo más cercano que hay al problema exacto** y no citarlo sería un agujero. |
| Detección de variables latentes por análisis dimensional | **del Rosario, Lee & Iaccarino (2019)**, *Lurking Variable Detection via Dimensional Analysis*, SIAM/ASA JUQ, DOI 10.1137/17M1155508 (preprint arXiv:1711.03918) | Test de hipótesis para variables omitidas usando una forma modificada del teorema Π. Detectan **desde los datos**; nosotros certificamos **desde el diseño**, antes de mirar la respuesta. |
| Estimabilidad en modelos lineales | Scheffé, *The Analysis of Variance*; Searle, *Linear Models* | Un funcional c'β es estimable ⟺ c está en el espacio fila del diseño. **El criterio de rango ES esto**, aplicado al log-diseño con features monomiales. No reclamar el resultado. |
| Estructura de alias en factoriales | Box, Hunter & Hunter, *Statistics for Experimenters* | Dos efectos aliasados por el diseño son indistinguibles. Misma idea, vocabulario clásico. |
| Identificabilidad estructural vs práctica | Bellman & Åström (1970); Walter & Pronzato (1997); Raue et al. (2009), *Bioinformatics* (profile likelihood) | De aquí viene el vocabulario **estructural / práctica**. Aviso de ChatGPT, correcto: en sistemas dinámicos "structural identifiability" significa otra cosa (identificabilidad de parámetros de un modelo postulado). **No invadir el término**: hay que decir "design identifiability" o equivalente y citar estos. |
| Diagnóstico de colinealidad | Belsley, Kuh & Welsch (1980) | Umbral κ > 30. **Solo válido con columnas escaladas a norma unidad** — corregido en el análisis a petición de Kimi. |

## 2. Adyacentes que conviene mirar antes de escribir

- **Bayesian Experimental Design for Symbolic Discovery** (arXiv:2211.15860). Diseño
  óptimo bayesiano para descubrimiento simbólico. Dirección directa otra vez: diseñar
  experimentos nuevos. Útil para posicionar el contraste.
- **Attractor Geometry Determines the Identifiability Limits of System Discovery**
  (arXiv:2607.18490, 2026). Muy reciente y en el mismo espíritu —"cuándo funciona
  realmente el descubrimiento automático de ecuaciones"— pero para sistemas dinámicos y
  geometría del atractor, no para diseño experimental en espacio Π. Hay que leerlo:
  es el trabajo que más se acerca al *mensaje*, aunque no al *mecanismo*.
- **Dimensionally consistent learning with Buckingham Pi**, Nature Computational
  Science (2022). Imponen consistencia dimensional al aprendizaje; **no** auditan el
  aliasing inducido por el diseño de cada fuente. Ese es el hueco a declarar.
- Batch effects / site confounding (Leek et al. 2010) y shortcut learning
  (Geirhos et al. 2020): ya conectados con la tesis del paper.

## 3. Cómo posicionarlo (consenso de las tres rondas)

La primera frase de la subsección debe decir de qué resultado clásico es instancia.
Literalmente: el criterio **es** estimabilidad en modelos lineales evaluada sobre el
log-diseño con base monomial, y su pariente en DOE **es** la estructura de alias. Lo que
se reclama es:

1. evaluarlo en el espacio de grupos adimensionales **desde metadatos de instalación**,
2. **por fuente**, no sobre el corpus agregado,
3. como **paso previo** de un protocolo de auditoría de confusión en descubrimiento
   simbólico, y
4. que su salida sea un **veredicto por fuente** sobre qué afirmaciones puede sostener
   un corpus heredado, más la receta experimental que rompe la degeneración.

Y una frase explícita de alcance, para que nadie crea que se reinventa DOE: el cribado
determina **qué puede distinguir el diseño**, no qué modelo es físicamente correcto.

## 4. Pendiente

- [ ] Leer el *Comment* de Technometrics 2013 sobre correlación espuria: es el que más
      cerca está de invalidar la novedad. Si ya contiene el criterio de rango por
      fuente, hay que rebajar la reclamación a "aplicación".
- [ ] Leer arXiv:2607.18490 y decidir si va en Related Work.
- [ ] Comprobar si Raue et al. usan "practical identifiability" con la misma definición
      operativa (perfil de verosimilitud) o si aquí se usa en sentido de
      condicionamiento; si difieren, decirlo.

## Fuentes

- [Experimental Design for Engineering Dimensional Analysis (Technometrics 2013)](https://www.tandfonline.com/doi/abs/10.1080/00401706.2012.746207)
- [Comment: Spurious Correlation and Other Observations on Experimental Design for Engineering Dimensional Analysis](https://www.tandfonline.com/doi/abs/10.1080/00401706.2013.778792)
- [Lurking Variable Detection via Dimensional Analysis (SIAM/ASA JUQ)](https://epubs.siam.org/doi/10.1137/17M1155508)
- [Lurking Variable Detection via Dimensional Analysis (preprint)](https://arxiv.org/abs/1711.03918)
- [Bayesian Experimental Design for Symbolic Discovery](https://arxiv.org/pdf/2211.15860)
- [Attractor Geometry Determines the Identifiability Limits of System Discovery](https://arxiv.org/html/2607.18490v1)
- [Dimensionally consistent learning with Buckingham Pi (Nature Comput Sci)](https://www.nature.com/articles/s43588-022-00355-5)
- [Parameter identifiability analysis and visualization in large-scale kinetic models](https://bmcsystbiol.biomedcentral.com/articles/10.1186/s12918-017-0428-y)
