# Validación sintética de Stage 0

**Fecha:** 2026-07-30 · **Script:** `stage0_validation.py` · **Log:** `run_log_validation.txt`

Objetivo: pasar de "un componente y un caso de estudio" a "una herramienta con
evidencia". El caso del windage muestra que el cribado detecta un problema real en un
corpus real, pero no que **generalice** ni que **mejore a la heurística obvia**
(marcar los pares con correlación log-log cercana a 1). Esto lo mide sobre datos con
verdad conocida.

Modelo generador: `log C_p = c + θ_Re·log Re + θ_M·log M + θ_g·log N_g + ε`, con los tres
grupos monomios en los mandos (Ω, R, p, T).

---

## A. La atribución la decide el prior del algoritmo, no los datos

Es la demostración directa del punto (ii) de la Proposición 1, y la más contundente.

Diseño degenerado tipo Liu (5 velocidades × 4 presiones, R y T fijos), con verdad
θ_Re = −0,03, θ_M = −0,12, **θ_g = 0** (la gravedad no interviene). Búsqueda de
estructura por BIC sobre los 7 subconjuntos:

| estructura | SSE | BIC | coeficientes |
|---|---|---|---|
| Re + M | 0,0083821 | **−146,561** | −0,0372, −0,1365 |
| Re + N_g | 0,0083821 | **−146,561** | −0,0372, **−0,0683** |
| Re + M + N_g | 0,0083821 | −143,565 | −0,0372, −0,0273, −0,0546 |
| … | | | |

**SSE y BIC idénticos hasta el último dígito.** La ley falsa "C_p depende de la gravedad
con exponente −0,0683" ajusta *exactamente* igual de bien que la verdadera — y −0,0683 es
justo θ_M/2, porque log N_g = 2·log M + cte. Ningún criterio de bondad de ajuste puede
separarlas: la elección la hace el desempate del buscador.

Añadiendo **dos temperaturas de gas** (rango 3, κ = 12,2), el empate desaparece: Re + M
gana en solitario (BIC −317,07 frente a −310,64 de Re + N_g) y recupera θ_M = −0,123
frente al −0,120 verdadero.

Esto es exactamente lo que pasa con Liu2024 en el corpus real, y aquí se ve con la
verdad delante.

## B. El condicionamiento predice el error de recuperación

400 diseños aleatorios de rango completo, σ = 0,02, verdad |θ_M| = 0,120:

| κ | n | κ mediana | error mediano en θ_M | percentil 90 |
|---|---|---|---|---|
| < 3 | 126 | 1,76 | **0,0107** | 0,0252 |
| 3 – 10 | 147 | 6,14 | 0,0291 | 0,0682 |
| 10 – 32 | 56 | 15,8 | 0,0631 | 0,183 |
| 32 – 100 | 33 | 52,9 | 0,185 | 0,597 |
| > 100 | 38 | 214 | **0,969** | 3,24 |

**Spearman(κ, |error|) = +0,706, p = 1,2e−61.** Error mediano con κ < 30: 0,020 (17% del
efecto). Con κ > 30: 0,336 (2,8 **veces** el efecto). El condicionamiento no es una
etiqueta cualitativa: es un predictor calibrado del error.

## C. Con dos grupos, la heurística de correlación NO es peor — y hay una razón

Resultado honesto y negativo para la versión fuerte de la afirmación.

3000 ensayos, 2 grupos (Re, M):

| pantalla | precisión | sensibilidad | acierto |
|---|---|---|---|
| Stage 0 (rango + κ ≤ 30) | 0,841 | 0,937 | 0,839 |
| heurística (\|corr\| < 0,99) | **0,883** | 0,867 | 0,835 |

Casos en que Stage 0 dice sí y la heurística no: 266. **Al revés: cero.**

La razón es matemática, no empírica: **para dos columnas centradas y normalizadas,
κ = √((1+|r|)/(1−|r|))**. Comprobado numéricamente (diferencia máxima 7,1e−07). Con dos
grupos, κ y la correlación **son el mismo estadístico**, y `|corr| < 0,99` equivale
exactamente a `κ < 14,1`. La diferencia observada es de umbral, no de criterio.

Barrido del umbral (2 grupos): κ ≤ 20 da el mejor acierto (0,842), κ ≤ 30 da 0,839 con
mayor sensibilidad. El umbral clásico de 30 es defendible, no óptimo.

## D. Con tres grupos, la heurística por pares se queda ciega

Aquí está la diferencia real, y es grande. Con m ≥ 3, un grupo puede ser casi
combinación lineal de los otros dos **sin que ninguna correlación por pares sea alta**.

4000 ensayos con los tres grupos:

| pantalla | precisión | sensibilidad | acierto |
|---|---|---|---|
| **Stage 0 (rango + κ ≤ 30)** | **0,720** | 0,534 | **0,845** |
| heurística por pares (\|corr\| < 0,99) | **0,376** | 0,590 | 0,679 |

La tasa de falsa promesa —decir "adelante" y que el exponente no se recupere— es del
**28% con Stage 0 y del 62% con la heurística**.

**Zona ciega de la heurística: 23,2% de los ensayos** tienen todas las correlaciones por
pares por debajo de 0,99 (mediana 0,969) y sin embargo κ conjunto > 30 (mediana **306**,
máximo 2,3e+04). De esos, solo el **15,8%** son recuperables — y la heurística los da
**todos** por buenos.

---

## Qué se puede afirmar con esto

1. **La parte de rango no es una heurística en absoluto**: es exacta, y además dice
   *qué* combinación sí es estimable. El experimento A lo demuestra con la verdad
   conocida: dos leyes distintas, ajuste idéntico, y la elección la hace el buscador.
2. **La parte de condicionamiento está calibrada**: κ predice el error de recuperación
   con ρ = 0,71 sobre dos órdenes de magnitud.
3. **Frente a la heurística de correlación**: con dos grupos son el mismo estadístico y
   hay que decirlo (§C) — no reclamar mejora donde no la hay. Con tres o más, la
   heurística por pares es ciega a la degeneración conjunta en el 23% de los diseños, y
   su tasa de falsa promesa dobla la de Stage 0.

Ese tercer punto es la justificación cuantitativa de sustituir el "pre-screening de
degeneración" por correlación que se había propuesto en la primera ronda. Y el segundo
es lo que permite reportar κ como magnitud, no como etiqueta.

## Limitaciones de esta validación

- El generador es log-lineal en los grupos: no prueba nada sobre estructuras no
  monomiales. El criterio tampoco pretende cubrirlas — es una afirmación sobre el
  espacio de exponentes.
- El umbral κ = 30 se importa de Belsley–Kuh–Welsch; el barrido de §C muestra que en
  este generador el óptimo está más cerca de 20. Conviene reportar el barrido y no
  presentar 30 como un valor mágico.
- "Recuperable" se define como error menor que la mitad del efecto verdadero. Es una
  elección; con otro criterio los números absolutos cambian, no el orden entre pantallas.
