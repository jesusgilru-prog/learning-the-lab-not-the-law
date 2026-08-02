# Identificabilidad por rango del diseño + test de similaridad de Reynolds

> **OBSOLETO (2026-07-30).** Los numeros de Liu2024 de este documento se calcularon
> antes de la re-digitalizacion a nivel de pixel de la Fig. 7(a). Usar `RESULTS_v2.md`.

**Fecha:** 2026-07-30
**Datos:** `/home/jesus/hyperscale-chief/data/processed/cross_rotor_dataset_v3.csv` (n=114, 4 fuentes, 12 geometrías)
**Script:** `ddp_rank_analysis.py` — **todos** los números de este documento salen de `run_log.txt`, regenerable con
`python ddp_rank_analysis.py`.

---

## Resumen

1. La proposición que íbamos a escribir ("ningún dataset de una sola instalación puede separar g de Re")
   **es falsa**: Liu2024 es un contraejemplo dentro del propio dataset del paper.
2. El criterio correcto no es una correlación, es un **rango**: el de la matriz de exponentes de los grupos
   adimensionales restringida a los controles que realmente se variaron. Exacto, computable desde el diseño,
   sin mirar la variable respuesta.
3. Ese criterio separa **identificabilidad estructural** (rango) de **identificabilidad práctica**
   (condicionamiento). Xia2024 es estructuralmente identificable y prácticamente degenerado — dos cosas
   distintas que la correlación 0.9998 confunde en una sola.
4. Aplicando el criterio, Liu2024 resulta ser el único diseño del corpus capaz de separar Re de un segundo
   grupo. Y al usarlo se **rechaza la similaridad de Reynolds** con F=1 instalación (p=0.025), sin ninguna
   comparación entre instalaciones y por tanto sin el problema de pocos clusters.
5. Ese mismo criterio dice que el efecto detectado **no puede atribuirse a Mach frente a nivel-g**: dentro de
   Liu2024 son *exactamente* degenerados. Ningún experimento del corpus puede separarlos, y sabemos
   exactamente cuál haría falta.

---

## 1. Contraejemplo a la proposición actual

`corr(log g, log Re)` **dentro** de cada instalación:

| Fuente | n | corr(log g, log Re) | ρ distintos | R distintos | Comentario |
|---|---|---|---|---|---|
| Xia2024 | 45 | **+0.999846** | 1 | 7 | degenerado en la práctica |
| Liu2024 | 20 | **+0.319422** | 4 | 1 | **no degenerado** |
| Vrancik1968 | 41 | — | 24 | 3 | g_level ≡ 1, no varía |
| Zheng2024 | 8 | — | 1 | 1 | g_level ≡ 1, no varía |

Liu2024 es un **factorial balanceado 5×4**: 5 velocidades de giro × 4 presiones de cámara
(10, 30, 50, 101 kPa). Variar la presión mueve ρ —y por tanto Re— **a Ω fijo**, que es justo lo que rompe la
degeneración. No hace falta comparar radios ni instalaciones: basta con tener una cámara de vacío.

La verificación de `corr = 0.9998` que teníamos era correcta, pero se hizo **solo dentro de Xia2024**, que es
precisamente la instalación que no varía la densidad. Generalizar de ahí a "ninguna instalación puede" era el
error. Es exactamente el resquicio que apuntaron Kimi (ν, T) y ChatGPT (gas, presión, temperatura) — solo que
no es hipotético: está en los datos del propio paper.

## 2. El criterio correcto: rango de la matriz de exponentes

Cada grupo adimensional es un monomio en los controles físicos (Ω, R, ρ, μ, a):

```
g_level  = Ω² R / g_tierra        → (2, 1, 0,  0,  0)
Re_Omega = ρ Ω R² / μ             → (1, 2, 1, -1,  0)
M_tip    = Ω R / a                → (1, 1, 0,  0, -1)
```

Verificado contra la tabla: error relativo máximo 4.4e-16 (Re), 1.3e-15 (M), 7.8e-04 (g en Liu), 3.4e-04
(g en Xia — redondeo de los g nominales publicados).

**Criterio.** Restringe la matriz a las columnas de los controles que la instalación *realmente varió*. Un
conjunto de grupos es estructuralmente identificable en esa instalación si y solo si sus filas restringidas son
linealmente independientes. Es exacto, se calcula antes de ver ninguna medida y no depende de la muestra.

| Fuente | controles variados | rango | pares exactamente degenerados |
|---|---|---|---|
| Liu2024 | Ω, ρ | **2** | **(g, M)** — corr = +1.000000 |
| Vrancik1968 | Ω, R, ρ, μ, a | **3** | ninguno |
| Xia2024 | Ω, R | **2** | ninguno (pero ver §3) |
| Zheng2024 | Ω | **1** | (Re, M) — corr = +1.000000 |

## 3. Estructural ≠ práctico

Xia2024 varía R (4.35 → 4.65 m, un 7%), así que **estructuralmente** podría separar g de Re. En la práctica no:

| Fuente | valores singulares de log-diseño (centrado) | cond |
|---|---|---|
| Vrancik1968 | 6.94, 1.69 | **4.1** |
| Liu2024 | 4.15, 2.58, 7.7e-05 | 5.4e+04 |
| Zheng2024 | 2.70, 3.6e-15 | 7.4e+14 |
| Xia2024 | 12.30, 0.0796, 5.7e-15 | **2.2e+15** |

En Xia la segunda dirección existe pero es 155× más débil que la primera, y la tercera es exactamente nula.
Es la distinción clásica entre identificabilidad estructural y práctica, y aquí aparece de forma limpia:
**el rango dice qué es imposible por diseño; el condicionamiento dice qué es inútil por magnitud.** Un
pre-screening basado solo en correlación mezcla los dos casos y no distingue "imposible" de "insuficiente".

## 4. Test de similaridad de Reynolds dentro de una sola instalación

Si de verdad C_p = f(Re) con Re ∝ ρΩ, en

```
log C_p = c + α·log Ω + β·log ρ
```

la similaridad de Reynolds **exige α = β**. Con R y T fijos en Liu2024, M ∝ Ω, así que **α − β es exactamente
el exponente del segundo grupo**.

| Muestra | n | α (Ω) | β (ρ) | α − β | p | veredicto |
|---|---|---|---|---|---|---|
| Los 20 puntos | 20 | −0.178 ± 0.055 | −0.035 ± 0.019 | **−0.144 ± 0.058** | 0.025 | rechaza |
| Solo digitalizados | 15 | −0.240 ± 0.067 | −0.048 ± 0.038 | **−0.192 ± 0.077** | 0.029 | rechaza |

Robustez (los 20 puntos):

- bootstrap 20 000 réplicas (semilla 20260730): IC95 = [−0.239, −0.028], P(α−β ≥ 0) = 0.0117
- jackknife: todas las 20 réplicas negativas, rango [−0.172, −0.114]
- leave-one-Ω-out (5): −0.087, −0.139, −0.156, −0.134, −0.189
- leave-one-presión-out (4): −0.192, −0.128, −0.101, −0.144

El mismo ajuste escrito en (Re, M):

```
log C_p = −2.200 + (−0.0345 ± 0.0187)·log Re + (−0.1438 ± 0.0585)·log M     R² = 0.447
          Re: p = 0.082  (no significativo)      M: p = 0.025  (significativo)
```

**La columna de 10 kPa (5 puntos) es `computed_from_fitted_constant`**, es decir C_p constante por
construcción (0.0880 ± 0.0002). Solo puede diluir el efecto: al quitarla el efecto **crece** de −0.144 a
−0.192. Los 15 puntos que lo sostienen son `digitized_from_figure` con error declarado del 5%.

Interpretación: en la única instalación del corpus con diseño de rango 2, el arrastre aerodinámico **no**
colapsa con el número de Reynolds. Hay un segundo grupo con exponente ≈ −0.15 a −0.19.

## 5. El límite: Mach y gravedad son indistinguibles, y sabemos por qué

Dentro de Liu2024, R y T son fijos, así que M ∝ Ω y g ∝ Ω²: en logaritmos son colineales exactos
(corr = +1.000000, comprobado). El exponente −0.144 se puede escribir igual de bien como efecto de Mach o
como efecto de nivel-g (con la mitad del exponente). **No hay dato en el corpus que los separe:**

- Liu2024: (g, M) exactamente degenerados por rango.
- Xia2024: (g, M) con corr 0.999983 — degenerados en la práctica.
- Vrancik1968 y Zheng2024: g ≡ 1, no hay variación de gravedad que atribuir.

El experimento que lo resolvería se deduce del criterio, no de la intuición: **variar la velocidad del sonido
a Ω y R fijos** — cambiar el gas de la cámara, o su temperatura — dentro de una centrifugadora de
hipergravedad. Eso mueve M dejando g intacto y hace la matriz de rango 3. Vrancik1968 varió T, pero está a
g = 1; Liu2024 varió la presión, que mueve ρ y no a. Nadie ha hecho la combinación que hace falta.

## 6. Contexto: por qué la ley agrupada sigue siendo un artefacto

| Modelo | R² en muestra | LOGO rmse(log) medio |
|---|---|---|
| C_p ~ Re^a | 0.828 | 0.737 |
| C_p ~ Re^a M^b | 0.828 | 0.752 |
| + 4 grupos geométricos | 0.870 | 1.759 |
| + geometría y Mach | **0.875** | **2.084** |

Añadir geometría sube el ajuste en muestra y **empeora** la extrapolación a una instalación nueva por un
factor 2.8. Es la tesis del paper, cuantificada: el ajuste agrupado compra R² pagando con identidad de
instalación.

Y los exponentes ajustados **dentro** de cada geometría (12) no se parecen al agrupado (−0.614):
media −0.196, mediana −0.119, sd 0.239, rango [−0.830, +0.070]. Vrancik (rotores confinados, holgura
0.003–0.015) da −0.43 de media; Xia (brazos en cámara abierta, holgura ≈ 0.11) da −0.077. La media
−0.196 cae casi encima del régimen canónico turbulento-rugoso q = −0.20 que usa el paper, pero con esa
dispersión no es una coincidencia sobre la que se pueda construir nada.

**Aviso:** las 12 geometrías están anidadas dentro de las 4 instalaciones. Usarlas como 12 clusters
independientes **no** resuelve el problema de pocos clusters — lo esconde. No es una salida al F=4.

---

## 7. Auditoría del manuscrito: qué toca el reetiquetado

Revisado `manuscript.tex` (líneas 79, 190, 902, 907–927, 984, 1032, 1118, 1158).

**Buena noticia: ningún número cambia.** `g_level` no es predictor en ninguna parte del pipeline. Los
candidatos son Re_Ω, M_tip y los 4 grupos geométricos; el exponente primario (−0.146, IC IM
[−0.362, +0.071]) es el de **Reynolds**, no el de gravedad. El error de "4 instalaciones de hipergravedad"
es de **descripción del corpus**, no de modelado.

Frases a corregir (solo texto):

- l. 79 (abstract) y l. 190 (intro): "4 hypergravity centrifuge facilities" → 2 centrifugadoras de
  hipergravedad (Liu2024, Xia2024) + 2 bancos rotativos a 1 g (Vrancik1968, Zheng2024).
- l. 902: "Our case study is windage power loss in hypergravity centrifuges" → el fenómeno es windage en
  maquinaria rotativa; las centrifugadoras de hipergravedad son dos de las cuatro fuentes.
- l. 907–916 y Tabla `tab:dataset`: añadir columna de nivel-g y de controles variados por fuente.
- l. 984: revisar "four independent hypergravity facilities".

Además, la sección de dataset ya dice (l. 925–927) que "Liu2024 occupies a disjoint Mach band from the other
sources --- hints at the confounding to come". Eso sigue siendo cierto **para el ajuste agrupado**, y ahora
además se puede decir que el diseño interno de Liu2024 es el único que permite estimar ese efecto Mach de
forma identificada. Las dos afirmaciones son compatibles y juntas son mucho más fuertes: el protocolo se
niega a certificar la ley de dos regímenes con datos agrupados, y el análisis por diseño dice qué sí se puede
concluir y de dónde.

---

## 8. Qué queda por comprobar antes de escribir nada en el .tex

- [ ] Verificar contra el paper original de Liu 2024 que las 4 presiones de cámara son condiciones
      experimentales reales y no un barrido calculado por los autores. Todo el hallazgo depende de eso.
- [ ] Confirmar el origen de la columna de 10 kPa (`computed_from_fitted_constant`) y decidir si entra.
- [ ] Prior art: identificabilidad estructural vs práctica en diseño de experimentos y en biología de
      sistemas. Hay literatura amplia; hay que posicionarse explícitamente, no reinventarla.
- [ ] Comprobar si μ debería variar con la presión en Liu2024 (para un gas ideal no, pero conviene decirlo).
