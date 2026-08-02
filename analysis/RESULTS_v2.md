# Auditoría de identificabilidad por diseño — resultados v2

**Fecha:** 2026-07-30 · **Sustituye a** `RESULTS_identifiability.md` (v1), cuyos números
de Liu2024 quedan obsoletos por la re-digitalización.

**Reproducible:**

```
python redigitize_liu_fig7a.py      # Fig. 7(a) a nivel de píxel  -> liu_fig7a_redigitized.csv
python build_corrected_dataset.py   # dataset corregido           -> cross_rotor_dataset_v4.csv
python design_identifiability_v2.py # análisis completo           -> run_log_v2.txt
```

Todos los números de este documento salen de `run_log_v2.txt`.

---

## 0. Qué ha cambiado respecto a v1

Tres correcciones de datos y una del criterio:

1. **Fig. 7(a) de Liu re-digitalizada a nivel de píxel.** Los valores anteriores se
   habían leído a ojo sobre una rejilla de 0,5 kW (±17% en el punto más pequeño).
   Ahora: calibración por los ticks del eje a **17,1339 px/kW**, residuos del ajuste
   lineal ≤ 0,20 px, **resolución 0,0584 kW**. Los valores viejos estaban mal de media
   un **8,1%**, con un máximo de **+22,7%** (101 kPa/600 g: 19,0 → 23,31 kW).
2. **Densidad del aire consistente con la temperatura tabulada** (288 K → 293,15 K,
   −1,88% en ρ). No afecta a ningún exponente interno, sí a los Re y M absolutos.
3. **La columna de 10 kPa sigue marcada** como `computed_from_fitted_constant`: son
   cinco valores que no están en la figura y se retro-calcularon del propio ajuste del
   paper. No aportan información independiente.
4. **El criterio de identificabilidad ahora se evalúa sobre el diseño realizado**,
   `rank(D_c Eᵀ)`, y en coordenadas de los mandos que el experimentador fija
   (Ω, R, p, T). Restringir E a los mandos variados es solo una **cota superior**:
   falla si dos mandos se mueven en tándem. Y ρ, μ, a no son mandos: son funciones de
   p y T (μ ∝ T^0,7, a ∝ T^0,5).

### Validación de la re-digitalización

| Comprobación | Resultado |
|---|---|
| Linealidad de los 7 ticks del eje | residuo máximo 0,20 px |
| Los marcadores rojos reproducen P = C ρ ω³ | **R² = 0,998717** |
| Reconstrucción por borde izquierdo vs centroide (13 marcadores limpios) | error máximo **0,45 px = 0,027 kW** |

La tercera comprobación importa porque en las tres filas de menor potencia el marcador
gris queda parcialmente tapado por el rojo; el borde izquierdo nunca lo está.

---

## 1. El hallazgo empírico: encoge a la cuarta parte, pero se consolida

Si C_p = f(Re) con Re ∝ ρΩ, en `log C_p = c + α·log Ω + β·log ρ` la similaridad de
Reynolds **exige α = β**. Con R y T fijos dentro de Liu2024, α − β es exactamente el
exponente del segundo grupo.

| Datos | n | α − β | R² | p |
|---|---|---|---|---|
| v3, lectura a ojo | 20 | −0,1438 ± 0,0585 | 0,447 | 0,025 |
| v3, lectura a ojo, digitalizados | 15 | −0,1920 ± 0,0772 | 0,544 | 0,029 |
| **v4, píxel** | 20 | **−0,0514 ± 0,0171** | 0,557 | **0,0079** |
| **v4, píxel, digitalizados** | 15 | **−0,0733 ± 0,0203** | 0,690 | **0,0036** |

El efecto es **cuatro veces menor** de lo que parecía y **casi diez veces mejor
determinado**. Esto es lo que debe pasar cuando se sustituye una lectura burda por una
medida: el sesgo de cuantización inflaba la magnitud y la varianza a la vez.

Robustez (15 puntos re-digitalizados):

| Prueba | Resultado |
|---|---|
| Monte Carlo de digitalización a ±0,06 kW (resolución real) | P(signo negativo) = 1,0000 · P(p<0,05) = **0,98** |
| Monte Carlo a ±0,15 kW | P(p<0,05) = 0,61 |
| Permutación (20 000) | p = 0,0250 |
| Bootstrap estratificado por presión | IC95 [−0,1297, −0,0353] · P(≥0) = 0,0001 |
| Bootstrap estratificado por velocidad | IC95 [−0,0974, −0,0493] · P(≥0) = 0,0001 |
| Potencia para el efecto observado, n=15 | 0,910 |
| f(Re) con término cuadrático | coef. residual de log Ω = −0,0774 ± 0,0206, p = 0,0032 |
| f(Re) con término cúbico | −0,0770 ± 0,0231, p = 0,0076 |

La última fila responde a la objeción de que imponer log-log podría fabricar α ≠ β: el
coeficiente de log Ω **a Re controlado** no supone que f sea una ley de potencia y
sobrevive.

### Falsación sin ningún modelo

En el factorial hay condiciones que alcanzan el mismo Re por caminos distintos. Si
C_p = f(Re), deben dar el mismo C_p sea cual sea f:

| Condición A | Condición B | ΔRe | ΔC_p |
|---|---|---|---|
| 101 kPa, 44,3 rad/s | 50 kPa, 88,5 rad/s | 1,0% | **−2,0%** |
| 50 kPa, 44,3 rad/s | 30 kPa, 76,7 rad/s | 3,8% | −1,6% |
| 50 kPa, 62,6 rad/s | 30 kPa, 99,0 rad/s | 5,4% | **−7,0%** |

En los tres, la condición de **mayor Ω tiene menor C_p al mismo Re**. Con el error de
lectura de 0,058 kW, el tercer par (potencias de 6,58 y 14,51 kW, error relativo <1%)
es concluyente; el primero está solo marginalmente por encima del error. Es evidencia
real, pero **modesta**: no sustituye a la regresión, la acompaña.

---

## 2. La limitación que decide: la corrección de fricción de rodamientos

Liu obtiene el windage como *potencia de motor × 0,95 − M_f·Ω*, con
**M_f = I·β = 845,69 × 0,013 = 10,99 N·m constante**, extrapolado de ensayos de
spin-down a ω < 1 rad/s y 3–10 kPa (p. 19 del original). Esa fricción pesa entre el
**2,3% y el 34,6%** del windage publicado, y vive **exactamente en la dirección del
segundo grupo**: es función pura de Ω.

| Hipótesis sobre el par real | α − β | p |
|---|---|---|
| M_f × 0 (sin corregir nada) | −0,1795 ± 0,0277 | 3,0e−05 |
| M_f × 0,50 | −0,1304 ± 0,0200 | 2,9e−05 |
| **M_f × 1,00 (lo publicado)** | **−0,0733 ± 0,0203** | **0,0036** |
| M_f × 1,25 | −0,0410 ± 0,0263 | 0,146 — pierde significación |
| M_f × 1,50 | −0,0055 ± 0,0357 | 0,88 — muerto |
| exceso **viscoso** ×1,5 a Ω máxima | −0,0700 ± 0,0234 | 0,011 — sobrevive |
| exceso **viscoso** ×2,0 a Ω máxima | −0,0658 ± 0,0284 | 0,039 — sobrevive |

**El hallazgo depende de que el par de fricción real no sea más de un 25% mayor que el
descontado.** Es un margen estrecho y hay que escribirlo así.

### 2.1 La Fig. 6 re-extraída acota M_f y excluye el escenario letal

La Fig. 6 del original (curvas de spin-down ω(t) a 3 y 10 kPa) está en el PDF como
**gráfico vectorial**, no como imagen: se extraen las coordenadas exactas de los
trazados, sin error de digitalización (calibración con residuo máximo 0,024 px en t y
0,021 px en ω). 79 puntos a 3 kPa + 78 a 10 kPa, ω ∈ [0,016, 1,05] rad/s. Ver
`extract_liu_fig6_spindown.py`, `liu_fig6_spindown.csv`, `run_log_fig6.txt`.

**Validación.** Ajuste con M_f constante: **M0 = 10,973 ± 0,035 N·m**, que reproduce el
I·β = 10,99 de Liu al 0,15%. El windage durante el propio spin-down es ≤ 0,007 N·m
(0,07% de M_f): las dos curvas son dos réplicas de fricción pura y coinciden entre sí.
Verificado de forma independiente: pendiente lineal −0,01294 rad/s² a 3 kPa y
−0,01302 a 10 kPa → 10,94 y 11,01 N·m.

**M_f baja con ω en el rango medido.** El término cuadrático en t es masivamente
significativo (F = 85 a 3 kPa, F = 104 a 10 kPa, p ~ 1e−14). Ajuste ODE:
M_f = (12,36 ± 0,11) − (2,46 ± 0,18)·ω N·m. Por ventanas locales:

| ω | M_f ponderado |
|---|---|
| [0,0 – 0,3) | 12,83 ± 0,18 N·m |
| [0,3 – 0,6) | 10,93 ± 0,09 N·m |
| [0,6 – 1,05] | **10,57 ± 0,08 N·m** |

La tendencia descendente **no** es solo la cola de régimen límite: excluyendo ω < 0,3
sigue con p = 1,5e−05, y excluyendo ω < 0,4 con p = 7,6e−04. Es la rama descendente de
Stribeck (régimen límite/mixto).

**Consecuencia.** El escenario que mata el hallazgo —M_f real constante y un 25% mayor,
13,7 N·m en 44–99 rad/s— exige que la componente *independiente de ω* valga eso. Pero
esa componente es justo lo que el spin-down mide: 10,97 de media y **cayendo**, ya en
10,57 en el tramo superior medido, por debajo del 10,99 que Liu descuenta. La
sobre-sustracción que implica Stribeck **refuerza** el efecto (M_f × 0,95 → p = 0,0016;
× 0,90 → p = 0,0007). **El escenario constante × 1,25 queda excluido por los propios
datos de Liu.**

**Lo que queda vivo, cuantificado.** Un par que *crece* con ω —rodadura EHL ∝ ω^0,6–0,8,
sellos ∝ ω— no es observable por debajo de 1 rad/s (allí aportaría < 0,4 N·m) y sí es un
confusor real. Con ΔM = (c−1)·M_f0·(ω/ω_max)^q:

| q | c = 1,25 | c = 1,50 | c = 2,00 |
|---|---|---|---|
| 0,0 (constante) | p = 0,146 | p = 0,88 | p = 0,25 |
| 0,6 | p = 0,018 | p = 0,077 | p = 0,51 |
| 0,8 | p = 0,010 | p = 0,029 | p = 0,163 |
| 1,0 | p = 0,006 | p = 0,011 | **p = 0,039** |

El signo negativo se mantiene en todos los casos hasta c = 2. La única escapatoria es un
par de rodamiento creciente con ω que a velocidad máxima sea ≥1,5–2× el descontado, no
observable desde la Fig. 6 y no descartable sin las especificaciones del rodamiento.

**Estatus resultante:** la magnitud −0,073 es una **cota superior** del efecto físico;
lo robusto es el signo. El spin-down se midió a ω < 1,05 rad/s y la operación llega a
99 rad/s — dos órdenes de magnitud — así que la extrapolación de la rama de Stribeck no
es legítima y hay que decirlo.

---

## 3. El criterio: rango del diseño realizado

Exponentes en coordenadas de mandos (Ω, R, p, T):

```
g_level  = Ω²R/g_e          → ( 2, 1, 0,  0   )
Re_Omega = pΩR²/(R_s T μ(T)) → ( 1, 2, 1, −1,7)
M_tip    = ΩR/a(T)           → ( 1, 1, 0, −0,5)
```

`rank(D_c Eᵀ)` sobre el diseño realizado:

| Fuente | mandos variados | rango a priori (cota) | rango realizado | κ (sin normalizar) | **κ (BKW, columnas a norma 1)** | degenerados exactos |
|---|---|---|---|---|---|---|
| Liu2024 | Ω, p | 2 | **2** | 1,61 | **1,62** | (g, M) |
| Vrancik1968 | Ω, R, p, T | 3 | **3** | 9,70 | **6,71** | ninguno |
| Xia2024 | Ω, R | 2 | **2** | 154,4 | **136,9** | ninguno |
| Zheng2024 | Ω | 1 | **1** | 1 | **1** | los tres |

En este corpus la cota a priori y el rango realizado **coinciden**, pero el enunciado
del paper debe ser el realizado: es el exacto.

El κ que se reporta es el de columnas escaladas a norma euclídea unidad; sin esa
normalización los κ de distintas instalaciones no son comparables entre sí ni con el
umbral clásico de 30 (Belsley, Kuh & Welsch 1980).

**El cribado es ciego a la respuesta.** Comprobado: sustituyendo C_p por ruido
lognormal, el rango y el κ de las cuatro instalaciones salen **idénticos** (Liu 2 /
1,622; Vrancik 3 / 6,713; Xia 2 / 136,897; Zheng 1 / 1,000). Perturbando en cambio la
temperatura del diseño un ±10%, el rango de Liu sube a 3 y su κ a 20,3, el de Xia a 3
con κ = 3192, y el de Zheng a 2. Es decir: el criterio depende del diseño y solo del
diseño, que es justo lo que se afirma de él.

**Estructural ≠ práctico.** Xia2024 varía R un 7%, así que estructuralmente podría
separar g de Re; con κ = 136,9 (frente a 1,62 de Liu) no puede en la práctica — muy por
encima del umbral 30. El rango dice qué es **imposible por diseño**; el
condicionamiento, qué es **inútil por magnitud**. Una correlación sola no distingue los
dos casos: en Xia da 0,9998 y en Zheng 1,0000, y son situaciones cualitativamente
distintas.

> Nota: en v1 el condicionamiento de Xia salía 2,2e15 porque incluía la dirección de
> rango nulo. Sobre el rango efectivo y con columnas normalizadas es **136,9**.
> El 2,2e15 no debe citarse.

---

## 4. Qué se puede y qué no se puede atribuir

Dentro de Liu2024, R y T son fijos ⟹ M ∝ Ω y g ∝ Ω² son **colineales exactos** en
logaritmos (corr = 1,000000, degeneración de rango, no de muestra). El −0,073 se
atribuye igual de bien a Mach que a nivel-g, y **ningún dato del corpus los separa**:

- Liu2024: (g, M) degenerados por rango.
- Xia2024: corr(log g, log M) = 0,999983.
- Vrancik1968 y Zheng2024: g_level tabulado ≡ 1.

Además, la clase de equivalencia es mayor que {Mach, g}: el residuo es el exponente de
**cualquier** grupo ∝ Ω^k a R, T y gas fijos, lo que incluye sistemáticas
experimentales puras en Ω — deformación aeroelástica del brazo, calentamiento por
disipación y la propia fricción de §2. La afirmación honesta es que la auditoría
rechaza *C_p = f(Re, geometría)* como modelo completo de los datos tabulados; la
**atribución** del residuo requiere el experimento que falta.

**El experimento que falta, deducido del criterio:** una tercera dirección independiente
que deje g intacto, es decir variar la velocidad del sonido a Ω y R fijos — cambiar el
gas de la cámara o su temperatura — dentro de una centrifugadora de hipergravedad.
Vrancik1968 varió T pero está a g = 1; Liu2024 varió la presión, que mueve ρ y no a.
Nadie ha hecho la combinación. Ojo al enunciarlo: variar T o el gas **también mueve Re
y μ**; lo que aporta no es "mover Mach dejando el resto quieto", sino añadir una
dirección independiente al diseño.

### Chequeo de consistencia en el único diseño de rango 3

En Vrancik1968, con efectos fijos por geometría:

| Lectura del segundo grupo | exponente | p |
|---|---|---|
| como Mach | −0,2070 ± 0,1173 | 0,086 |
| como g monómico | −0,1002 ± 0,0578 | 0,092 |

(con Re = −0,32 ± 0,035, R² = 0,894). Mismo **signo** que Liu, magnitud unas 3× mayor,
en una instalación independiente y con geometría muy distinta (holgura 0,003–0,015
frente a 0,15). Se reporta como **chequeo de consistencia, no como réplica**: asume
exponente común entre las geometrías de Vrancik —dudoso, porque sus pendientes
individuales van de −0,23 a −0,83— y la identificación viene de comparar sub-diseños.

---

## 5. Semántica de "nivel g": el agujero que nadie había visto

| Fuente | g_level tabulado | Ω²R/g real |
|---|---|---|
| Liu2024 | 200 – 1000 | 200 – 999 |
| Xia2024 | 10 – 1540 | 10 – 1541 |
| **Vrancik1968** | **1** | **888 – 14 274** |
| **Zheng2024** | **1** | **266 – 19 703** |

Las dos fuentes etiquetadas "a 1 g" **muestrean el monomio más alto del corpus**.
`g_level` es la gravedad de la *carga útil* declarada por la instalación; el monomio
Ω²R/g_e —el que entra en el análisis dimensional— lo tiene todo rotor.

El paper tiene que declarar cuál de los dos objetos es la hipótesis física. Y esto
**refuerza** su tesis: la etiqueta "hypergravity facility" es exactamente el tipo de
metadato de dominio que se confunde con un mecanismo físico.

---

## 6. Contexto: la ley agrupada sigue siendo un artefacto

| Modelo | R² en muestra | LOGO rmse(log) medio |
|---|---|---|
| C_p ~ Re^a | 0,825 | 0,760 |
| C_p ~ Re^a M^b | 0,825 | 0,774 |
| + 4 grupos geométricos | 0,866 | 1,832 |
| + geometría y Mach | 0,871 | 2,187 |

Y los exponentes ajustados **dentro** de cada geometría no promedian al agrupado:

- combinado por precisión sobre las 12 geometrías: **−0,042**
- agrupado sobre los 114 puntos: **−0,615** (R² 0,825)
- **Q de Cochran = 178,7, gl = 11, p ≈ 0, I² = 93,8%**

Con esa heterogeneidad, el exponente agrupado no es un promedio con sentido físico de
los locales: es un artefacto de la disposición de las instalaciones en el plano
(log Re, log C_p).

**Aviso:** las 12 geometrías están anidadas en 4 instalaciones. Tratarlas como 12
clusters independientes **no** resuelve el problema de pocos clusters.

---

## 7. Auditoría del manuscrito

**Ningún número del manuscrito cambia por el reetiquetado.** `g_level` no es predictor
en ninguna parte del pipeline; el exponente primario (−0,146, IC IM [−0,362, +0,071]) es
el de Reynolds. Frases a corregir (solo texto): l. 79 (abstract), l. 190 (intro),
l. 902, l. 907–927 y `tab:dataset`, l. 984.

Lo que sí obliga a decidir es §5: qué es exactamente la hipótesis de "hipergravedad".

---

## 8. Estado de los puntos abiertos de v1

| Punto | Estado |
|---|---|
| ¿Las 4 presiones de cámara son condiciones experimentales reales? | **Sí.** Tabla 3 del original: 101/50/30/10 kPa × 5 aceleraciones (más 3 kPa a 1000 g), con unidad de vacío. |
| Origen de la columna de 10 kPa | Confirmado: no está en la Fig. 7, se retro-calculó del ajuste del paper. Marcada. |
| Bug de temperatura | Corregido (293,15 K). |
| ¿μ varía con la presión? | No para gas ideal; μ solo depende de T. Recogido en la parametrización por mandos. |
| Prior art | **Pendiente**: estimabilidad en modelos lineales (Scheffé, Searle), estructura de alias en DOE (Box–Hunter), identificabilidad estructural (Bellman–Åström, Walter–Pronzato, Raue), diseño en espacio Π (Albrecht 2013), detección de variables latentes por análisis dimensional (del Rosario 2019). Hay que posicionarse explícitamente. |
