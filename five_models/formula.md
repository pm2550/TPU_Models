# Segment-chain execution model

We consider a single Edge TPU–compatible model $M$ that is partitioned into an ordered sequence of $n$ segments $\{S_1, S_2, \ldots, S_n\}$ and executed continuously as a chain. An end-to-end inference consists of invoking $S_1, S_2, \ldots, S_n$ in order. Let $d^{in}_i$ and $d^{out}_i$ denote the input and output tensors of segment $S_i$ respectively,

$$ d^{in}_{i+1} = d^{out}_i, \quad 1 \le i < n. $$

At the chain boundaries, $d^{in}_1$ is produced by the host from the original model input, and $d^{out}_n$ is returned to the host as the final model output.

We model the input and output tensor transmission of each segment $S_i$ explicitly. Define

- $C^{in}_i$: time to transfer $d^{in}_i$ from host to device (USB, h2d) before $S_i$ executes.
- $C^{out}_i$: time to transfer $d^{out}_i$ from device to host (USB, d2h) after $S_i$ completes.

With effective link bandwidths $B^{h2d}$ and $B^{d2h}$ at runtime,

$$ C^{in}_i = \frac{d^{in}_i}{B^{h2d}}, \qquad C^{out}_i = \frac{d^{out}_i}{B^{d2h}}. $$

Formally a segment is characterized as

$$ S_i := ( C^{in}_i, C^{out}_i, C^e_i, w^{tot}_i, C^{w}_i ), $$

where

- $C^e_i$: pure cumulative TPU computation if fully fed (no stalls).
- $w^{tot}_i$: total weight bytes required to complete $S_i$.
- $C^{w}_i$: effective weight transfer cost that cannot be hidden by overlap with compute.

The makespan for one pass through $S_i$ is

$$ W_i = C^{in}_i + C^{out}_i + C^e_i + C^w_i + \epsilon, $$

where $\epsilon$ is a small fixed control overhead.

To understand $C^w_i$, decompose total weight bytes as

$$ w^{tot}_i = w^{warm}_i + w^{rem}_i, $$

with

- $w^{warm}_i$: warm-up weight bytes required on device before compute can start.
- $w^{rem}_i$: remaining weight bytes that must be streamed during $S_i$.

Accordingly,

$$ C^w_i := t^{warm}_i + t^{rem}_i. $$

Warm-up time converts from bytes:

$$
 t^{warm}_i = 
 \begin{cases}
  \dfrac{w^{warm}_i}{B^{h2d}} & \text{if $w^{warm}_i$ not in SRAM}, \\
  0 & \text{if $w^{warm}_i$ in SRAM.}
 \end{cases}
$$

Once compute begins, TPU execution and USB streaming proceed in parallel. Treat $C^e_i$ as an overlap window; up to $B^{h2d}\, C^e_i$ bytes can be streamed without extending the makespan, and any excess becomes a residual tail:

$$
 t^{rem}_i = \max\!\left( \frac{w^{rem}_i - B^{h2d} C^e_i}{B^{h2d}},\, 0 \right)
 = \max\!\left( \frac{w^{tot}_i - w^{warm}_i}{B^{h2d}} - C^e_i,\, 0 \right).
$$

Bounds for $W_i$ in Eq. above:

- Lower bound (perfect streaming overlap):

$$ \widecheck{W_i} = C^{in}_i + C^{out}_i + C^e_i + t^{warm}_i + t^{rem}_i + \epsilon. $$

- Upper bound (no overlap):

$$ \widehat{W_i} = C^{in}_i + C^{out}_i + C^e_i + t^{warm}_i + \frac{w^{rem}_i}{B^{h2d}} + \epsilon. $$

Summing $W_i$ over $i=1,\dots,n$ yields the end-to-end latency for running $M$ as a segment chain.


## Host-side handling overhead (based on input span)

Empirically, there exists a host-side handling overhead that correlates strongly with the input I/O envelope span of each segment. Let

- $U^{in}_i$: the union/envelope span for input I/O of $S_i$ (ms), measured from usbmon span statistics.
- $T^{host}_M$: a model-specific fixed host handling baseline (ms). This is the per-model constant you referred to as `Th(host)`; we denote it $T^{host}_M$.
- $\kappa$: a global coefficient (ms/ms) capturing the linear dependence on $U^{in}_i$.

We model the per-segment host-side overhead as

$$
\Delta_i \;=\; T^{host}_M \; + \; \kappa\, U^{in}_i.
$$

Accordingly, the per-segment makespan with host handling becomes

$$
W'_i \;=\; W_i \; + \; \Delta_i \;=\; C^{in}_i + C^{out}_i + C^e_i + C^{w}_i + \epsilon \; + \; T^{host}_M \; + \; \kappa\, U^{in}_i.
$$

Summing over the chain, the added host contribution is $\sum_i \Delta_i = n\,T^{host}_M + \kappa\, \sum_i U^{in}_i$.

Notes:

- Calibration. From our current fit across five models and 8 segments (40 rows), a global linear fit suggests $\kappa \approx 0.299\;\mathrm{ms/ms}$ with an intercept around $0.553\;\mathrm{ms}$ when using a single global constant. For better accuracy, estimate $T^{host}_M$ per model using that model's data (keeping a shared $\kappa$), or estimate both per model if needed.
- Naming. To match your notation, you can read $T^{host}_M$ as `Th(host)` for model $M$.
