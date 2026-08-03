# Recurrent Transformer

Linear attention can be viewed as a recurrent key-value memory. Its evolution is mainly a story of increasingly precise control over what that fixed-size state retains.

![Evolution from Linear Attention to Kimi Delta Attention](./assets/attention-evolution.svg)

Here, `Sₜ ∈ ℝᵈᵏˣᵈᵛ` is the recurrent memory, `kₜ`, `vₜ`, and `qₜ` are the key, value, and query, `βₜ` is the write strength, and `αₜ` is the forget gate; the readout is `oₜ = Sₜᵀqₜ`. Gated DeltaNet uses a scalar `αₜ`, while KDA promotes it to a vector and applies `Diag(αₜ)`.

## Kimi K3 architecture

Kimi K3 has 93 Transformer layers. It organizes them as 23 hybrid groups of four layers—three KDA-based layers followed by one Gated-MLA-based layer—then adds one final Gated-MLA-based layer. This gives 69 KDA and 24 Gated MLA layers in total.

```mermaid
%%{init: { "htmlLabels": true, "flowchart": { "wrappingWidth": 360 } } }%%
flowchart TD
    IN["<b>Token stream</b>"]

    subgraph B["Hybrid group × 23 · four Transformer layers per group"]
        direction TD
        K1["<b>KDA-based Transformer layer</b><br/><small>KDA token mixer → Stable LatentMoE</small>"]
        K2["<b>KDA-based Transformer layer</b><br/><small>KDA token mixer → Stable LatentMoE</small>"]
        K3["<b>KDA-based Transformer layer</b><br/><small>KDA token mixer → Stable LatentMoE</small>"]
        MLA["<b>Gated-MLA-based Transformer layer</b><br/><small>Gated MLA token mixer → Stable LatentMoE</small>"]
        K1 --> K2 --> K3 --> MLA
    end

    LAST["<b>Final Gated-MLA-based Transformer layer</b><br/><small>Gated MLA token mixer → Stable LatentMoE<br/>the final token mixer is global</small>"]
    OUT["<b>Output</b>"]

    IN --> K1
    MLA -->|after group 23| LAST --> OUT

    classDef kda fill:#fff7ed,stroke:#fb923c,color:#7c2d12;
    classDef mla fill:#eef2ff,stroke:#818cf8,color:#312e81;
    classDef io fill:#f7f7f5,stroke:#9ca3af,color:#111827;
    class K1,K2,K3 kda;
    class MLA,LAST mla;
    class IN,OUT io;
```

Each box inside a hybrid group is one complete Transformer layer: a token mixer followed by a Stable LatentMoE channel mixer. Normalization and Attention Residual routing are omitted from this overview; AttnRes lets modules retrieve from the embedding, earlier groups, and the current partial group instead of relying only on the immediately preceding residual stream.

### Attention Residuals: routing across depth

Attention Residuals (AttnRes) operate across network depth, not across token positions. Standard PreNorm residuals pass one running sum from layer to layer, so every earlier layer output has a fixed coefficient of `1`. AttnRes instead uses a learned pseudo-query for each module to compute input-dependent softmax weights over earlier representations.

![Classic residual connections compared with Attention Residuals](./assets/attention-residuals.svg)

Full AttnRes attends to every preceding sublayer output. Kimi K3 uses the scalable Block AttnRes variant: it keeps the embedding, completed hybrid-group summaries, and the current partial-group sum as candidates. This reduces residual-state memory from `O(Ld)` to `O(Nd)`, where `L` is the number of sublayers and `N` is the number of groups.

### Inside the KDA token mixer

This expands only the KDA token-mixing component of a KDA-based Transformer layer—not the entire layer or its Stable LatentMoE component.

```mermaid
%%{init: { "htmlLabels": true, "flowchart": { "wrappingWidth": 520 } } }%%
flowchart TD
    X["<b>Input xₜ</b>"]
    QK["<b>qₜ, kₜ</b><br/><small>Linear → ShortConv → Swish → L2Norm</small>"]
    V["<b>vₜ</b><br/><small>Linear → ShortConv → Swish</small>"]
    A["<b>Retention αₜ</b><br/><small>channel-wise, lower-bounded decay</small>"]
    BT["<b>Write βₜ</b><br/><small>Linear → sigmoid</small>"]
    MEM["<b>Recurrent memory</b><br/><code>Sₜ&nbsp;=&nbsp;(I&nbsp;−&nbsp;βₜkₜkₜᵀ)Diag(αₜ)Sₜ₋₁&nbsp;+&nbsp;βₜkₜvₜᵀ</code><br/><small>read: õₜ = Sₜᵀqₜ</small>"]
    NORM["<b>Head-wise RMSNorm</b>"]
    GATE["<b>Output gate</b><br/><small>sigmoid(Wg xₜ)</small>"]
    MIX["<b>Gate × normalized readout</b>"]
    PROJ["<b>Output projection Wₒ</b>"]
    Y["<b>Output yₜ</b>"]

    X --> QK --> MEM
    X --> V --> MEM
    X --> A --> MEM
    X --> BT --> MEM
    MEM --> NORM --> MIX
    X --> GATE --> MIX
    MIX --> PROJ --> Y

    classDef input fill:#f7f7f5,stroke:#9ca3af,color:#111827;
    classDef projection fill:#ecfdf5,stroke:#34d399,color:#064e3b;
    classDef state fill:#fff7ed,stroke:#fb923c,color:#7c2d12;
    classDef output fill:#eef2ff,stroke:#818cf8,color:#312e81;
    class X input;
    class QK,V,A,BT projection;
    class MEM state;
    class NORM,GATE,MIX,PROJ,Y output;
```

Kimi K3 keeps the KDA recurrence from Kimi Linear but bounds the decay and uses a full-rank, input-dependent output gate. The recurrence is sequential across chunks and parallel within each chunk.

## References

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) — Katharopoulos et al., 2020
- [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) — Schlag et al., 2021
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — Yang et al., ICLR 2025
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — Kimi Team, 2025
- [Attention Residuals](https://arxiv.org/abs/2603.15031) — Kimi Team, 2026
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653) — Kimi Team, 2026
