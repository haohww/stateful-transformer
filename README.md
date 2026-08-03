# Stateful Transformer

Linear attention can be viewed as a recurrent key-value memory. Its evolution is mainly a story of increasingly precise control over what that fixed-size state retains.

```mermaid
%%{init: { "htmlLabels": true, "flowchart": { "wrappingWidth": 640 } } }%%
flowchart TD
    LA["<b>Linear Attention</b> · 2020<br/><code>Sₜ&nbsp;=&nbsp;Sₜ₋₁&nbsp;+&nbsp;kₜvₜᵀ</code><br/><small>Add every key-value association</small>"]
    DN["<b>DeltaNet</b> · 2021<br/><code>Sₜ&nbsp;=&nbsp;(I&nbsp;−&nbsp;βₜkₜkₜᵀ)Sₜ₋₁&nbsp;+&nbsp;βₜkₜvₜᵀ</code><br/><small>Correct the value stored at the current key</small>"]
    GDN["<b>Gated DeltaNet</b> · 2025<br/><code>Sₜ&nbsp;=&nbsp;αₜ(I&nbsp;−&nbsp;βₜkₜkₜᵀ)Sₜ₋₁&nbsp;+&nbsp;βₜkₜvₜᵀ</code><br/><small>Add scalar, token-wise state decay</small>"]
    KDA["<b>Kimi Delta Attention</b> · 2025<br/><code>Sₜ&nbsp;=&nbsp;(I&nbsp;−&nbsp;βₜkₜkₜᵀ)Diag(αₜ)Sₜ₋₁&nbsp;+&nbsp;βₜkₜvₜᵀ</code><br/><small>Replace scalar decay with fine-grained diagonal decay</small>"]

    LA -->|additive memory cannot edit associations| DN
    DN -->|targeted edits cannot quickly clear unrelated memory| GDN
    GDN -->|one decay rate is too coarse| KDA

    classDef linear fill:#f7f7f5,stroke:#9ca3af,color:#111827;
    classDef delta fill:#ecfdf5,stroke:#34d399,color:#064e3b;
    classDef gated fill:#eef2ff,stroke:#818cf8,color:#312e81;
    classDef kimi fill:#fff7ed,stroke:#fb923c,color:#7c2d12;
    class LA linear;
    class DN delta;
    class GDN gated;
    class KDA kimi;
```

Here, `Sₜ ∈ ℝᵈᵏˣᵈᵛ` is the recurrent memory, `kₜ`, `vₜ`, and `qₜ` are the key, value, and query, `βₜ` is the write strength, and `αₜ` is the forget gate; the readout is `oₜ = Sₜᵀqₜ`. Gated DeltaNet uses a scalar `αₜ`, while KDA promotes it to a vector and applies `Diag(αₜ)`.

## Kimi K3 architecture

KDA is the recurrent token mixer; the `3 KDA : 1 Gated MLA` cadence is the surrounding hybrid backbone. Kimi K3 repeats that block 23 times, then adds one final Gated MLA layer: 69 KDA and 24 Gated MLA layers in total.

```mermaid
%%{init: { "htmlLabels": true, "flowchart": { "wrappingWidth": 360 } } }%%
flowchart TD
    IN["<b>Token stream</b>"]

    subgraph B["Hybrid block × 23"]
        direction TD
        K1["<b>KDA</b><br/><small>fixed-size recurrent state</small>"]
        K2["<b>KDA</b><br/><small>fixed-size recurrent state</small>"]
        K3["<b>KDA</b><br/><small>fixed-size recurrent state</small>"]
        MLA["<b>Gated MLA</b><br/><small>global attention · latent KV cache</small>"]
        K1 --> K2 --> K3 --> MLA
    end

    LAST["<b>Final Gated MLA</b><br/><small>the last layer is always global</small>"]
    OUT["<b>Output</b>"]

    IN --> K1
    MLA -->|after block 23| LAST --> OUT

    classDef kda fill:#fff7ed,stroke:#fb923c,color:#7c2d12;
    classDef mla fill:#eef2ff,stroke:#818cf8,color:#312e81;
    classDef io fill:#f7f7f5,stroke:#9ca3af,color:#111827;
    class K1,K2,K3 kda;
    class MLA,LAST mla;
    class IN,OUT io;
```

Every attention layer is paired with a Stable LatentMoE channel mixer. Attention Residuals separately let a layer retrieve from the embedding, earlier blocks, and the current partial block instead of relying only on the immediately preceding residual stream.

### Inside a KDA layer

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
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653) — Kimi Team, 2026
