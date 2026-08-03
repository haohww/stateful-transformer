# Stateful Transformer

Linear attention can be viewed as a recurrent key-value memory. Its evolution is mainly a story of increasingly precise control over what that fixed-size state retains.

```mermaid
flowchart TD
    LA["<b>Linear Attention</b> · 2020<br/><code>Sₜ = Sₜ₋₁ + kₜvₜᵀ</code><br/><small>Add every key-value association</small>"]
    DN["<b>DeltaNet</b> · 2021<br/><code>Sₜ = (I − βₜkₜkₜᵀ)Sₜ₋₁ + βₜkₜvₜᵀ</code><br/><small>Correct the value stored at the current key</small>"]
    GDN["<b>Gated DeltaNet</b> · 2025<br/><code>Sₜ = αₜ(I − βₜkₜkₜᵀ)Sₜ₋₁ + βₜkₜvₜᵀ</code><br/><small>Add scalar, token-wise state decay</small>"]
    KDA["<b>Kimi Delta Attention</b> · 2025<br/><code>Sₜ = (I − βₜkₜkₜᵀ)Diag(αₜ)Sₜ₋₁ + βₜkₜvₜᵀ</code><br/><small>Replace scalar decay with fine-grained diagonal decay</small>"]

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

## References

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) — Katharopoulos et al., 2020
- [Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) — Schlag et al., 2021
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — Yang et al., ICLR 2025
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — Kimi Team, 2025
