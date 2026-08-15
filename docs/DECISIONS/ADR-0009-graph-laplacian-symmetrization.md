# ADR-0009: Graph Laplacian Symmetrization Protocol for Chebyshev Convolutions

**Status:** Accepted
**Date:** 2026-08-15
**Resolves:** Spectral graph convolution on directed road networks (PEMS-BAY)

## Context

The canonical PEMS-BAY dataset contains a directed sensor distance graph (`adj_mx_bay.pkl`) where $W_{ij} \ne W_{ji}$ due to asymmetric road connectivity and directionality.

Chebyshev polynomial graph convolutions (Hammond et al., 2011; Defferrard et al., 2016; Yu et al., 2018) require a normalized graph Laplacian $L = I - D^{-1/2} W D^{-1/2}$ defined on a symmetric (undirected) adjacency matrix $W = W^T$ to guarantee real eigenvalues bounded in $[0, 2]$ and orthonormal graph Fourier basis functions.

## Decision

1. **Strict Invariant in Operator Library:** `st_dssm.graph.calculate_normalized_laplacian` requires a symmetric adjacency matrix and raises `GraphError` if $W \ne W^T$.
2. **Explicit Symmetrization Protocol:** For directed networks such as PEMS-BAY, the canonical symmetrization operator is $W_{\text{sym}} = \frac{1}{2}(W + W^T)$ implemented by `st_dssm.graph.symmetrize_adjacency(adj, method="average")`.
3. **No Silent Transformations:** The CLI and modeling pipelines must not silently mutate graphs. Symmetrization must be explicitly declared via parameter `symmetrize_graph: bool` or CLI argument `--symmetrize-graph`.
4. **Auditability:** When symmetrization is enabled, the execution log and run manifest must explicitly record `"symmetrize_graph": true` and `"symmetrization_method": "average"`.

## Consequences

Graph preprocessing is explicit, fully documented, auditable in manifests, and prevents silent semantic drift between baselines and ST-DSSM models.
