"""Unit 4 -- Unbiased Term Selection and Grouping Pipeline."""

from dataclasses import dataclass
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from gsea_tool.data_ingestion import CohortData
from gsea_tool.cherry_picked import CategoryGroup


@dataclass
class UnbiasedSelectionStats:
    """Statistics collected during unbiased selection for notes.md."""
    total_significant_terms: int  # after step 1
    terms_after_dedup: int  # after step 3
    terms_selected: int  # after step 4 (top N)
    n_clusters: int  # step 5 parameter
    random_seed: int  # step 5 parameter
    clustering_algorithm: str  # e.g. "scipy.cluster.hierarchy (Ward linkage)"


def pool_significant_terms(
    cohort: CohortData,
    fdr_threshold: float,
) -> dict[str, float]:
    """Step 1-2: Pool terms passing FDR threshold in any mutant, compute max abs NES.

    Returns dict mapping term_name -> max_absolute_nes, sorted by value descending.
    """
    # Step 1: Pool terms that pass FDR threshold in at least one mutant
    pooled_terms: set[str] = set()
    for mutant_id in cohort.mutant_ids:
        profile = cohort.profiles[mutant_id]
        for term_name, record in profile.records.items():
            if record.fdr < fdr_threshold:
                pooled_terms.add(term_name)

    # Step 2: For pooled terms, compute max abs NES across ALL mutants
    term_max_abs_nes: dict[str, float] = {}
    for term_name in pooled_terms:
        max_abs = 0.0
        for mutant_id in cohort.mutant_ids:
            profile = cohort.profiles[mutant_id]
            record = profile.records.get(term_name)
            if record is not None:
                abs_nes = abs(record.nes)
                if abs_nes > max_abs:
                    max_abs = abs_nes
        term_max_abs_nes[term_name] = max_abs

    # Sort by max abs NES descending, ties broken alphabetically by term name
    sorted_items = sorted(
        term_max_abs_nes.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return dict(sorted_items)


def remove_redundant_terms(
    ranked_terms: dict[str, float],
) -> dict[str, float]:
    """Step 3: Remove lexically redundant terms.

    For each pair of terms sharing substantial word overlap (Jaccard similarity
    of word sets > 0.5), retain only the term with higher max abs NES.
    """
    # Process terms in rank order (dict preserves insertion order, already sorted)
    surviving: list[tuple[str, float]] = []
    surviving_word_sets: list[set[str]] = []

    for term_name, max_abs_nes in ranked_terms.items():
        term_words = set(term_name.split())
        is_redundant = False

        for existing_words in surviving_word_sets:
            intersection = len(term_words & existing_words)
            union = len(term_words | existing_words)
            if union > 0:
                jaccard = intersection / union
                if jaccard > 0.5:
                    is_redundant = True
                    break

        if not is_redundant:
            surviving.append((term_name, max_abs_nes))
            surviving_word_sets.append(term_words)

    return dict(surviving)


def select_top_n(
    ranked_terms: dict[str, float],
    top_n: int,
) -> list[str]:
    """Step 4: Select top N terms from deduplicated ranked list."""
    return list(ranked_terms.keys())[:top_n]


def _compute_mean_abs_nes(term_name: str, cohort: CohortData) -> float:
    """Compute mean absolute NES for a term across all mutants.

    Uses NES=0 for mutants where the term is absent.
    """
    total = 0.0
    for mutant_id in cohort.mutant_ids:
        profile = cohort.profiles[mutant_id]
        rec = profile.records.get(term_name)
        if rec is not None:
            total += abs(rec.nes)
    return total / len(cohort.mutant_ids)


def cluster_terms(
    term_names: list[str],
    cohort: CohortData,
    n_groups: int,
    random_seed: int,
) -> list[CategoryGroup]:
    """Steps 5-6: Cluster selected terms by NES profile and auto-label groups.

    Uses hierarchical agglomerative clustering (Ward linkage) on the NES profile
    matrix (terms as rows, mutants as columns). Missing NES values are treated as 0.0.
    Each group is labeled with the term having the highest mean absolute NES
    within that group. Terms within each group are sorted by mean absolute NES
    descending. Groups are sorted by the position of their highest-ranked member
    in the original top-N ranking.
    """
    # Set random seed defensively
    np.random.seed(random_seed)

    n_terms = len(term_names)

    # Build NES profile matrix: terms x mutants
    n_mutants = len(cohort.mutant_ids)
    profile_matrix = np.zeros((n_terms, n_mutants))

    for i, term_name in enumerate(term_names):
        for j, mutant_id in enumerate(cohort.mutant_ids):
            profile = cohort.profiles[mutant_id]
            rec = profile.records.get(term_name)
            if rec is not None:
                profile_matrix[i, j] = rec.nes

    # Handle edge case: only 1 term -> single group
    if n_terms == 1:
        return [CategoryGroup(
            category_name=term_names[0],
            term_names=list(term_names),
        )]

    # Hierarchical agglomerative clustering with Ward linkage
    Z = linkage(profile_matrix, method='ward')
    cluster_labels = fcluster(Z, t=n_groups, criterion='maxclust')

    # Build rank position map from the original term_names ordering
    rank_position = {name: idx for idx, name in enumerate(term_names)}

    # Group terms by cluster label
    cluster_to_terms: dict[int, list[str]] = {}
    for i, label in enumerate(cluster_labels):
        label_int = int(label)
        if label_int not in cluster_to_terms:
            cluster_to_terms[label_int] = []
        cluster_to_terms[label_int].append(term_names[i])

    # Build CategoryGroup for each cluster
    groups: list[CategoryGroup] = []
    for label, terms in cluster_to_terms.items():
        # Sort terms within group by mean absolute NES descending
        terms.sort(key=lambda t: _compute_mean_abs_nes(t, cohort), reverse=True)

        # Label the group with the term having highest mean abs NES
        group_label = terms[0]

        # Find the highest-ranked member (lowest rank position)
        min_rank = min(rank_position[t] for t in terms)

        groups.append((min_rank, CategoryGroup(
            category_name=group_label,
            term_names=terms,
        )))

    # Sort groups by the rank position of their highest-ranked member
    groups.sort(key=lambda x: x[0])

    return [g for _, g in groups]


def select_unbiased_terms(
    cohort: CohortData,
    fdr_threshold: float = 0.05,
    top_n: int = 20,
    n_groups: int = 4,
    random_seed: int = 42,
) -> tuple[list[CategoryGroup], UnbiasedSelectionStats]:
    """Top-level entry point for unbiased term selection (Figure 2).

    Returns the grouped terms and collection statistics for notes.md.
    """
    assert top_n > 0, "top_n must be a positive integer"
    assert n_groups > 0, "n_groups must be a positive integer"
    assert n_groups <= top_n, "Cannot have more groups than selected terms"

    # Step 1-2: Pool and rank significant terms
    pooled = pool_significant_terms(cohort, fdr_threshold)
    total_significant = len(pooled)

    # Check if we have enough terms for clustering
    if total_significant < n_groups:
        raise ValueError(
            f"Insufficient significant terms: found {total_significant} terms "
            f"passing FDR threshold {fdr_threshold}, but need at least {n_groups} "
            f"for clustering into {n_groups} groups"
        )

    # Step 3: Remove redundant terms
    deduped = remove_redundant_terms(pooled)
    terms_after_dedup = len(deduped)

    # Step 4: Select top N
    selected = select_top_n(deduped, top_n)
    terms_selected = len(selected)

    # Adjust n_groups if we have fewer terms than requested groups
    actual_n_groups = min(n_groups, terms_selected)

    # Steps 5-6: Cluster and label
    groups = cluster_terms(selected, cohort, actual_n_groups, random_seed)

    stats = UnbiasedSelectionStats(
        total_significant_terms=total_significant,
        terms_after_dedup=terms_after_dedup,
        terms_selected=terms_selected,
        n_clusters=actual_n_groups,
        random_seed=random_seed,
        clustering_algorithm="scipy.cluster.hierarchy (Ward linkage)",
    )

    return groups, stats
