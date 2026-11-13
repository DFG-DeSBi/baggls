import numpy as np
from scipy.stats import norm
import pandas as pd


def avg_contributions(fimo_res, attr, abs_avg=True):
    '''Calculate the average attribution score for each motif in the FIMO results.

    Args:
        fimo_res (pd.DataFrame): FIMO results for the sequences.
        attr (np.ndarray): Attribution scores for the sequences.
        abs_avg (bool): Whether to take the absolute value of the scores before averaging.

    Returns:
        pd.DataFrame: A dataframe with the average attribution scores for each motif per sequence.
    '''

    # Initialize the output dataframe with cumulative sums and counts for averaging
    motifs = fimo_res['motif'].unique()
    seq_range = range(attr.shape[0])

    output_df = pd.DataFrame(index=range(attr.shape[0]), columns=motifs, dtype=float)
    occurrence_count = pd.DataFrame(index=seq_range, columns=motifs, dtype=float).fillna(0)

    # Calculate the maximum attribution score for each motif in the FIMO results
    for motif in motifs:
        motif_data = fimo_res[fimo_res['motif'] == motif]
        for _, row in motif_data.iterrows():
            seq_idx = int(row['sequence'])
            start = int(row['start']) - 1  # Convert to 0-based index
            end = int(row['end'])          

            # Extract values and compute the maximum
            if abs_avg:
                avg_score = np.average(np.abs(attr[seq_idx, start:end]))
            else:
                avg_score = np.average(attr[seq_idx, start:end])

            # Update the maximum value
            if pd.isna(output_df.at[seq_idx, motif]) or output_df.at[seq_idx, motif] < avg_score:
                output_df.at[seq_idx, motif] = avg_score
    output_df = output_df.fillna(0)

    return output_df

def absolute_effects_REST(baggls, top_n=20, single_effects=False, verbose=False):
    exp_z = baggls.mu_z + np.sqrt(baggls.sigma2_z) * (2 * baggls.y - 1) * norm.pdf(baggls.mu_z / np.sqrt(baggls.sigma2_z)) / norm.cdf((2 * baggls.y - 1) * baggls.mu_z / np.sqrt(baggls.sigma2_z))
    beta_mean = np.matmul(baggls.b_beta, exp_z)
    # Get the largest effects
    abs_effects = np.abs(beta_mean)
    top_indices = np.argsort(abs_effects)[-top_n:]  # Top 20 effects
    top_indices = top_indices[::-1]
    top_effect_names = [baggls.effect_names[idx] for idx in top_indices]
    top_effect_values = [beta_mean[idx] for idx in top_indices]
    
    if "MA0138.2.REST.1 AND MA0138.2.REST.2" in top_effect_names:
        rest_idx = top_effect_names.index("MA0138.2.REST.1 AND MA0138.2.REST.2")
        rest_value = top_effect_values[rest_idx]
    else:
        rest_idx = None
        rest_value = None
    
    if single_effects:
        rest1_idx = None
        rest2_idx = None
        rest1_value = None
        rest2_value = None

        if "MA0138.2.REST.1" in top_effect_names:
            rest1_idx = top_effect_names.index("MA0138.2.REST.1")
            rest1_value = top_effect_values[rest1_idx]
        if "MA0138.2.REST.2" in top_effect_names:
            rest2_idx = top_effect_names.index("MA0138.2.REST.2")
            rest2_value = top_effect_values[rest2_idx]

    if verbose:
        print(f"Top {top_n} strongest effects:")
        for idx in top_indices:
            print(f"{baggls.effect_names[idx]}: {beta_mean[idx]:.6f}")

    if single_effects:
        return rest_idx, rest_value, rest1_idx, rest1_value, rest2_idx, rest2_value
    else:
        return rest_idx, rest_value