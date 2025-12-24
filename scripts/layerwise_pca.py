import argparse
import logging
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from einops import rearrange
import numpy as np
import matplotlib.patches as mpatches

from colorrep.steering_vector_utils import load_steering_vector
from colorrep.config import INDIRECT_COLORS, DIRECT_COLORS, COLORS_TO_HEX

def procrusted_pca(args, output_dir, colors):
    """
    This function performs PCA on the steering vectors and performs procrusted alignment: 
    i.e. it finds the optimal rotation and translation that minimizes the difference between the
    two adjacent layers. 
    This means that the variance explained by PC1 and PC2 is not guaranteed to be the same 
    as the variance explained by the original PCA dimensions!
    """
    
    hex_values = [COLORS_TO_HEX[color] for color in colors]

    previous_projections = None
    aligned_projections = []
    for layer_num in tqdm(range(32), desc="Processing layers"):
        svs = []
        for color in colors:
            sv = load_steering_vector(args.sv_dir, color, layer_num)
            svs.append(sv)
        pca = PCA(n_components=2)
        current_projections = pca.fit_transform(svs)
        
        # 2. PROCRUSTES ALIGNMENT: Enforces rotation invariance
        if previous_projections is not None:
            # standardizes the orientation by finding the optimal rotation R
            # that maps current_projections -> previous_projections
            R, _ = orthogonal_procrustes(previous_projections, current_projections)
            
            # Apply the rotation. 
            # Note: orthogonal_procrustes calculates A @ R.T (conceptually), 
            # but scipy returns the matrix R such that A ~ B @ R.
            # So we update current to align with previous.
            current_projections = current_projections @ R
        
        total_var = np.sum(pca.explained_variance_ratio_)
            
        # Store for next iteration
        previous_projections = current_projections
        aligned_projections.append(current_projections)

        plt.scatter(current_projections[:, 0], current_projections[:, 1], c=hex_values, s=35)
        plt.title(f"Layer {layer_num} - Total Var Expl: {total_var:.2%}")
        plt.xlabel(f"Aligned Dimension 1")
        plt.ylabel(f"Aligned Dimension 2")
        plt.savefig(f"{output_dir}/layer_{layer_num}_procrusted.png")
        plt.close()

    return aligned_projections


def reflection_handling_pca(args, output_dir, colors):
    """
    This function performs PCA on the steering vectors and handles reflections: 
    i.e. if the difference between two adjacent layers would be minimzed by reflecting one or both
    of the PCA dimensions, then we reflect the dimensions.
    """
    hex_values = [COLORS_TO_HEX[color] for color in colors]

    previous_components = None
    for layer_num in tqdm(range(32), desc="Processing layers"):
        svs = []
        for color in colors:
            sv = load_steering_vector(args.sv_dir, color, layer_num)
            svs.append(sv)
        # rearrange to (n_samples, n_dimensions)
        # svs = rearrange(svs, "c d -> d c")
        pca = PCA(n_components=2)
        pca.fit(svs)

        if previous_components is not None:
            for i in range(2):
                dot_prod = np.dot(pca.components_[i], previous_components[i])
                if dot_prod < 0:
                    pca.components_[i] *= -1

        previous_components = pca.components_.copy()
        projections = pca.transform(svs)

        plt.scatter(projections[:, 0], projections[:, 1], c=hex_values, s=45, edgecolors='black', linewidths=0.5)
        plt.title(f"Layer {layer_num} PCA")
        plt.xlabel(f"PC1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PC2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2f})")
        legend_elements = [mpatches.Patch(facecolor=hex_val, label=color_name) 
                          for color_name, hex_val in zip(colors, hex_values)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.26, 1.06), loc='upper right', title="Colors", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/layer_{layer_num}_reflectioned.png")
        plt.close()

    return projections


def plot_variance_over_layers(args, output_dir, colors):
    """
    Analyzes the dimensionality of steering vectors across layers by tracking
    cumulative variance explained for different k.
    """
    n_colors = len(colors)
    variance_history = [] 
    for layer_num in tqdm(range(32), desc="Processing layers"):
        svs = []
        for color in colors:
            sv = load_steering_vector(args.sv_dir, color, layer_num)
            svs.append(sv)

        pca = PCA(n_components=n_colors)
        pca.fit(svs)
        
        # Calculate cumulative variance: [PC1, PC1+PC2, PC1+PC2+PC3...]
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        variance_history.append(cum_var)

    # 3. Process Data for Plotting
    # Pad arrays with 1.0 if some layers had fewer components than others (rare but possible)
    max_k_found = max(len(v) for v in variance_history)
    variance_matrix = np.ones((len(variance_history), max_k_found))
    
    for i, v in enumerate(variance_history):
        variance_matrix[i, :len(v)] = v

    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Define which k values to plot lines for
    # We dynamically cap k at the number of components actually found
    ks_to_plot = [1, 2, 3, 5, 7, 10]
    
    # Use a sequential colormap
    colors = plt.cm.magma(np.linspace(0.2, 0.8, len(ks_to_plot)))

    for idx, k in enumerate(ks_to_plot):
        # variance_matrix is 0-indexed, so k=1 is at index 0
        y_values = variance_matrix[:, k-1] 
        
        plt.plot(range(32), y_values, 
                 label=f'{k} Components', 
                 color=colors[idx], 
                 linewidth=2.5, 
                 alpha=0.9)

    plt.title("Dimensionality of Color Subspace Across Layers", fontsize=14)
    plt.ylabel("Cumulative Variance Explained", fontsize=12)
    plt.xlabel("Layer Number", fontsize=12)
    
    # Force Y-axis to 0-1 for clarity
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Number of Components")
    plt.tight_layout()
    
    save_path = f"{output_dir}/explained_variance_over_layers.png"
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved dimensionality plot to {save_path}")
    plt.close()


def main(args):
    output_dir = f"{args.output_dir}/{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    colors = INDIRECT_COLORS if args.dataset_name == "indirect" else DIRECT_COLORS
    reflection_handling_pca(args, output_dir, colors)
    plot_variance_over_layers(args, output_dir, colors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sv_dir", default="/projectnb/cs599m1/projects/color-representations/outputs_latest/steering_vectors_predict/regular_templates/fundamental_colors", type=str, help="Path to the steering vectors")
    parser.add_argument("--output_dir", default="/projectnb/cs599m1/projects/color-representations/img/layerwise_pca", type=str, help="Path to the output directory")
    parser.add_argument("--dataset_name", default="indirect", type=str, help="Name of the dataset")
    args = parser.parse_args()
    main(args)
