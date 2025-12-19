import argparse
import logging
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from einops import rearrange
import numpy as np

from colorrep.steering_vector_utils import load_steering_vector
from colorrep.config import FUNDAMENTAL_COLORS, COLORS_TO_HEX

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    hex_values = [COLORS_TO_HEX[color] for color in FUNDAMENTAL_COLORS]

    previous_components = None
    for layer_num in tqdm(range(32), desc="Processing layers"):
        svs = []
        for color in FUNDAMENTAL_COLORS:
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

        plt.scatter(projections[:, 0], projections[:, 1], c=hex_values, s=15)
        plt.title(f"Layer {layer_num} PCA")
        plt.xlabel(f"PC1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PC2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2f})")
        plt.savefig(f"{args.output_dir}/layer_{layer_num}.png")
        plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sv_dir", default="/projectnb/cs599m1/projects/color-representations/outputs_latest/steering_vectors_predict/regular_templates/fundamental_colors", type=str, help="Path to the steering vectors")
    parser.add_argument("--output_dir", default="/projectnb/cs599m1/projects/color-representations/img/layerwise_pca", type=str, help="Path to the output directory")
    args = parser.parse_args()
    main(args)
