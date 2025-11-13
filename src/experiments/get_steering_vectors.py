from src.vector_compute.diff_in_means import DiffInMeansCompute
from src.dataset.synthetic_dataset import SyntheticDataset
from src.managers.model_manager import ModelManager
from src.managers.vector_manager import VectorManager
from src.config import Config

class DiffInMeansExp:

    def __init__(self, config_path="config/config.yaml"):
        self.config = Config(config_path)
        self.dataset = SyntheticDataset(self.config.synthetic_data_path)
        self.model = ModelManager(self.config).load_model()
        self.vector_manager = VectorManager()
        self.get_vectors = DiffInMeansCompute(self.vector_manager)

    def run(self):
        exp_cfg = self.config.get_experiment()
        template_type = exp_cfg.get("template_type", "both")
        color_type = exp_cfg.get("color_type", "fundamental")
        activation_pool = exp_cfg.get("activation_pool", "mean")


        contrastive_pairs = self.dataset.get_contastive_pairs(
            template_type, color_type
        )

        print(f"[INFO] Starting Diff-in-Means computation for {len(contrastive_pairs)} colors")

        results = self.get_vectors.get_and_save(
            model=self.model,
            contrastive_pairs=contrastive_pairs,
            template_type=template_type,
            color_type=color_type,
            activation_pool=activation_pool
        )

        print("[INFO] Experiment complete.")
        return results


if __name__ == "__main__":
    exp = DiffInMeansExp("config/config.yaml")
    exp.run()
