from src.vector_compute.diff_in_means import DiffInMeansCompute
from src.managers.model_manager import ModelManager
from src.managers.vector_manager import VectorManager
from src.colorep.config import Config
from src.dataset.synthetic_dataset import SyntheticDataset, CommonObjects



def load_dataset(dataset_cfg):
    dtype = dataset_cfg.get("type", "synthetic")
    path  = dataset_cfg.get("path")

    if dtype == "synthetic":
        return SyntheticDataset(path)

    elif dtype == "common_objects":
        return CommonObjects(path)

    else:
        raise ValueError(f"Unknown dataset type: {dtype}")


class DiffInMeansExp:

    def __init__(self, config_path="config/config.yaml"):
        self.config = Config(config_path)

        # Load dataset dynamically
        self.dataset = load_dataset(self.config.get_dataset_config())

        # Load model + vector utilities
        self.model = ModelManager(self.config).load_model()
        self.vector_manager = VectorManager()
        self.get_vectors = DiffInMeansCompute(self.vector_manager, self.config.save_dir)

    def run(self):
        exp_cfg = self.config.get_experiment()

        template_type   = exp_cfg.get("template_type", "both")
        color_type      = exp_cfg.get("color_type", "fundamental")
        activation_pool = exp_cfg.get("activation_pool", "mean")
        mode            = exp_cfg.get("mode", "contrastive")

        print(f"[INFO] Running experiment in mode: {mode}")
        print(f"[INFO] Dataset: {self.dataset.__class__.__name__}")

        if mode == "contrastive":
            return self._run_contrastive(template_type, color_type, activation_pool)

        elif mode == "predictions":
            return self._run_predictions(template_type, color_type, activation_pool)

        else:
            raise ValueError(f"Invalid mode in config: {mode}")

    def _run_contrastive(self, template_type, color_type, activation_pool):
        color_sentences = self.dataset.get_color_sentences(
            template_type, color_type
        )

        print(f"[INFO] Starting contrastive Diff-in-Means for {color_sentences.keys()} colors")

        results = self.get_vectors.get_and_save(
            model=self.model,
            color_sentences=color_sentences,
            template_type=template_type,
            color_type=color_type,
            activation_pool=activation_pool
        )

        print("[INFO] Contrastive experiment complete.")
        return results

    def _run_predictions(self, template_type, color_type, activation_pool):

        sents = self.dataset.get_all_common_objects()

        print(f"[INFO] Starting prediction-based Diff-in-Means for {len(sents)} sentences")

        results = self.get_vectors.get_and_save_predictions(
            model=self.model,
            sents=sents,
            template_type=template_type,
            color_type=color_type,
            activation_pool=activation_pool
        )

        print("[INFO] Prediction-based experiment complete.")
        return results


if __name__ == "__main__":
    exp = DiffInMeansExp("config/config_.yaml")
    exp.run()

