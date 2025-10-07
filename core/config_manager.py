"""
Configuration Manager for MEDAF Multi-Label Classification
Senior-level configuration management with YAML support
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration dataclass for type safety and IDE support"""

    def __init__(self, config_dict: Dict[str, Any]):
        # Flatten nested dictionaries for easier access
        self._config = config_dict
        self._flatten_config()

    def _flatten_config(self):
        """Flatten nested configuration for easier access"""

        def _flatten(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        self._flat_config = _flatten(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        # Try flat key first
        if key in self._flat_config:
            return self._flat_config[key]

        # Try nested access
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self._setup_logging()
        self._validate_config()

    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            "config.yaml",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        raise FileNotFoundError(
            f"Configuration file not found. Searched: {possible_paths}"
        )

    def _load_config(self) -> Config:
        """Load and parse YAML configuration with environment variable substitution"""
        try:
            with open(self.config_path, "r") as f:
                config_str = f.read()

            # Environment variable substitution
            config_str = os.path.expandvars(config_str)

            config_dict = yaml.safe_load(config_str)
            return Config(config_dict)

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_level = getattr(logging, self.config.get("logging.level", "INFO").upper())

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Configuration loaded from: {self.config_path}")

    def _validate_config(self):
        """Validate critical configuration parameters"""
        required_keys = [
            "data.source",
            "training.batch_size",
            "training.num_epochs",
            "training.learning_rate",
            "model.num_classes",
        ]

        missing_keys = []
        for key in required_keys:
            if self.config.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # Validate paths
        paths_to_check = [
            "data.train_csv",
            "data.test_csv",
            "data.image_root",
        ]

        for path_key in paths_to_check:
            path_value = self.config.get(path_key)
            if path_value and not Path(path_value).exists():
                self.logger.warning(f"Path does not exist: {path_key} = {path_value}")

    def get_model_args(self) -> Dict[str, Any]:
        """Get model-specific arguments"""
        return {
            "img_size": int(self.config.get("data.img_size", 224)),
            "backbone": self.config.get("model.backbone", "resnet18"),
            "num_classes": int(self.config.get("model.num_classes", 8)),
            "gate_temp": int(self.config.get("model.gate_temp", 100)),
            "loss_keys": self.config.get(
                "model.loss_keys", ["b1", "b2", "b3", "gate", "divAttn", "total"]
            ),
            "acc_keys": self.config.get(
                "model.acc_keys", ["acc1", "acc2", "acc3", "accGate"]
            ),
            "loss_wgts": self.config.get("model.loss_weights", [0.7, 1.0, 0.01]),
        }

    def get_training_args(self) -> Dict[str, Any]:
        """Get training-specific arguments"""
        return {
            "batch_size": int(self.config.get("training.batch_size", 32)),
            "num_epochs": int(self.config.get("training.num_epochs", 50)),
            "learning_rate": float(self.config.get("training.learning_rate", 5e-5)),
            "weight_decay": float(self.config.get("training.weight_decay", 1e-4)),
            "val_ratio": float(self.config.get("training.val_ratio", 0.1)),
            "num_workers": int(self.config.get("training.num_workers", 1)),
        }

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration"""
        return {
            "type": self.config.get("training.optimizer.type", "adamw"),
            "betas": self.config.get("training.optimizer.betas", [0.9, 0.999]),
            "eps": float(self.config.get("training.optimizer.eps", 1e-8)),
            "amsgrad": bool(self.config.get("training.optimizer.amsgrad", False)),
        }

    def get_loss_config(self) -> Dict[str, Any]:
        """Get loss configuration"""
        return {
            "type": self.config.get("training.loss.type", "focal"),
            "focal_alpha": float(self.config.get("training.loss.focal_alpha", 1.0)),
            "focal_gamma": float(self.config.get("training.loss.focal_gamma", 2.0)),
            "class_weighting_enabled": bool(
                self.config.get("training.loss.class_weighting.enabled", True)
            ),
            "class_weighting_method": self.config.get(
                "training.loss.class_weighting.method", "inverse_freq"
            ),
        }

    def create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.config.get("checkpoints.dir"),
            self.config.get("logging.metrics_dir"),
        ]

        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {dir_path}")


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Convenience function to load configuration"""
    return ConfigManager(config_path)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    try:
        config_manager = load_config()
        print("✅ Configuration loaded successfully")
        print(f"Model args: {config_manager.get_model_args()}")
        print(f"Training args: {config_manager.get_training_args()}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
