"""
Example Usage of Enhanced MEDAF Trainer
Senior-level implementation with best practices
"""

from medaf_trainer import MEDAFTrainer
from core.config_manager import load_config


def train_example():
    """Example of training MEDAF model"""
    print("🚀 Starting MEDAF Training Example")

    # Create trainer with default config
    trainer = MEDAFTrainer("config.yaml")

    # Start training
    results = trainer.train()

    print(f"✅ Training completed!")
    print(f"   Best validation loss: {results['best_val_loss']:.6f}")
    print(f"   Total training time: {results['total_training_time']:.2f} seconds")
    print(f"   Training plots saved to: logs/plots/")


def evaluate_example():
    """Example of evaluating MEDAF model"""
    print("🔍 Starting MEDAF Evaluation Example")

    # Load configuration
    config_manager = load_config("config.yaml")

    # Create trainer
    trainer = MEDAFTrainer("config.yaml")

    # Evaluate with specific checkpoint
    checkpoint_path = config_manager.config.get("checkpoints.evaluation_checkpoint")
    results = trainer.evaluate(checkpoint_path)

    print(f"✅ Evaluation completed!")


def config_example():
    """Example of working with configuration"""
    print("⚙️  Configuration Example")

    # Load configuration
    config_manager = load_config("config.yaml")

    # Access configuration values
    print(f"Model backbone: {config_manager.config.get('model.backbone')}")
    print(f"Learning rate: {config_manager.config.get('training.learning_rate')}")
    print(f"Loss type: {config_manager.config.get('training.loss.type')}")
    print(f"Focal gamma: {config_manager.config.get('training.loss.focal_gamma')}")

    # Get structured arguments
    model_args = config_manager.get_model_args()
    training_args = config_manager.get_training_args()

    print(f"Model args: {model_args}")
    print(f"Training args: {training_args}")


if __name__ == "__main__":
    # Run examples
    # config_example()

    # Uncomment to run training
    train_example()

    # Uncomment to run evaluation
    # evaluate_example()
