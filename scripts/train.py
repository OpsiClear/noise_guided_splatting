import os
import tyro
from ngs.config import TrainerConfig as Config
from ngs.trainer import main as train
from gsplat.distributed import cli

def main():
    """Main function to train Gaussian splats."""
    cfg = tyro.cli(Config)
    cli(train, cfg, verbose=True)

if __name__ == "__main__":
    main()