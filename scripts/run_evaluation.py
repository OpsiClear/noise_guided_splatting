import os
import tyro
from ngs.config import EvaluatorConfig as Config
from ngs.evaluator import SplatsEvaluator
from ngs.utils.utils import save_ply

def main():
    """Main function to evaluate Gaussian splats."""
    cfg = tyro.cli(Config)

    # Ensure required fields from config are present (previously handled by argparse required=True)
    if cfg.data_dir is None:
        print("Error: --data_dir must be specified.")
        return 
    if cfg.surface_pcloud is None:
        print("Error: --surface_pcloud must be specified.") # Tyro converts snake_case to kebab-case for CLI
        return


    # Create evaluator
    evaluator = SplatsEvaluator(cfg)

    # Save ground truth images once before starting evaluations
    evaluator.save_gt_images()

    # Load surface gaussians
    evaluator.load_surface_splats(cfg.surface_pcloud) # Use cfg.surface_pcloud

    # Evaluate surface gaussians alone
    stats = evaluator.evaluate(output_prefix="surface", transparency=False)
    print(stats)

    # If inside gaussians path is provided, add them and evaluate
    if cfg.inside_pcloud is not None: 
        # Add inside gaussians (green)
        evaluator.add_inside_splats(cfg.inside_pcloud, color_inside=[0.0, 1.0, 0.0], opacity_inside=1.0)

        # Evaluate combined model
        stats = evaluator.evaluate(output_prefix="infill", transparency=False)
        print(stats)
        save_ply(evaluator.splats, os.path.join(cfg.result_dir, "infill.ply"))

        # Colorize surface gaussians to red for better visualization
        evaluator.colorize_surface_splats([1.0, 0.0, 0.0])

        # Evaluate with red surface and green inside
        stats = evaluator.evaluate(output_prefix="transparency", transparency=True)
        print(stats)
        save_ply(evaluator.splats, os.path.join(cfg.result_dir, "transparency.ply"))

if __name__ == "__main__":
    main()