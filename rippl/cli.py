import argparse
import sys
import os
import yaml
import torch
from rippl.api import train, simulate
from rippl.models.registry import load_model
from rippl.export.exporter import export_model
from rippl.core.config import ConfigParser

def main():
    """
    Rippl CLI Entrypoint.
    Provides commands for training, simulation, and model export.
    """
    parser = argparse.ArgumentParser(
        description="Rippl: Modular Physics-ML Framework for Enterprise Wave PDEs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a PINN model from a configuration file")
    train_parser.add_argument("config", help="Path to the YAML or JSON configuration file")
    train_parser.add_argument("--output", "-o", default="checkpoint", help="Directory to save the trained model")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run inference or a numerical solver")
    sim_parser.add_argument("config", help="Path to the configuration file")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export a trained model to TorchScript or ONNX")
    export_parser.add_argument("config", help="Path to the configuration file used for training")
    export_parser.add_argument("model_dir", help="Directory containing the weights.pt and config.json")
    export_parser.add_argument("--format", default="torchscript", choices=["torchscript", "onnx"], 
                                help="Target export format (default: torchscript)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    try:
        if args.command == "train":
            print(f"[Rippl] Initializing training from {args.config}...")
            model, results = train(args.config)
            
            # Save the model
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, "weights.pt"))
            
            # Save model metadata for registry compatibility
            full_config = ConfigParser.load(args.config)
            model_metadata = {
                "name": full_config["model"]["name"],
                "model_config": full_config["model"]["config"],
                "scales": results.get("meta", {}).get("scales", {})
            }
            ConfigParser.save(model_metadata, os.path.join(args.output, "config.json"))
            
            print(f"[Rippl] Training successful. Final Loss: {results['loss']:.4e}")
            print(f"[Rippl] Model saved to {args.output}/")

        elif args.command == "simulate":
            print(f"[Rippl] Running simulation from {args.config}...")
            result = simulate(args.config)
            print(f"[Rippl] Simulation complete. Output tensor shape: {result.shape}")

        elif args.command == "export":
            print(f"[Rippl] Exporting model from {args.model_dir} to {args.format} format...")
            model = load_model(args.model_dir)
            
            # Load metadata from training config
            config = ConfigParser.load(args.config)
            export_model(model, args.model_dir, format=args.format, metadata=config)
            
            print(f"[Rippl] Export successful. Artifacts located in {args.model_dir}/")

    except Exception as e:
        print(f"[Rippl ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
