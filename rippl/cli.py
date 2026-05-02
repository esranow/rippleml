import argparse
import sys
import os
import yaml
import torch
from rippl.api import train, simulate
from rippl.nn.registry import load_model
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
            
            with open(args.config, 'r') as f:
                raw_cfg = yaml.safe_load(f)
                
            from rippl.config.models import DomainConfig, EquationConfig, OperatorConfig
            
            # 2. Enforce strict Pydantic validation before PyTorch logic
            if "domain" in raw_cfg:
                DomainConfig(**raw_cfg["domain"])
            if "equation" in raw_cfg:
                EquationConfig(**raw_cfg["equation"])
            if "operators" in raw_cfg:
                for op in raw_cfg["operators"]:
                    OperatorConfig(**op)
            
            # 3 & 4. Handle Neural Network IoC
            model_cfg = raw_cfg.get("model", {})
            if "script" in model_cfg and "class_name" in model_cfg:
                import importlib.util
                script_path = model_cfg["script"]
                class_name = model_cfg["class_name"]
                
                spec = importlib.util.spec_from_file_location("custom_model", script_path)
                custom_module = importlib.util.module_from_spec(spec)
                sys.modules["custom_model"] = custom_module
                spec.loader.exec_module(custom_module)
                
                model_cls = getattr(custom_module, class_name)
                kwargs = {k: v for k, v in model_cfg.items() if k not in ["script", "class_name", "type"]}
                net = model_cls(**kwargs)
            else:
                import rippl.nn as rnn
                model_type = model_cfg.get("type", "MLP")
                model_cls = getattr(rnn, model_type)
                kwargs = {k: v for k, v in model_cfg.items() if k != "type"}
                net = model_cls(**kwargs)
                
            from rippl.core.engine import Engine
            engine = Engine(net)
            
            # 5. Compile, fit, and save
            engine.compile()
            
            epochs = raw_cfg.get("training", {}).get("epochs", 10)
            engine.fit(epochs=epochs, **raw_cfg.get("training", {}))
            
            engine.validate()
            
            engine.save("output.rpx")
            
            print(f"[Rippl] Training successful. Model saved to output.rpx")

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
