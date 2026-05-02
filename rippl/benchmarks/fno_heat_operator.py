import torch
import math
from rippl.nn.fno import FNO
from rippl.core.operator_experiment import OperatorDataset, OperatorExperiment

def generate_heat_data(n_samples=1000, n_points=64, alpha=0.01, t=1.0):
    # Domain [0, 1]
    x = torch.linspace(0, 1, n_points)
    
    inputs = []
    outputs = []
    
    for _ in range(n_samples):
        # Generate random IC: a(x) = sum_{k=1}^5 c_k sin(k pi x)
        coeffs = torch.randn(5)
        a_x = torch.zeros(n_points)
        u_xt = torch.zeros(n_points)
        
        for k in range(1, 6):
            ck = coeffs[k-1]
            mode = torch.sin(k * math.pi * x)
            a_x += ck * mode
            u_xt += ck * math.exp(-alpha * (k * math.pi)**2 * t) * mode
            
        inputs.append(a_x.view(-1, 1)) # (N, 1)
        outputs.append(u_xt.view(-1, 1))
        
    return torch.stack(inputs), torch.stack(outputs)

def main():
    print("Generating Heat Equation Operator Data...")
    train_in, train_out = generate_heat_data(1000)
    test_in, test_out = generate_heat_data(200)
    
    train_dataset = OperatorDataset(train_in, train_out)
    
    model = FNO(n_modes=12, width=32, input_dim=1, output_dim=1)
    exp = OperatorExperiment(model, train_dataset)
    
    print("Training FNO Operator mapping a(x) -> u(x, t=1.0)...")
    exp.train(epochs=10, lr=1e-3) # Low epochs for bench script
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(test_in)
        l2_err = torch.norm(pred - test_out) / torch.norm(test_out)
        print(f"Test Relative L2 Error: {l2_err.item():.4f}")

if __name__ == "__main__":
    main()
