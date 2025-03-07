import torch

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the TorchScript model
model_path = "/media/citi-ai/matthew/uav-human-detection/results/experiment_20250219/exp_1_yolop2s_VisDrone_SGD_lr0.01_frzNone_coslrTrue/weights/best.torchscript"
model = torch.jit.load(model_path, map_location=device)

# Set to evaluation mode
model.eval()
model.to(device)

print("TorchScript model loaded successfully!")

# Display the state_dict (model parameters)
print("Model State Dict:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Display the weights of the model
print("Model Weights:")
for name, param in model.named_parameters():
    print(f"Layer: {name}")
    print(f"Weights: {param.data}")  # Print the weights for each layer
    print(f"Shape: {param.shape}")
    print("-" * 50)  # Just to separate each layer's weights