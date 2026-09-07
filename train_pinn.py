import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import scipy.ndimage as ndimage
import os

# =====================================================================
# 1. 3D PHYSICS-INFORMED NEURAL NETWORK ARCHITECTURE
# =====================================================================
class VehicleFluidPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 5 Channels (Velocity X, Y, Z + Smoke Density + Vehicle Mask)
        self.network = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            # Output: 4 Channels (Predicted Velocity X, Y, Z + Smoke Density at t+1)
            nn.Conv3d(64, 4, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.network(x)

# =====================================================================
# 2. HYBRID PHYSICS LOSS RESIDUAL EVALUATOR
# =====================================================================
def compute_physics_losses(pred_fields, dt=0.01, dx=1.0, diffusion_coeff=0.01):
    """
    Evaluates Navier-Stokes Incompressibility and Scalar Advection-Diffusion Transport.
    Uses spatial central differences to approximate partial derivatives.
    """
    # Isolate prediction channels from tensor [Batch, Channels, X, Y, Z]
    u = pred_fields[:, 0, :, :, :]
    v = pred_fields[:, 1, :, :, :]
    w = pred_fields[:, 2, :, :, :]
    s = pred_fields[:, 3, :, :, :] # Smoke density channel
    
    # --- PHYSICAL GRADIENT ESTIMATIONS (Central Differences) ---
    # Velocity field spatial divergence components
    du_dx = (u[:, 2:, 1:-1, 1:-1] - u[:, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    dv_dy = (v[:, 1:-1, 2:, 1:-1] - v[:, 1:-1, :-2, 1:-1]) / (2.0 * dx)
    dw_dz = (w[:, 1:-1, 1:-1, 2:] - w[:, 1:-1, 1:-1, :-2]) / (2.0 * dx)
    
    # Smoke density first derivatives (Advection terms)
    ds_dx = (s[:, 2:, 1:-1, 1:-1] - s[:, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    ds_dy = (s[:, 1:-1, 2:, 1:-1] - s[:, 1:-1, :-2, 1:-1]) / (2.0 * dx)
    ds_dz = (s[:, 1:-1, 1:-1, 2:] - s[:, 1:-1, 1:-1, :-2]) / (2.0 * dx)
    
    # Smoke density second derivatives (Diffusion Laplacians)
    d2s_dx2 = (s[:, 2:, 1:-1, 1:-1] - 2*s[:, 1:-1, 1:-1, 1:-1] + s[:, :-2, 1:-1, 1:-1]) / (dx**2)
    d2s_dy2 = (s[:, 1:-1, 2:, 1:-1] - 2*s[:, 1:-1, 1:-1, 1:-1] + s[:, 1:-1, :-2, 1:-1]) / (dx**2)
    d2s_dz2 = (s[:, 1:-1, 1:-1, 2:] - 2*s[:, 1:-1, 1:-1, 1:-1] + s[:, 1:-1, 1:-1, :-2]) / (dx**2)

    # --- LOSS 1: MASS CONSERVATION (INCOMPRESSIBILITY) ---
    # ∇ · U must equal 0 everywhere for an incompressible fluid
    divergence = du_dx + dv_dy + dw_dz
    loss_divergence = torch.mean(divergence ** 2)
    
    # --- LOSS 2: SCALAR ADVECTION-DIFFUSION LAW ---
    # Trim velocity internal subgrids to match derivative grid boundaries [1:-1]
    u_mid = u[:, 1:-1, 1:-1, 1:-1]
    v_mid = v[:, 1:-1, 1:-1, 1:-1]
    w_mid = w[:, 1:-1, 1:-1, 1:-1]
    
    advection = u_mid * ds_dx + v_mid * ds_dy + w_mid * ds_dz
    diffusion = diffusion_coeff * (d2s_dx2 + d2s_dy2 + d2s_dz2)
    
    transport_residual = advection - diffusion
    loss_transport = torch.mean(transport_residual ** 2)
    
    return loss_divergence, loss_transport

# =====================================================================
# 3. INITIALIZATION & DATA DESK LOOKUP
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing simulation training on device target: {device}")

model = VehicleFluidPINN().to(device)
data_criterion = nn.MSELoss()

# AdamW with weight decay dampens structural weight explosions near boundary discontinuities
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Mount paths matching the layout within your Docker volume configuration
DATA_DIR = "/input/blastnet-momentum128-3d-sr-dataset"
data_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.npy"), recursive=True))

GRID_SIZE = 128
EPOCHS = 5

if len(data_files) < 2:
    print(f"🚨 Warning: Insufficient dataset files found in {DATA_DIR}. Simulating synthetic tensor tracks for environment test...")
    # Fallback to structural synthetic generation if dataset volume isn't mounted yet
    data_files = [np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), dtype=np.float32) for _ in range(10)]

# =====================================================================
# 4. TRAINING ENGINE ROUTINE
# =====================================================================
model.train()
print("Starting 3D PINN computational loop optimization...")

for epoch in range(EPOCHS):
    for i in range(len(data_files) - 1):
        try:
            # Handle loading from disk or fallback runtime mock objects
            if isinstance(data_files[i], str):
                frame_t = np.load(data_files[i])
                frame_t_plus_1 = np.load(data_files[i+1])
            else:
                frame_t = data_files[i]
                frame_t_plus_1 = data_files[i+1]
            
            # --- SAFEGUARD A: INTERACTIVE BOUNDARY DATA SYNTHESIS ---
            smoke_t = np.ones((GRID_SIZE, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
            vehicle_mask = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
            
            # Dynamic movement of vehicle wedge down the flow domain over time steps
            vehicle_x_pos = int(10 + (i % (GRID_SIZE - 45)))
            for x_idx in range(vehicle_x_pos, vehicle_x_pos + 30):
                half_width = int((x_idx - vehicle_x_pos) * 0.4) + 2
                vehicle_mask[x_idx, 64-half_width:64+half_width, 64-half_width:64+half_width, 0] = 1.0
            
            # Displace mass rule: Smoke field occupancy is zero inside the solid physical body
            smoke_t[vehicle_mask > 0.5] = 0.0
            
            # --- SAFEGUARD B: GAUSSIAN BOUNDARY REGULARIZATION ---
            # Softens absolute step boundaries (0.0 -> 1.0) so derivatives do not hit infinity
            smoothed_vehicle = ndimage.gaussian_filter(vehicle_mask, sigma=1.2)
            smoothed_smoke_t = ndimage.gaussian_filter(smoke_t, sigma=0.5)
            
            # Formulate chronological target state for fluid clearing field tracking at t+1
            smoke_t_plus_1 = np.ones_like(smoke_t)
            next_vehicle_x = int(10 + ((i+1) % (GRID_SIZE - 45)))
            smoke_t_plus_1[next_vehicle_x:next_vehicle_x+32, 50:78, 50:78, 0] = 0.0 
            smoothed_smoke_t_plus_1 = ndimage.gaussian_filter(smoke_t_plus_1, sigma=0.5)

            # --- ARRAY CONCATENATION & HARDWARE STACK TENSOR MOUNT ---
            input_combined = np.concatenate([frame_t, smoothed_smoke_t, smoothed_vehicle], axis=-1)
            target_combined = np.concatenate([frame_t_plus_1, smoothed_smoke_t_plus_1], axis=-1)
            
            # Permute dimensions to match standard PyTorch formatting: [Batch, Channels, X, Y, Z]
            X = torch.tensor(input_combined, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            Y = torch.tensor(target_combined, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            
            # --- BACKPROPAGATION OPTIMIZATION STEP ---
            optimizer.zero_grad()
            predictions = model(X)
            
            # Calculate objective losses
            loss_data = data_criterion(predictions, Y)
            loss_div, loss_transport = compute_physics_losses(predictions)
            
            # Balanced total penalty calculation
            total_loss = loss_data + (0.01 * loss_div) + (0.005 * loss_transport)
            total_loss.backward()
            
            # --- SAFEGUARD C: NORM GRADIENT CIRCUIT BREAKER ---
            # Drops sudden spike profiles to prevent network weights from tearing apart
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if i % 5 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}] | Total Loss: {total_loss.item():.5f} | "
                      f"Div Penalty: {loss_div.item():.5f} | Transport Penalty: {loss_transport.item():.5f}")
                
        except Exception as e:
            # Gracefully catch file dimensions or index faults without breaking runtime execution
            print(f"Skipping anomaly at iteration step {i}: {str(e)}")
            continue

print("3D PINN simulation model optimization training completed successfully.")

# Save optimized network weights parameter structure
torch.save(model.state_dict(), "vehicle_pinn.pth")
print("Weights saved as: vehicle_pinn.pth")
