# -*- coding: utf-8 -*-
"""
B_PASS_simulation_modeling_cleaned.py

This script loads a Marmousi elastic model, adds water and sediment layers,
simulates the effect of CO2 injection, runs seismic surveys, and visualizes the results.
"""

# =============================================================================
# 1. Initialization and Setup
# =============================================================================
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import scipy.ndimage
import deepwave
import matplotlib.gridspec as gridspec


def setup_device_and_memory():
    """
    Checks for CUDA availability, sets the device, and clears GPU memory cache.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ Using CUDA device for computations.")
        # Clear GPU memory cache for a clean start
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared.")
    else:
        device = torch.device('cpu')
        print("⚠️ CUDA not available, using CPU instead. Computations may be slower.")

    print(f"Selected device: {device}\n")
    return device

# =============================================================================
# 2. Function Definitions
# =============================================================================
def load_and_flip_tensor(filepath):
    """
    Loads a .npy file and transposes its spatial dimensions to (depth, distance).
    Assumes input shape is (components, nx, ny) and flips to (components, ny, nx).

    Args:
        filepath (str): The full path to the .npy file.

    Returns:
        numpy.ndarray: The loaded data in (components, depth, distance) format,
                       or None if the file is not found or an error occurs.
    """
    print(f"Attempting to load file from: {filepath}")

    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at '{filepath}'")
        print("Please ensure the .npy file is in the correct directory.")
        return None

    try:
        tensor = np.load(filepath)
        print(f"File loaded successfully! Original shape: {tensor.shape}")
        # Transpose from (components, distance, depth) to (components, depth, distance)
        tensor_flipped = tensor.transpose(0, 2, 1)
        print(f"✅ Tensor flipped to conventional shape (depth, distance): {tensor_flipped.shape}\n")
        return tensor_flipped
    except Exception as e:
        print(f"❌ An error occurred while loading or flipping the file: {e}")
        return None

def add_water_and_sediment_layers(tensor, dx):
    """
    Replaces the top of the model with a water layer and a soft sediment layer
    containing random velocity anomalies.

    Args:
        tensor (numpy.ndarray): The input elastic model tensor with shape (3, depth, distance).
        dx (float): The grid spacing in meters.

    Returns:
        numpy.ndarray: The modified tensor with the added layers, or None if input is invalid.
    """
    if tensor is None:
        return None

    print("Adding shallow water and soft sediment layers...")
    # Create a copy to avoid modifying the original tensor, and apply a small velocity increase
    modified_tensor = tensor.copy() * 1.3

    # --- 1. Define Water Layer (0m to 350m) ---
    water_vp = 1500.0
    water_vs = 0.0
    water_rho = 1000.0
    water_end_idx = int(350 / dx)

    print(f"  -> Creating water layer down to {350}m (index {water_end_idx}).")
    modified_tensor[0, :water_end_idx, :] = water_vp
    modified_tensor[1, :water_end_idx, :] = water_vs
    modified_tensor[2, :water_end_idx, :] = water_rho

    # --- 2. Define High Attenuation Sediment Layer (350m to 450m) ---
    sediment_start_idx = int(350 / dx)
    sediment_end_idx = int(450 / dx)

    sediment_vp = 850.0
    sediment_vs = 280.0
    sediment_rho = 1800.0

    print(f"  -> Creating base sediment layer from {350}m to {450}m (indices {sediment_start_idx} to {sediment_end_idx}).")
    # This loop applies a slight gradient and random noise to the base sediment values
    for z_idx in range(sediment_start_idx, sediment_end_idx):
        # A small depth-dependent gradient
        gradient_factor = (z_idx - sediment_end_idx) / (sediment_start_idx - sediment_end_idx) # Normalizes from 0 to 1

        modified_tensor[0, z_idx, :] = sediment_vp * np.random.uniform(0.9, 1.1) + gradient_factor * 100
        modified_tensor[1, z_idx, :] = sediment_vs * np.random.uniform(0.9, 1.1) + gradient_factor * 50
        modified_tensor[2, z_idx, :] = sediment_rho * np.random.uniform(0.9, 1.1) + gradient_factor * 100

    # --- 3. Add Random Velocity Anomaly Chunks ---
    num_chunks = np.random.randint(1000, 2000)
    ny, nx = modified_tensor.shape[1:]

    print(f"  -> Adding {num_chunks} random velocity anomalies to the sediment layer...")
    for _ in range(num_chunks):
        # Define random chunk dimensions
        height = np.random.randint(2, 8)
        width = np.random.randint(5, 25)

        # Define random location within the sediment layer
        top = np.random.randint(sediment_start_idx, sediment_end_idx - height)
        left = np.random.randint(0, nx - width)

        # Apply a multiplicative random factor to the chunk
        modified_tensor[0, top:top+height, left:left+width] *= np.random.uniform(0.85, 1.15)  # Vp
        modified_tensor[1, top:top+height, left:left+width] *= np.random.uniform(0.85, 1.15)  # Vs
        modified_tensor[2, top:top+height, left:left+width] *= np.random.uniform(0.85, 1.15)  # Rho

    print("✅ Shallow layers with anomalies added successfully.\n")
    return modified_tensor

def plot_elastic_model(tensor, dx=4.0, title_prefix=""):
    """
    Plots the Vp, Vs, and Density components of an elastic model tensor.

    Args:
        tensor (numpy.ndarray): The elastic model tensor of shape (3, depth, distance).
        dx (float): The grid spacing in meters.
        title_prefix (str): A prefix for the main plot title.
    """
    if tensor is None or not (isinstance(tensor, np.ndarray) and tensor.ndim == 3 and tensor.shape[0] == 3):
        print("❌ Cannot plot the model. Input must be a NumPy array with shape (3, depth, distance).")
        return

    vp, vs, rho = tensor[0, :, :], tensor[1, :, :], tensor[2, :, :]
    ny, nx = vp.shape # ny is depth, nx is distance

    components = [vp, vs, rho]
    titles = ['P-Wave Velocity (Vp)', 'S-Wave Velocity (Vs)', 'Density (Rho)']
    cmaps = ['jet', 'jet', 'viridis']
    units = ['m/s', 'm/s', 'kg/m³']

    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    # Define image extent: [left, right, bottom, top] for correct axis labels
    extent = [0, nx * dx, ny * dx, 0]

    for i, ax in enumerate(axes):
        im = ax.imshow(components[i], cmap=cmaps[i], extent=extent, aspect='auto')
        ax.set_title(titles[i], fontsize=14, pad=10)
        ax.set_ylabel('Depth (m)', fontsize=12)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.03)
        cbar.set_label(units[i], fontsize=11)

    axes[-1].set_xlabel('Horizontal Distance (m)', fontsize=12)

    fig.suptitle(f'{title_prefix} Elastic Model', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to prevent title overlap
    plt.show()

def compute_elastic_params(vp, vs, rho):
    mu = rho * vs**2
    lamb = rho * (vp**2 - 2 * vs**2)
    buoyancy = 1.0 / rho
    return lamb.to(device), mu.to(device), buoyancy.to(device)


# =============================================================================
# 3. Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    # IMPORTANT: Please update this path to point to your .npy file.
    filename = 'siesmic model /Marmousi2_elastic/model.npy'
    dx = 4.0

    # --- Run Workflow ---
    device = setup_device_and_memory()

    # 1. Load and prepare the base model
    marmousi_tensor = load_and_flip_tensor(filename)

    # 2. Add custom top layers
    model_original_numpy = add_water_and_sediment_layers(marmousi_tensor, dx=dx)

    # 3. Plot the initial model
    if model_original_numpy is not None:
        plot_elastic_model(model_original_numpy, dx=dx, title_prefix="Initial Marmousi")
    else:
        print("❌ Execution stopped because the model could not be loaded or created.")
        exit() # Exit if model loading fails

    # =========================================================================
    # 4. CO2 Injection Modeling
    # =========================================================================
    print("\n--- Starting CO2 Injection Simulation ---")
    vp_original = torch.from_numpy(model_original_numpy[0]).to(device).float()
    vs_original = torch.from_numpy(model_original_numpy[1]).to(device).float()
    rho_original = torch.from_numpy(model_original_numpy[2]).to(device).float()
    print(f"✅ Loaded elastic model components to Torch: Vp {vp_original.shape}, Vs {vs_original.shape}, Rho {rho_original.shape}")

    # --- Define CO2 Injection Parameters ---
    target_depth_m = 1340.0
    layer_thickness_m = 160.0
    plume_center_x_m = 11530.0
    plume_width_m = 840.0
    velocity_reduction_factor = 0.75  # 25% reduction
    assumed_dz_m = dx
    print(f"Assuming dz = {assumed_dz_m:.1f}m for vertical sampling.")

    # --- Calculate Indices for the CO2 Plume ---
    y_center_idx = int(round(target_depth_m / assumed_dz_m))
    h_samples = int(round(layer_thickness_m / assumed_dz_m))
    if layer_thickness_m > 0 and h_samples == 0:
        h_samples = 1
    y_start_idx = max(0, y_center_idx - h_samples // 2)
    y_end_idx = min(vp_original.shape[0], y_start_idx + h_samples)
    x_plume_start_m_calc = plume_center_x_m - plume_width_m / 2
    x_plume_end_m_calc = plume_center_x_m + plume_width_m / 2
    x_start_idx = max(0, int(round(x_plume_start_m_calc / dx)))
    x_end_idx = min(vp_original.shape[1], int(round(x_plume_end_m_calc / dx)))
    print("Targeting CO2 plume:")
    print(f"  Vertical:   Depth {target_depth_m}m  => y_indices [{y_start_idx} - {y_end_idx-1}]")
    print(f"  Horizontal: Dist  {x_plume_start_m_calc:.1f}m - {x_plume_end_m_calc:.1f}m => x_indices [{x_start_idx} - {x_end_idx-1}]")

    # --- Create CO2-injected models for Vp, Vs, and rho ---
    vp_co2 = vp_original.clone()
    vs_co2 = vs_original.clone()
    rho_co2 = rho_original.clone()
    if y_start_idx < y_end_idx and x_start_idx < x_end_idx:
        sharp_mask = np.zeros(vp_original.shape, dtype=float)
        sharp_mask[y_start_idx:y_end_idx, x_start_idx:x_end_idx] = 1.0
        sigma_y = max(1.0, h_samples / 4.0)
        sigma_x = max(1.0, (x_end_idx - x_start_idx) / 6.0)
        print(f"Applying Gaussian smoothing with sigma_y={sigma_y:.1f}, sigma_x={sigma_x:.1f} samples.")
        smoothed_mask_np = scipy.ndimage.gaussian_filter(sharp_mask, sigma=(sigma_y, sigma_x))
        if smoothed_mask_np.max() > 0:
            smoothed_mask_np /= smoothed_mask_np.max()
        smoothed_mask = torch.from_numpy(smoothed_mask_np).to(device).type_as(vp_original)
        vp_multiplier = 1.0 - smoothed_mask * (1.0 - velocity_reduction_factor)
        vp_co2 = vp_original * vp_multiplier
        vs_increase_factor = 1.02
        vs_multiplier = 1.0 + smoothed_mask * (vs_increase_factor - 1.0)
        vs_co2 = vs_original * vs_multiplier
        density_reduction_factor = 0.95
        rho_multiplier = 1.0 - smoothed_mask * (1.0 - density_reduction_factor)
        rho_co2 = rho_original * rho_multiplier
        print(f"✅ Applied -25% to Vp, +2% to Vs, -5% to Rho in CO2 plume.")
        model_description = "Modified"
    else:
        print("⚠️ Warning: Calculated plume indices are invalid. Models not modified.")
        model_description = "Original"

    # --- Visualize Vp, Vs, and Density before and after CO2 injection ---
    fig, axes = plt.subplots(3, 3, figsize=(20, 18), sharex=True, sharey=True)
    plt.suptitle(f'Elastic Model Analysis: CO2 Injection Effect ({model_description})', fontsize=20, y=1.0)

    params = [('Vp', vp_original, vp_co2), ('Vs', vs_original, vs_co2), ('Density', rho_original, rho_co2)]
    units = ['(m/s)', '(m/s)', '(kg/m³)']
    v_extent = [0, vp_original.shape[1] * dx, vp_original.shape[0] * assumed_dz_m, 0]

    for i, (name, original, modified) in enumerate(params):
        vmax = torch.max(torch.cat([original.flatten(), modified.flatten()]))
        vmin = torch.min(torch.cat([original.flatten(), modified.flatten()]))
        unit_label = units[i]

        im0 = axes[i, 0].imshow(original.cpu().numpy(), aspect='auto', cmap='jet', extent=v_extent, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Original {name}')
        axes[i, 0].set_ylabel('Depth (m)')
        fig.colorbar(im0, ax=axes[i, 0], label=f'{name} {unit_label}', fraction=0.046, pad=0.04)

        im1 = axes[i, 1].imshow(modified.cpu().numpy(), aspect='auto', cmap='jet', extent=v_extent, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Modified {name}')
        fig.colorbar(im1, ax=axes[i, 1], label=f'{name} {unit_label}', fraction=0.046, pad=0.04)

        diff = modified - original
        max_abs_diff = torch.abs(diff).max()
        im2 = axes[i, 2].imshow(diff.cpu().numpy(), aspect='auto', cmap='coolwarm_r', extent=v_extent,
                               vmin=-max_abs_diff, vmax=max_abs_diff)
        axes[i, 2].set_title(f'{name} Change')
        axes[i, 2].set_xlabel('Distance (m)')
        fig.colorbar(im2, ax=axes[i, 2], label=f'Difference {unit_label}', fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    rect_props = {'linewidth': 1.5, 'edgecolor': 'orange', 'facecolor': 'none', 'linestyle': '--', 'label': 'CO2 Plume Boundary'}
    for ax_row in axes:
        for ax in ax_row:
            ax.add_patch(plt.Rectangle(
                (x_start_idx * dx, y_start_idx * assumed_dz_m),
                (x_end_idx - x_start_idx) * dx,
                (y_end_idx - y_start_idx) * assumed_dz_m,
                **rect_props
            ))
    output_dir = 'Elastic_Model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_filename = os.path.join(output_dir, f'Elastic_Model_{model_description}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Plot saved to: {plot_filename}")

    # =========================================================================
    # 5. ELASTIC SIMULATION AND PLOTTING
    # =========================================================================
    print("\n--- Starting ELASTIC Wave Propagation Simulation ---")
    lamb_orig, mu_orig, buoy_orig = compute_elastic_params(vp_original, vs_original, rho_original)
    lamb_co2, mu_co2, buoy_co2 = compute_elastic_params(vp_co2, vs_co2, rho_co2)

    conditions=['Marine Source','Surface Source','Borehole Source']
    freq = 25
    nt = 750
    dt = 0.004
    peak_time = 1.5 / freq
    print("Setting up VSP comparison...")

    source_depths_m = [100, 335, 820]
    source_depths_grid = [int(d / dx) for d in source_depths_m]
    n_shots = 1
    n_receivers = 100
    receiver_start_depth = int(0 / dx)
    receiver_end_depth = vp_original.shape[0] - 0
    source_x_location = int(8800 / dx)
    source_amplitudes_base = deepwave.wavelets.ricker(freq, nt, dt, peak_time)
    vsp_results = []

    for depth_m, depth_grid in zip(source_depths_m, source_depths_grid):
        print(f"\n--- Simulating for Source Depth: {depth_m}m (grid point {depth_grid}) ---")
        source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
        source_locations[..., 0] = source_x_location
        source_locations[..., 1] = depth_grid
        source_amplitude_scale = 1.0
        source_amplitudes_z = source_amplitudes_base.reshape(1, 1, -1).repeat(n_shots, 1, 1).to(device) * source_amplitude_scale
        source_amplitudes_x = torch.zeros_like(source_amplitudes_z)
        receiver_locations = torch.zeros(n_shots, n_receivers, 2, dtype=torch.long, device=device)
        receiver_z_fixed = int(360 / dx)
        receiver_x_start = int(10 / dx)
        receiver_x_end = vp_original.shape[1] - 10
        receiver_x_coords = torch.linspace(receiver_x_start, receiver_x_end, n_receivers)
        receiver_locations[0, :, 0] = receiver_x_coords.round().long()
        receiver_locations[0, :, 1] = receiver_z_fixed
        prop_kwargs = {
            'grid_spacing': dx, 'dt': dt, 'pml_freq': freq,
            'source_amplitudes_y': source_amplitudes_z,
            'source_amplitudes_x': source_amplitudes_x,
            'source_locations_y': source_locations,
            'source_locations_x': source_locations,
            'receiver_locations_y': receiver_locations,
        }
        print(f"  -> Running simulation on Original Model...")
        out_original = deepwave.elastic(lamb_orig.T, mu_orig.T, buoy_orig.T, **prop_kwargs)
        print(f"  -> Running simulation on CO2-Injected Model...")
        out_co2 = deepwave.elastic(lamb_co2.T, mu_co2.T, buoy_co2.T, **prop_kwargs)
        vsp_difference = out_co2[1][0] - out_original[1][0]
        vsp_results.append({
            'depth_m': depth_m,
            'difference_data': vsp_difference,
            'original_data': out_original[1][0],
            'co2_data': out_co2[1][0]
        })
        print(f"✅ Simulation for {depth_m}m complete.")

    print("\n📊 Computing energy per trace for each source depth (only x ∈ [11300, 12800])...")
    energy_summary = []
    x_min, x_max = 11300, 12800
    receiver_xs = receiver_locations[0, :, 0].cpu().numpy() * dx
    mask = (receiver_xs >= x_min) & (receiver_xs <= x_max)
    receiver_indices_in_range = np.where(mask)[0]
    print(f"Selected {len(receiver_indices_in_range)} receivers in x ∈ [{x_min}, {x_max}] m")

    for result in vsp_results:
        diff_data = result['difference_data'].cpu().numpy()
        diff_selected = diff_data[receiver_indices_in_range, :]
        energy_per_trace = (diff_selected**2).sum(axis=1)
        mean_energy, std_energy = energy_per_trace.mean(), energy_per_trace.std()
        energy_summary.append({'depth_m': result['depth_m'], 'mean_energy': mean_energy, 'std_energy': std_energy})
        print(f"Depth {result['depth_m']} m --> Mean: {mean_energy:.2e}, SD: {std_energy:.2e}")
    energy_summary.sort(key=lambda x: x['depth_m'])
    means = [r['mean_energy'] for r in energy_summary]
    stds = [r['std_energy'] for r in energy_summary]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(conditions, means, color='gray', edgecolor='black', alpha=0.9)
    for i, (bar, mean, sd) in enumerate(zip(bars, means, stds)):
        bar_center = bar.get_x() + bar.get_width() / 2
        ax.vlines(bar_center, mean - sd, mean + sd, color='black', linewidth=1.5)
        ax.plot([bar_center], [mean + sd], 'k_', markersize=8)
        ax.plot([bar_center], [mean - sd], 'k_', markersize=8)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_ylabel("Mean Energy per Trace (Amplitude²)", fontsize=12)
    ax.set_title("Shot Gather Energy vs Source Depth\n(at CO2 Plume )", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("energy_by_source_depth_variance_windowed.png", dpi=600, bbox_inches='tight')
    plt.show()

    # =========================================================================
    # 6. SCALAR (P-WAVE) SIMULATION AND PLOTTING
    # =========================================================================
    print("\n--- Starting SCALAR (P-Wave) Wave Propagation Simulation ---")
    vsp_results = []
    shot_gathers_original = [] # To store original shot gathers for final plot
    source_amplitudes = (deepwave.wavelets.ricker(freq, nt, dt, peak_time).reshape(1, 1, -1).to(device))

    for depth_m, depth_grid in zip(source_depths_m, source_depths_grid):
        print(f"\n--- Simulating for Source Depth: {depth_m}m (grid point {depth_grid}) ---")
        source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
        source_locations[..., 0] = source_x_location
        source_locations[..., 1] = depth_grid

        print(f"  -> Running simulation on Original Model...")
        out_original = deepwave.scalar(vp_original.T, dx, dt, source_amplitudes=source_amplitudes, source_locations=source_locations, receiver_locations=receiver_locations, accuracy=8, pml_freq=freq)[0]
        shot_gathers_original.append(out_original[0].detach().cpu()) # Save for final plot

        print(f"  -> Running simulation on CO2-Injected Model...")
        out_co2 = deepwave.scalar(vp_co2.T, dx, dt, source_amplitudes=source_amplitudes, source_locations=source_locations, receiver_locations=receiver_locations, accuracy=8, pml_freq=freq)[0]

        vsp_difference = out_co2[0] - out_original[0]
        vsp_results.append({'depth_m': depth_m, 'difference_data': vsp_difference.detach()})
        print(f"✅ Simulation for {depth_m}m complete.")

    # --- PLOT 1: SURVEY GEOMETRY ---
    print("\nPlotting survey geometry...")
    fig_model = plt.figure(figsize=(18, 6))
    ax_model = fig_model.add_subplot(111)
    model_depth_m, model_width_m = vp_co2.shape[0] * dx, vp_co2.shape[1] * dx
    v_extent = [0, model_width_m, model_depth_m, 0]
    im_model = ax_model.imshow(vp_co2.cpu().numpy(), aspect='auto', cmap='jet', extent=v_extent)
    fig_model.colorbar(im_model, ax=ax_model, label='Velocity (m/s)', shrink=0.8)
    receiver_plot_x = receiver_locations[0, :, 0].cpu().numpy() * dx
    receiver_plot_y = receiver_locations[0, :, 1].cpu().numpy() * dx
    ax_model.scatter(receiver_plot_x, receiver_plot_y, c='black', marker='v', s=80, label='Borehole Receivers', zorder=10)
    colors = ['cyan', 'lime', 'magenta']
    for i, (depth_m, depth_grid) in enumerate(zip(source_depths_m, source_depths_grid)):
        ax_model.scatter(source_x_location * dx, depth_grid * dx, c=colors[i], marker='*', s=250, label=f'{conditions[i]}', edgecolors='black', zorder=14)
    plume_rect_x = x_start_idx * dx
    plume_rect_y = y_start_idx * assumed_dz_m
    plume_rect_w = (x_end_idx - x_start_idx) * dx
    plume_rect_h = (y_end_idx - y_start_idx) * assumed_dz_m
    ax_model.add_patch(plt.Rectangle((plume_rect_x, plume_rect_y), plume_rect_w, plume_rect_h, **rect_props))
    ax_model.legend(loc='lower right')
    ax_model.set_title('Survey Geometry on CO2-Injected Vp Model', fontsize=20)
    ax_model.set_xlabel('Distance (m)', fontsize=22)
    ax_model.set_ylabel('Depth (m)', fontsize=22)
    ax_model.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig('survey_geometry.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- PLOT 2: SHOT GATHER DIFFERENCE ---
    print("\n📈 Plotting improved shot gather difference results...")
    fig_vsp, axes = plt.subplots(len(vsp_results), 1, figsize=(12, 16), sharex=True)
    if len(vsp_results) == 1: axes = [axes]
    fig_vsp.suptitle('Shot Gather Time-Lapse Difference (CO₂ - Baseline)', fontsize=24, y=0.92)
    for i, result in enumerate(vsp_results):
        ax = axes[i]
        im = ax.imshow(result['difference_data'].cpu().numpy().T, cmap='seismic', aspect='auto', extent=[receiver_x_start * dx, receiver_x_end * dx, nt * dt, 0])
        ax.set_title(f"{conditions[i]} ", fontsize=20)
        ax.set_ylabel('Time (s)', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        cbar = fig_vsp.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
        cbar.set_label('Amplitude Difference', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
    axes[-1].set_xlabel('Offset (m)', fontsize=18)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.94])
    plt.savefig('ShotGather_Difference_VerticalPanels.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- PLOT 3: COMBINED FIGURE ---
    print("\n📊 Creating aligned combined figure...")
    fig = plt.figure(figsize=(16, 19))
    gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 0.04], height_ratios=[1.2, 1, 1, 1], hspace=0.3, wspace=0.2)
    ax_model = fig.add_subplot(gs[0, 0:2])
    im_model = ax_model.imshow(vp_co2.cpu().numpy(), aspect='auto', cmap='jet', extent=v_extent)
    ax_model.scatter(receiver_plot_x, receiver_plot_y, c='black', marker='v', s=80, label='Receivers')
    for i, depth_grid in enumerate(source_depths_grid):
        ax_model.scatter(source_x_location * dx, depth_grid * dx, c=colors[i], marker='*', s=250, edgecolors='black', label=conditions[i])
    ax_model.add_patch(plt.Rectangle((plume_rect_x, plume_rect_y), plume_rect_w, plume_rect_h, linewidth=1.5, edgecolor='orange', facecolor='none', linestyle='--', label='CO₂ Plume'))
    ax_model.set_title("Survey Geometry and Source Location", fontsize=20)
    ax_model.set_ylabel("Depth (m)", fontsize=16)
    ax_model.legend(fontsize=12)
    cax_model = fig.add_subplot(gs[0, 2])
    cbar_model = fig.colorbar(im_model, cax=cax_model)
    cbar_model.set_label("Velocity (m/s)", fontsize=14)
    cbar_model.ax.tick_params(labelsize=12)

    for i in range(3):
        data_left = shot_gathers_original[i].T.numpy()
        vmin_sg, vmax_sg = np.percentile(data_left, [5, 95])
        ax_left = fig.add_subplot(gs[i+1, 0])
        ax_left.imshow(data_left, cmap='seismic', aspect='auto', extent=[receiver_x_coords.min().item()*dx, receiver_x_coords.max().item()*dx, nt*dt, 0], vmin=vmin_sg, vmax=vmax_sg)
        ax_left.set_title(f"{conditions[i]} - Shot Gather", fontsize=16)
        ax_left.set_ylabel("Time (s)", fontsize=14)
        if i == 2: ax_left.set_xlabel("Offset (m)", fontsize=14)
        ax_left.tick_params(labelsize=12)

        data_diff = vsp_results[i]['difference_data'].cpu().numpy().T
        vmax_diff = np.max(np.abs(data_diff)) * 0.7 # Clip for viz
        ax_right = fig.add_subplot(gs[i+1, 1])
        im_diff = ax_right.imshow(data_diff, cmap='seismic', aspect='auto', extent=[receiver_x_coords.min().item()*dx, receiver_x_coords.max().item()*dx, nt*dt, 0], vmin=-vmax_diff, vmax=vmax_diff)
        ax_right.set_title(f"{conditions[i]} - Wavefield Difference", fontsize=16)
        if i == 2: ax_right.set_xlabel("Offset (m)", fontsize=14)
        ax_right.tick_params(labelsize=12)

        cax_diff = fig.add_subplot(gs[i+1, 2])
        cbar_diff = fig.colorbar(im_diff, cax=cax_diff)
        cbar_diff.set_label("Amplitude Difference", fontsize=13)
        cbar_diff.ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Aligned_Model_ShotGather_WavefieldDiff.png", dpi=300, bbox_inches='tight')
    plt.show()
