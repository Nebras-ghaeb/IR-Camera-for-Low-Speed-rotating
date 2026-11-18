import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.cm as cm
import seaborn as sns

# Load the data from CSV
#data = np.loadtxt('./projections/projection_35.csv', delimiter=',', comments='#')
data = np.loadtxt('./projections/projection_02.csv', delimiter=',',comments='#')

print(f"Data shape: {data.shape}")
print(f"Temperature range: [{data.min():.2f}, {data.max():.2f}]")
print(f"Mean temperature: {data.mean():.2f}")

# Check if data is uniform (all values the same)
is_uniform = np.all(data == data[0, 0])

if is_uniform:
    print("\n⚠️ WARNING: All temperature values are identical!")
    print("The data shows uniform temperature of 200.0 K across the entire cross-section.")
    print("This suggests either thermal equilibrium, perfect insulation, or initial conditions.\n")

# Create visualization with multiple plots
fig = plt.figure(figsize=(16, 12))

# 1. Raw Temperature Distribution
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(data, cmap='hot', aspect='equal')
plt.colorbar(im1, ax=ax1, label='Temperature (K)')
ax1.set_title('Raw Temperature Distribution')
ax1.set_xlabel('X Pixel')
ax1.set_ylabel('Y Pixel')

# 2. Temperature Distribution with Isotherms (contours)
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(data, cmap='hot', aspect='equal')

# Handle uniform data case
if is_uniform:
    # For uniform data, use a single contour level
    contour_levels = [data[0, 0]]
    contours = ax2.contour(data, levels=contour_levels, colors='white', linewidths=2, alpha=0.7)
    ax2.text(data.shape[1]/2, data.shape[0]/2, f'Uniform: {data[0,0]:.1f} K', 
             color='white', fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))
else:
    # For non-uniform data, create multiple contour levels
    contour_levels = np.linspace(data.min(), data.max(), 15)
    contours = ax2.contour(data, levels=contour_levels, colors='white', linewidths=0.5, alpha=0.7)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='%.1f K')

plt.colorbar(im2, ax=ax2, label='Temperature (K)')
ax2.set_title('Temperature with Isotherms')
ax2.set_xlabel('X Pixel')
ax2.set_ylabel('Y Pixel')

# 3. Enhanced Isotherm Visualization
ax3 = plt.subplot(2, 3, 3)
# Use a smoother colormap
im3 = ax3.imshow(data, cmap='plasma', aspect='equal', interpolation='gaussian')

if is_uniform:
    # For uniform data, show a single contour
    highlight_levels = [data[0, 0]]
    contours_highlight = ax3.contour(data, levels=highlight_levels, 
                                     colors='cyan', linewidths=3, alpha=0.8)
    ax3.text(data.shape[1]/2, data.shape[0]/2, 'UNIFORM\nTEMPERATURE', 
             color='white', fontsize=14, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.7))
else:
    # For non-uniform data
    highlight_levels = np.linspace(data.min(), data.max(), 8)
    contours_highlight = ax3.contour(data, levels=highlight_levels, 
                                     colors='cyan', linewidths=1.5, alpha=0.8)
    ax3.clabel(contours_highlight, inline=True, fontsize=9, fmt='%.1f K')

plt.colorbar(im3, ax=ax3, label='Temperature (K)')
ax3.set_title('Enhanced Isotherm Visualization')
ax3.set_xlabel('X Pixel')
ax3.set_ylabel('Y Pixel')

# 4. 3D Surface Plot
ax4 = plt.subplot(2, 3, 4, projection='3d')
X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
surf = ax4.plot_surface(X, Y, data, cmap='viridis', 
                       alpha=0.8, linewidth=0, antialiased=True)
ax4.set_xlabel('X Pixel')
ax4.set_ylabel('Y Pixel')
ax4.set_zlabel('Temperature (K)')
ax4.set_title('3D Temperature Surface')
fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=5, label='Temperature (K)')

# 5. Temperature Profile Analysis
ax5 = plt.subplot(2, 3, 5)
# Extract horizontal and vertical profiles
horizontal_profile = data[data.shape[0]//2, :]
vertical_profile = data[:, data.shape[1]//2]

ax5.plot(horizontal_profile, label=f'Horizontal Profile (Row {data.shape[0]//2})', 
         color='red', linewidth=2)
ax5.plot(vertical_profile, label=f'Vertical Profile (Column {data.shape[1]//2})', 
         color='blue', linewidth=2)
ax5.set_xlabel('Pixel Position')
ax5.set_ylabel('Temperature (K)')
ax5.set_title('Temperature Profiles')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Statistical Analysis and Heat Map
ax6 = plt.subplot(2, 3, 6)
# Calculate temperature statistics
temp_mean = np.mean(data)
temp_std = np.std(data)
temp_min = np.min(data)
temp_max = np.max(data)

# Create a heat map showing deviations from mean
if is_uniform:
    # For uniform data, deviation is zero everywhere
    deviation = np.zeros_like(data)
    cmap = 'gray'
    vmin = vmax = 0
else:
    deviation = data - temp_mean
    cmap = 'RdBu_r'
    vmin = -temp_std
    vmax = temp_std

im6 = ax6.imshow(deviation, cmap=cmap, aspect='equal', 
                vmin=vmin, vmax=vmax)
plt.colorbar(im6, ax=ax6, label='Deviation from Mean (K)')
ax6.set_title(f'Temperature Deviation\nMean: {temp_mean:.2f}K, Std: {temp_std:.2f}K')
ax6.set_xlabel('X Pixel')
ax6.set_ylabel('Y Pixel')

plt.tight_layout()
plt.show()

# Additional Analysis
print("\n" + "="*60)
print("ADDITIONAL STATISTICAL ANALYSIS")
print("="*60)

# Calculate basic statistics
print(f"\nBasic Statistics:")
print(f"Minimum Temperature: {temp_min:.2f} K")
print(f"Maximum Temperature: {temp_max:.2f} K")
print(f"Temperature Range: {temp_max - temp_min:.2f} K")
print(f"Mean Temperature: {temp_mean:.2f} K")
print(f"Standard Deviation: {temp_std:.2f} K")

if temp_mean != 0:
    print(f"Coefficient of Variation: {(temp_std/temp_mean*100):.2f}%")
else:
    print(f"Coefficient of Variation: N/A (mean is zero)")

# Calculate temperature gradients (will be zero for uniform data)
gradient_x, gradient_y = np.gradient(data)
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

print(f"\nGradient Analysis:")
print(f"Max X Gradient: {np.abs(gradient_x).max():.4f} K/pixel")
print(f"Max Y Gradient: {np.abs(gradient_y).max():.4f} K/pixel")
print(f"Max Gradient Magnitude: {gradient_magnitude.max():.4f} K/pixel")

# Regional analysis
rows, cols = data.shape
center_region = data[rows//4:3*rows//4, cols//4:3*cols//4]
edge_region = np.concatenate([data[0, :], data[-1, :], data[:, 0], data[:, -1]])

print(f"\nRegional Analysis:")
print(f"Center Region Mean: {center_region.mean():.2f} K")
print(f"Center Region Std: {center_region.std():.2f} K")
print(f"Edge Region Mean: {edge_region.mean():.2f} K")
print(f"Edge Region Std: {edge_region.std():.2f} K")

# Special message for uniform data
if is_uniform:
    print(f"\n⚠️ ANALYSIS NOTE:")
    print(f"The temperature is perfectly uniform across the entire cross-section.")
    print(f"This could indicate:")
    print(f"  1. Thermal equilibrium has been reached")
    print(f"  2. Perfect insulation or no heat transfer")
    print(f"  3. Initial/boundary conditions")
    print(f"  4. Measurement under controlled conditions")
    print(f"\nRecommendations:")
    print(f"  1. Check other rotation angles (if available)")
    print(f"  2. Verify measurement setup")
    print(f"  3. Consider adding heat sources for analysis")
    print(f"  4. Examine time-series data for transient behavior")

# Create histogram of temperatures
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes2[0, 0].hist(data.flatten(), bins=50, edgecolor='black', alpha=0.7)
axes2[0, 0].axvline(temp_mean, color='red', linestyle='--', 
                    label=f'Mean: {temp_mean:.2f}K')
if not is_uniform:
    axes2[0, 0].axvline(temp_mean + temp_std, color='orange', linestyle=':', 
                        label=f'±1σ: {temp_std:.2f}K')
    axes2[0, 0].axvline(temp_mean - temp_std, color='orange', linestyle=':')
axes2[0, 0].set_xlabel('Temperature (K)')
axes2[0, 0].set_ylabel('Frequency')
if is_uniform:
    axes2[0, 0].set_title('Temperature Distribution (Uniform)')
else:
    axes2[0, 0].set_title('Temperature Distribution Histogram')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

# Gradient Magnitude
im_grad = axes2[0, 1].imshow(gradient_magnitude, cmap='coolwarm', aspect='equal')
cbar = plt.colorbar(im_grad, ax=axes2[0, 1], label='Gradient Magnitude (C/pixel)')

cbar.ax.tick_params(labelsize=14)
cbar.set_label('Gradient Magnitude (C/pixel)', size=16)


#axes2[0, 1].set_title('Temperature Gradient Magnitude')
axes2[0, 1].set_xlabel('X Pixel', fontsize = 16)
axes2[0, 1].set_ylabel('Y Pixel', fontsize = 16)
axes2[0, 1].tick_params(axis='both', which='major', labelsize=16)


# 2D Gradient Vectors
axes2[1, 0].imshow(data, cmap='gray', aspect='equal', alpha=0.5)
# Only plot vectors if there are significant gradients
if gradient_magnitude.max() > 0.001:
    # Subsample for clarity
    subsample = 4
    X_grid, Y_grid = np.meshgrid(np.arange(0, cols, subsample), 
                                np.arange(0, rows, subsample))
    axes2[1, 0].quiver(X_grid, Y_grid, 
                      gradient_x[::subsample, ::subsample], 
                      -gradient_y[::subsample, ::subsample], 
                      scale=50, color='red', alpha=0.7)
    axes2[1, 0].set_title('Temperature Gradient Vectors')
else:
    axes2[1, 0].set_title('Temperature Gradient Vectors (Zero Gradients)')
    axes2[1, 0].text(cols/2, rows/2, 'NO GRADIENTS\nUNIFORM FIELD', 
                    color='red', fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))
axes2[1, 0].set_xlabel('X Pixel', fontsize=16)
axes2[1, 0].set_ylabel('Y Pixel', fontsize=16)
axes2[1, 0].tick_params(axis='both', which='major', labelsize=16)


# Box plot comparison - FIXED: using 'tick_labels' instead of 'labels'
regions_data = [center_region.flatten(), edge_region]
boxes = axes2[1, 1].boxplot(regions_data, tick_labels=['Center', 'Edges'])  # Fixed here
axes2[1, 1].set_ylabel('Temperature (K)')
axes2[1, 1].set_title('Temperature Distribution: Center vs Edges')
axes2[1, 1].grid(True, alpha=0.3)

# Add annotation if uniform
if is_uniform:
    axes2[1, 1].text(1.5, temp_mean, 'IDENTICAL', 
                    color='green', fontsize=12, ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Save analysis results
print(f"\nSaving analysis results...")

# Save processed data
np.savetxt('temperature_distribution.csv', data, delimiter=',', 
           header='Temperature (K) for 0° rotation')
np.savetxt('temperature_gradients.csv', gradient_magnitude, delimiter=',',
           header='Temperature gradient magnitude (K/pixel)')

# Create summary statistics file - FIXED: using ASCII characters only
with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("INFRARED THERMAL IMAGE ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Image Dimensions: {data.shape[1]} × {data.shape[0]} pixels\n")
    f.write(f"Total Pixels: {data.size}\n\n")
    f.write("TEMPERATURE STATISTICS:\n")
    f.write(f"  Minimum: {temp_min:.2f} K\n")
    f.write(f"  Maximum: {temp_max:.2f} K\n")
    f.write(f"  Mean: {temp_mean:.2f} K\n")
    f.write(f"  Standard Deviation: {temp_std:.2f} K\n")
    f.write(f"  Range: {temp_max - temp_min:.2f} K\n\n")
    f.write("TEMPERATURE UNIFORMITY:\n")
    if is_uniform:
        f.write(f"  [✓] PERFECTLY UNIFORM: All pixels at {temp_mean:.2f} K\n")
    else:
        f.write(f"  [X] NOT UNIFORM: Temperature varies across cross-section\n")  # Fixed here
    f.write(f"\nGRADIENT ANALYSIS:\n")
    f.write(f"  Max X Gradient: {np.abs(gradient_x).max():.4f} K/pixel\n")
    f.write(f"  Max Y Gradient: {np.abs(gradient_y).max():.4f} K/pixel\n")
    f.write(f"  Max Gradient Magnitude: {gradient_magnitude.max():.4f} K/pixel\n\n")
    f.write("REGIONAL ANALYSIS:\n")
    f.write(f"  Center Region Mean: {center_region.mean():.2f} K\n")
    f.write(f"  Edge Region Mean: {edge_region.mean():.2f} K\n")
    f.write(f"  Center-Edge Difference: {center_region.mean() - edge_region.mean():.2f} K\n")
    
    if is_uniform:
        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRETATION AND RECOMMENDATIONS\n")
        f.write("="*60 + "\n\n")
        f.write("The temperature field is perfectly uniform, which suggests:\n")
        f.write("1. Thermal equilibrium may have been reached\n")
        f.write("2. The shaft may be perfectly insulated\n")
        f.write("3. This could be initial/boundary condition data\n")
        f.write("4. Measurement might be from a controlled environment\n\n")
        f.write("Recommended next steps:\n")
        f.write("1. Analyze data from other rotation angles\n")
        f.write("2. Check for transient/temporal data\n")
        f.write("3. Verify measurement and experimental setup\n")
        f.write("4. Compare with theoretical models\n")

print("Analysis complete! Files saved:")
print("  1. temperature_distribution.csv - Raw temperature data")
print("  2. temperature_gradients.csv - Gradient magnitude data")
print("  3. analysis_summary.txt - Summary statistics")

# Optional: Create a simple demonstration with synthetic non-uniform data
print("\n" + "="*60)
print("DEMONSTRATION WITH SYNTHETIC NON-UNIFORM DATA")
print("="*60)
print("\nTo show how the analysis works with temperature variations,")
print("here's what a non-uniform temperature field would look like:")

# Create synthetic non-uniform data for demonstration
synthetic_data = np.ones_like(data) * 200.0
# Add a Gaussian hot spot
x_center, y_center = data.shape[1]//2, data.shape[0]//2
X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
hot_spot = 50 * np.exp(-((X - x_center)**2 + (Y - y_center)**2) / (2 * 20**2))
# Add some edge cooling
edge_cooling = -20 * (np.exp(-((X - 0)**2) / (2 * 30**2)) + 
                      np.exp(-((X - data.shape[1])**2) / (2 * 30**2)) +
                      np.exp(-((Y - 0)**2) / (2 * 30**2)) + 
                      np.exp(-((Y - data.shape[0])**2) / (2 * 30**2)))

synthetic_data = 200.0 + hot_spot + edge_cooling

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Plot synthetic data
im_synth1 = axes3[0].imshow(synthetic_data, cmap='hot', aspect='equal')
plt.colorbar(im_synth1, ax=axes3[0], label='Temperature (K)')
axes3[0].set_title('Synthetic: Temperature with Hot Spot')
axes3[0].set_xlabel('X Pixel')
axes3[0].set_ylabel('Y Pixel')

# With contours
im_synth2 = axes3[1].imshow(synthetic_data, cmap='hot', aspect='equal')
contour_levels_synth = np.linspace(synthetic_data.min(), synthetic_data.max(), 10)
contours_synth = axes3[1].contour(synthetic_data, levels=contour_levels_synth, 
                                  colors='white', linewidths=1, alpha=0.7)
axes3[1].clabel(contours_synth, inline=True, fontsize=8, fmt='%.1f K')
plt.colorbar(im_synth2, ax=axes3[1], label='Temperature (K)')
axes3[1].set_title('Synthetic: With Isotherms')
axes3[1].set_xlabel('X Pixel')
axes3[1].set_ylabel('Y Pixel')

# 3D view
ax_synth3d = fig3.add_subplot(133, projection='3d')
surf_synth = ax_synth3d.plot_surface(X, Y, synthetic_data, cmap='coolwarm', 
                                     alpha=0.8, linewidth=0, antialiased=True)
ax_synth3d.set_xlabel('X Pixel')
ax_synth3d.set_ylabel('Y Pixel')
ax_synth3d.set_zlabel('Temperature (K)')
ax_synth3d.set_title('Synthetic: 3D Surface View')
fig3.colorbar(surf_synth, ax=ax_synth3d, shrink=0.6, aspect=10, label='Temperature (K)')

plt.tight_layout()
plt.show()

# Final message with recommendations
print("\n" + "="*60)
print("FINAL RECOMMENDATIONS FOR YOUR RESEARCH")
print("="*60)
print("\nBased on the analysis of your 0° rotation data:")
print("\n1. CURRENT FINDING:")
print(f"   - Temperature is perfectly uniform at {temp_mean:.2f} K")
print(f"   - No thermal gradients detected")
print(f"   - All regions show identical temperature")
print("\n2. NEXT STEPS:")
print("   - Analyze data from other rotation angles (if available)")
print("   - Check for time-series data to see temporal evolution")
print("   - Verify if this is an initial condition or steady state")
print("   - Compare with theoretical predictions")
print("\n3. EXPERIMENTAL CONSIDERATIONS:")
print("   - Consider introducing a heat source/sink for analysis")
print("   - Check if insulation is too effective")
print("   - Verify IR camera calibration and settings")
print("\n4. DATA ANALYSIS EXTENSIONS:")
print("   - Fourier analysis if periodic patterns are expected")
print("   - Edge detection algorithms for boundary identification")
print("   - Correlation with mechanical stress/strain data")
