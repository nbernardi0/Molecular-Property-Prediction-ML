import matplotlib.pyplot as plt
from Utils import save_plot
from rdkit.Chem import Draw
from rdkit import Chem
import os
import seaborn as sns
import pandas as pd

# ------------------------------ main_input.py plots ----------------------------------
def plot_frequency(freq_values, label, saving_name, threshold, save_dir, x_labels=None):
    fig = plt.figure(figsize=(12, 5))

    x_vals = list(range(len(freq_values)))
    
    # Assign colors: red if below threshold, green if above or equal
    colors = ["green" if val >= threshold else "red" for val in freq_values]
    plt.bar(x_vals, freq_values, color=colors)
    
    # Plot threshold line
    plt.axhline(threshold, color='black', linestyle='--', label=f"Threshold = {threshold}")
    plt.legend(fontsize=16)

    if x_labels is not None:
        plt.xticks(x_vals, x_labels, fontsize=14, rotation=90)
    else:
        plt.xticks(fontsize=14)

    plt.xlabel(f"{label} ID", fontsize=20)
    plt.ylabel("Frequency (occurrences)", fontsize=20)
    plt.yscale("log")
    plt.yticks(fontsize=20)
    plt.tight_layout()
    save_plot(fig, saving_name, save_dir)
    #plt.show()

def plot_functional_groups(fparams, fids, freq_func_groups=None, sort_by_freq=False, save_path=None):
                        
    fg_mols = []
    fg_labels = []

    # Build (fid, freq) list
    if freq_func_groups is not None:
        func_with_freq = [(fid, freq_func_groups[fid]) for fid in fids]
        if sort_by_freq:
            func_with_freq = sorted(func_with_freq, key=lambda x: x[1], reverse=True)
    else:
        func_with_freq = [(fid, None) for fid in fids]

    for fid, freq in func_with_freq:
        fg = fparams.GetFuncGroup(fid)
        label = fg.GetProp("_Name") if fg.HasProp("_Name") else f"FG-{fid}"
        smarts = Chem.MolToSmarts(fg)
        mol = Chem.MolFromSmarts(smarts)
        if mol:
            fg_mols.append(mol)
            if freq is not None:
                fg_labels.append(f"#{fid+1} {label} ({int(freq)})")
            else:
                fg_labels.append(f"#{fid+1} {label}")

    img = Draw.MolsToGridImage(
        fg_mols,
        molsPerRow=5,
        subImgSize=(250, 250),
        legends=fg_labels
    )
    img.save(save_path + ".png")
    img.save(save_path + ".pdf")
    print(f"Saved: {save_path}.png and {save_path}.pdf")
    
def plot_correlation_heatmap(Y_labels, property_names, save_dir):

    # Convert labels into a DataFrame and compute Pearson correlation matrix
    labels_df = pd.DataFrame(Y_labels, columns=property_names)
    corr = labels_df.corr()

    # Create the heatmap plot (single-color gradient)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", cbar=True,
                xticklabels=property_names, yticklabels=property_names, ax=ax)
    ax.set_title("Correlation heatmap of molecular properties")
    fig.tight_layout()

    save_plot(fig, "correlation_heatmap", save_dir)
    plt.close(fig)

# ------------------------------ NN_training.py plots ----------------------------------

def plot_training_curves(MSE_training_history, MSE_val_history, plots_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(MSE_training_history, label='Training Loss')
    plt.plot(MSE_val_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.yscale("log")
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_png = os.path.join(plots_dir, "training_loss.png") # plot saving
    loss_pdf = os.path.join(plots_dir, "training_loss.pdf")
    plt.savefig(loss_png, dpi=300)                          # dpi (= dots per inch) for better quality
    plt.savefig(loss_pdf, format="pdf")
    plt.show()
    print(f"[INFO] Training curves saved in {plots_dir}")

def plot_parity_plots(Y_test, Y_pred, r2_scores, property_names, plots_dir):
    fig, axes = plt.subplots(4, 3, figsize=(16, 13))
    # fig.suptitle('Parity Plots for Test Molecules', fontsize=18)
    for i, ax in enumerate(axes.ravel()): # from shape (4,3) to (12,) --> enumerate to obtain both index and axis
        ax.scatter(Y_test[:, i], Y_pred[:, i], alpha=0.4, edgecolor='k', linewidth=0.3, s=20) # x-axes = true values, y-axes = predicted values
        ax.plot([Y_test[:, i].min(), Y_test[:, i].max()],
                [Y_test[:, i].min(), Y_test[:, i].max()], 'r--', linewidth=1.2) # diagonal line y=x for reference
        ax.set_title(property_names[i], fontsize=13)
        ax.text(0.05, 0.90, f"$R^2$ = {r2_scores[i]:.3f}", transform=ax.transAxes,   # to add R2 value in the plot
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))
        ax.set_xlabel('True', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])                   # to adjust layout to fit suptitle (0.95 to leave top space for suptitle)
    parity_png = os.path.join(plots_dir, "parity_plots.png") # plot saving
    parity_pdf = os.path.join(plots_dir, "parity_plots.pdf")
    plt.savefig(parity_png, dpi=300)
    plt.savefig(parity_pdf, format="pdf")

    print(f"[INFO] Parity plots saved in {plots_dir}")
    
    
# ------------------------------ RandomForest.py plots ----------------------------------
    
def plot_rf_r2_vs_time(n_estimators, R2_global, Training_time, plots_dir):
   
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- Left Y-axis: Global R² ---
    color = 'tab:blue'
    ax1.set_xscale("log")  # logarithmic scale for x-axis
    ax1.plot(n_estimators, R2_global, '-o', color=color, linewidth=2, markersize=8, label="R² global")
    ax1.set_xlabel("n_estimators", fontsize=14)
    ax1.set_ylabel("R² global", color=color, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.8, 0.9)  # fix y-limits for clarity
    ax1.grid(True, which="both", linestyle="--", linewidth=0.6)

    # --- Right Y-axis: Training Time ---
    ax2 = ax1.twinx()  # create a second y-axis
    color = 'tab:red'
    ax2.plot(n_estimators, Training_time, '-s', color=color, linewidth=2, markersize=8, label="Training time [s]")
    ax2.set_ylabel("Training Time [s]", color=color, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=color)

    # --- Title and Combined Legend ---

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    # Save and show
    plt.tight_layout()
    save_plot(fig, "rf_r2_vs_time", plots_dir)
    plt.show()
    print(f"[INFO] RF R2 vs Training time plot saved in {plots_dir}")

    
# -------------------- Regression Plots -------------------

def plot_coefficients_grid(coef, property_names, model_choice, plots_dir, alpha=None):
 
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))

    # One subplot per property
    for i, ax in enumerate(axes.ravel()):
        if i < coef.shape[0]:
            ax.bar(range(coef.shape[1]), coef[i])
            ax.set_title(property_names[i], fontsize=12)
            ax.set_xlabel("Variable index", fontsize=9)
            ax.set_ylabel("Coefficient value (scaled)", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, linestyle="--", linewidth=0.5)

    # Layout + save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    coef_plot_png = os.path.join(plots_dir, f"coefficients_grid_{model_choice}.png")
    coef_plot_pdf = os.path.join(plots_dir, f"coefficients_grid_{model_choice}.pdf")

    plt.savefig(coef_plot_png, dpi=300)
    plt.savefig(coef_plot_pdf)

    plt.show()
    plt.close(fig)

    print(f"[INFO] Coefficient grid plot saved in {coef_plot_png} and {coef_plot_pdf}")
