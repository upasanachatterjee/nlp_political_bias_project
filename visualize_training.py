import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def parse_training_log(log_file_path):
    """Parse the training log file and extract relevant data."""
    
    data = []
    task_names = set()
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # First, find all unique task names from the Task Averages sections
    task_avg_pattern = r'Task Averages:\s+\{([^}]+)\}'
    task_matches = re.findall(task_avg_pattern, content)
    
    for task_match in task_matches:
        # Extract individual task names from the dictionary string
        task_pattern = r"'([^']+)'"
        tasks_in_match = re.findall(task_pattern, task_match)
        task_names.update(tasks_in_match)
    
    task_names = sorted(list(task_names))  # Sort for consistent ordering
    print(f"Detected tasks: {task_names}")
    
    # Use a more flexible approach: extract step info and task averages separately
    step_base_pattern = r'Step\s+(\d+)\s+\|\s+LR:\s+([\d.e+-]+)\s+\|\s+Epoch:\s+(\d+)\s+\|\s+Progress:\s+([\d.]+)%.*?Task Averages:\s+\{([^}]+)\}'
    
    matches = re.findall(step_base_pattern, content, re.DOTALL)
    
    for match in matches:
        step, lr, epoch, progress, task_dict_str = match
        
        entry = {
            'step': int(step),
            'learning_rate': float(lr),
            'epoch': int(epoch),
            'progress': float(progress),
        }
        
        # Parse the task averages dictionary string
        task_value_pattern = r"'([^']+)':\s+([\d.]+)"
        task_matches = re.findall(task_value_pattern, task_dict_str)
        
        # Add task averages
        for task_name, task_value in task_matches:
            entry[f'{task_name}_avg'] = float(task_value)
        
        # Ensure all known tasks have values (fill with NaN if missing)
        for task_name in task_names:
            if f'{task_name}_avg' not in entry:
                entry[f'{task_name}_avg'] = np.nan
        
        data.append(entry)
    
    df = pd.DataFrame(data)
    # Store task names as a proper attribute using pandas metadata
    df.attrs['task_names'] = task_names
    return df

def create_task_averages_plot(df, save_path=None):
    """Create a comprehensive plot of task averages over training steps."""
    
    task_names = df.attrs.get('task_names', [])
    # Generate distinct colors for each task
    cmap = plt.colormaps.get_cmap('tab10')  # Good for up to 10 distinct colors
    colors = [cmap(i / max(1, len(task_names) - 1)) for i in range(len(task_names))]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Task averages over steps
    for i, task_name in enumerate(task_names):
        col_name = f'{task_name}_avg'
        if col_name in df.columns:
            ax1.plot(df['step'], df[col_name], 
                    label=f'{task_name.title()} Loss', 
                    color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Task Average Loss')
    ax1.set_title('Task Averages Over Training Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Task averages with learning rate overlay
    ax2_lr = ax2.twinx()
    
    for i, task_name in enumerate(task_names):
        col_name = f'{task_name}_avg'
        if col_name in df.columns:
            ax2.plot(df['step'], df[col_name], 
                    label=f'{task_name.title()} Loss', 
                    color=colors[i], linewidth=2)
    
    ax2_lr.plot(df['step'], df['learning_rate'], label='Learning Rate', color='green', 
                linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Task Average Loss', color='black')
    ax2_lr.set_ylabel('Learning Rate', color='green')
    ax2.set_title('Task Averages with Learning Rate Schedule')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_lr.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    return fig

def main():
    
    log_file = "tmp/triplet_mlm.txt"
    
    df = parse_training_log(log_file)
    
    if df.empty:
        return
    
    print(f"Found {len(df)} training steps")
    
    # Create visualizations
    create_task_averages_plot(df, save_path="training_analysis.png")

if __name__ == "__main__":
    main()