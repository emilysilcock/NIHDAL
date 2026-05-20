import pickle
import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
plt.style.use('ggplot')
sns.set_palette("colorblind")

# Define the directory containing the pickle files
directory = 'data/sim_results_0427/'

# Find all pickle files in the directory
pickle_files = glob.glob(os.path.join(directory, '*.pkl'))

# Create lists to store our results
f1_results = []
count_results = []

# Process each pickle file
for file_path in pickle_files:
    try:
        # Get just the filename without path
        filename = os.path.basename(file_path)
        
        print(f"Processing: {filename}")
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract method and seed from filename
        # Expected format: hate_speech_METHOD_results_SEED_unbiased.pkl
        method = "unknown"
        seed = "unknown"
        
        if "hate_speech_" in filename and "_results_" in filename:
            method_part = filename.split("hate_speech_")[1].split("_results_")[0]
            method = method_part
            
            # Extract seed
            seed_part = filename.split("_results_")[1].split("_")[0]
            if seed_part.isdigit():
                seed = seed_part
        
        print(f"  Extracted method: {method}, seed: {seed}")
        
        # Debug: Print first few iterations of data structure for Random
        if method == "Random" and seed == "42":
            print(f"DEBUG - Random method structure:")
            for i, result in enumerate(data[:3]):  # First 3 iterations
                print(f"  Iteration {i}:")
                if 'counts' in result and result['counts'] is not None:
                    print(f"    Counts present: {result['counts'].keys()}")
                    if 'all' in result['counts']:
                        print(f"    All counts: {result['counts']['all']}")
                    else:
                        print("    No 'all' key in counts")
                else:
                    print("    No counts data")
                print(f"    Test F1: {result.get('Test F1', 'N/A')}")
        
        # Iterate through each iteration's results
        for iteration, result in enumerate(data):
            # 1. Extract overall Test F1
            if 'Test F1' in result:
                f1_results.append({
                    'filename': filename,
                    'method': method,
                    'seed': seed,
                    'iteration': iteration,
                    'religion': 'overall',
                    'test_f1': result['Test F1']
                })
            
            # 2. Extract religion-specific Test F1 scores
            for key in result:
                if key.startswith('Test F1_') and not key.endswith('overall'):
                    religion = key.replace('Test F1_', '')
                    f1_results.append({
                        'filename': filename,
                        'method': method,
                        'seed': seed,
                        'iteration': iteration,
                        'religion': religion,
                        'test_f1': result[key]
                    })
            
            # 3. Extract count information
            if 'counts' in result:
                counts = result['counts']
                if counts is not None:
                    # Overall counts
                    if 'all' in counts:
                        count_results.append({
                            'filename': filename,
                            'method': method,
                            'seed': seed,
                            'iteration': iteration,
                            'religion': 'all',
                            'selected': counts['all'].get('selected', 0),
                            'target': counts['all'].get('target', 0),
                            'target_ratio': counts['all'].get('target', 0) / counts['all'].get('selected', 1) if counts['all'].get('selected', 0) > 0 else 0
                        })
                        
                        # Extra debug info for Random method
                        if method == "Random":
                            print(f"  Random - Iter {iteration}: Selected={counts['all'].get('selected', 0)}, Target={counts['all'].get('target', 0)}")
                    
                    # Religion-specific counts
                    for religion, info in counts.items():
                        if religion != 'all':
                            count_results.append({
                                'filename': filename,
                                'method': method,
                                'seed': seed,
                                'iteration': iteration,
                                'religion': religion,
                                'selected': info.get('selected', 0),
                                'target': info.get('target', 0),
                                'target_ratio': info.get('target', 0) / info.get('selected', 1) if info.get('selected', 0) > 0 else 0
                            })
            else:
                print(f"  WARNING: No counts data for {method}, seed {seed}, iteration {iteration}")
                
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Create DataFrames from results
f1_df = pd.DataFrame(f1_results)
count_df = pd.DataFrame(count_results)

# Print out column names and sample data to debug
print("F1 DataFrame columns:", f1_df.columns.tolist())
if not f1_df.empty:
    print("F1 DataFrame first row:", f1_df.iloc[0].to_dict())

print("Count DataFrame columns:", count_df.columns.tolist())
if not count_df.empty:
    print("Count DataFrame first row:", count_df.iloc[0].to_dict())

# Print summary stats by method
if not count_df.empty and 'method' in count_df.columns:
    print("\nSummary of counts by method:")
    method_summary = count_df[count_df['religion'] == 'all'].groupby('method').agg({
        'selected': ['sum', 'mean'],
        'target': ['sum', 'mean'],
        'target_ratio': ['mean']
    })
    print(method_summary)

    # Check if iteration 0 has different patterns (initial data)
    print("\nIteration 0 (initial data) vs. rest:")
    init_data = count_df[(count_df['religion'] == 'all') & (count_df['iteration'] == 0)].groupby('method').agg({
        'selected': ['sum', 'mean'],
        'target': ['sum', 'mean'],
        'target_ratio': ['mean']
    })
    print("Initial data (iteration 0):")
    print(init_data)
    
    active_data = count_df[(count_df['religion'] == 'all') & (count_df['iteration'] > 0)].groupby('method').agg({
        'selected': ['sum', 'mean'],
        'target': ['sum', 'mean'],
        'target_ratio': ['mean']
    })
    print("Active learning data (iterations > 0):")
    print(active_data)

# Save the DataFrames to CSV
f1_df.to_csv('debug_f1_results.csv', index=False)
count_df.to_csv('debug_counts_results.csv', index=False)

# Create plots if data is available
if not f1_df.empty and not count_df.empty and 'method' in f1_df.columns and 'method' in count_df.columns:
    try:
        # Plot F1 vs Target Ratio
        # Filter for overall F1 scores and all counts
        overall_f1 = f1_df[f1_df['religion'] == 'overall'].copy()
        overall_counts = count_df[count_df['religion'] == 'all'].copy()
        
        # Merge the data
        merged_data = pd.merge(
            overall_f1, 
            overall_counts, 
            on=['method', 'seed', 'iteration'], 
            suffixes=('_f1', '_counts')
        )
        
        # Create a plot of F1 vs Target Ratio by method
        plt.figure(figsize=(12, 8))
        methods = merged_data['method'].unique()
        
        for method in methods:
            method_data = merged_data[merged_data['method'] == method]
            plt.scatter(
                method_data['target_ratio'], 
                method_data['test_f1'], 
                label=method,
                alpha=0.7,
                s=60
            )
        
        plt.xlabel('Target Ratio (Positives / Total Selected)', fontsize=12)
        plt.ylabel('Test F1 Score', fontsize=12)
        plt.title('F1 Score vs Target Sample Ratio by Method', fontsize=14)
        plt.legend(title='Method', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('f1_vs_target_ratio.png', dpi=300)
        plt.close()
        
        # Plot Iteration vs Target Count per Method (not cumulative, actual per iteration)
        plt.figure(figsize=(12, 8))
        
        # To understand the linearity, look at per-iteration (not cumulative) counts
        iter_counts = overall_counts.groupby(['method', 'iteration']).agg({
            'selected': 'mean',
            'target': 'mean',
            'target_ratio': 'mean'
        }).reset_index()
        
        # Plot target counts per iteration
        for method in iter_counts['method'].unique():
            method_data = iter_counts[iter_counts['method'] == method]
            plt.plot(
                method_data['iteration'],
                method_data['target'],
                marker='o',
                linewidth=2,
                label=method
            )
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Target Count (Per Iteration)', fontsize=12)
        plt.title('Target Samples Selected Per Iteration by Method', fontsize=14)
        plt.legend(title='Method', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('target_per_iteration.png', dpi=300)
        plt.close()
        
        # Plot Test F1 vs Iteration Together with Target Ratio
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Method order for consistent colors
        method_order = overall_f1.groupby('method')['test_f1'].mean().sort_values(ascending=False).index.tolist()
        
        # Plot F1 scores by iteration - top subplot
        for method in method_order:
            method_data = overall_f1[overall_f1['method'] == method]
            # Group by iteration and get mean F1 score across seeds
            iter_means = method_data.groupby('iteration')['test_f1'].mean()
            ax1.plot(iter_means.index, iter_means.values, marker='o', linewidth=2, label=method)
        
        ax1.set_ylabel('Test F1 Score', fontsize=12)
        ax1.set_title('Test F1 Score by Method and Iteration', fontsize=14)
        ax1.legend(title='Method', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot target ratio by iteration - bottom subplot
        for method in method_order:
            method_data = overall_counts[overall_counts['method'] == method]
            # Group by iteration and calculate target ratio
            iter_means = method_data.groupby('iteration')['target_ratio'].mean()
            ax2.plot(iter_means.index, iter_means.values, marker='o', linewidth=2, label=method)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Target Ratio (Positives / Total)', fontsize=12)
        ax2.set_title('Target Ratio by Method and Iteration', fontsize=14)
        ax2.legend(title='Method', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('f1_and_target_ratio.png', dpi=300)
        plt.close()
                
    except Exception as e:
        print(f"Error creating debug plots: {e}") 