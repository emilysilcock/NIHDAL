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
                            'target': counts['all'].get('target', 0)
                        })
                    
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
                                'target': info.get('target', 0)
                            })
            
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

# Save the DataFrames to CSV
f1_df.to_csv('religion_specific_f1_results.csv', index=False)
count_df.to_csv('religion_specific_counts.csv', index=False)

# Display summary of results - only if we have data
if not f1_df.empty and 'method' in f1_df.columns:
    print("\nF1 Score Results by Method:")
    try:
        pivot_table = f1_df.groupby(['method', 'religion', 'iteration'])['test_f1'].mean().reset_index()
        print(pivot_table.pivot_table(
            index=['method', 'iteration'], 
            columns='religion', 
            values='test_f1'
        ))
    except Exception as e:
        print(f"Error creating F1 pivot table: {e}")
        print("F1 DataFrame summary:")
        print(f1_df.head())

if not count_df.empty and 'method' in count_df.columns:
    print("\nCount Results by Method:")
    try:
        pivot_table = count_df.groupby(['method', 'religion', 'iteration'])[['selected', 'target']].sum().reset_index()
        print(pivot_table.pivot_table(
            index=['method', 'iteration'], 
            columns='religion', 
            values=['selected', 'target']
        ))
    except Exception as e:
        print(f"Error creating count pivot table: {e}")
        print("Count DataFrame summary:")
        print(count_df.head())

# Create plots if data is available
if not f1_df.empty and 'method' in f1_df.columns:
    try:
        # Filter for overall F1 scores
        overall_f1 = f1_df[f1_df['religion'] == 'overall'].copy()

        # Sort for consistent ordering of methods
        method_order = overall_f1.groupby('method')['test_f1'].mean().sort_values(ascending=False).index.tolist()
        
        # Create line plot of Test F1 by iteration for different methods
        plt.figure(figsize=(12, 6))
        
        # Plot F1 scores by iteration
        for method in method_order:
            method_data = overall_f1[overall_f1['method'] == method]
            # Group by iteration and get mean F1 score across seeds
            iter_means = method_data.groupby('iteration')['test_f1'].mean()
            plt.plot(iter_means.index, iter_means.values, marker='o', linewidth=2, label=method)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Test F1 Score', fontsize=12)
        plt.title('Test F1 Score by Method and Iteration', fontsize=14)
        plt.legend(title='Method', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('test_f1_by_method.png', dpi=300)
        plt.close()
        
        # Create bar plot of final Test F1 by method
        plt.figure(figsize=(10, 6))
        
        # Get the final iteration for each method
        final_f1 = overall_f1.groupby(['method', 'seed']).apply(lambda x: x.loc[x['iteration'].idxmax()]).reset_index(drop=True)
        final_f1_avg = final_f1.groupby('method')['test_f1'].agg(['mean', 'std']).reset_index()
        
        # Sort by mean F1 score
        final_f1_avg = final_f1_avg.sort_values('mean', ascending=False)
        
        # Create bar plot with error bars
        plt.bar(final_f1_avg['method'], final_f1_avg['mean'], yerr=final_f1_avg['std'], 
                capsize=5, color=sns.color_palette("colorblind", len(final_f1_avg)))
        
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Final Test F1 Score', fontsize=12)
        plt.title('Final Test F1 Score by Method', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('final_test_f1_by_method.png', dpi=300)
        plt.close()
        
        print("Created F1 score plots: test_f1_by_method.png and final_test_f1_by_method.png")
        
    except Exception as e:
        print(f"Error creating F1 plots: {e}")

if not count_df.empty and 'method' in count_df.columns:
    try:
        # Filter for overall counts
        overall_counts = count_df[count_df['religion'] == 'all'].copy()
        
        # Calculate cumulative counts by method
        methods = overall_counts['method'].unique()
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots for selected and target counts
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Sort methods by the final cumulative target count
        method_order = overall_counts.groupby('method')['target'].sum().sort_values(ascending=False).index.tolist()
        
        # Plot cumulative selected counts
        for method in method_order:
            method_data = overall_counts[overall_counts['method'] == method]
            # Skip iteration 0 as it's usually the initial data
            if 0 in method_data['iteration'].values:
                method_data = method_data[method_data['iteration'] > 0]
            
            # Group by iteration and calculate mean across seeds
            iter_means = method_data.groupby('iteration')['selected'].mean()
            cumulative = iter_means.cumsum()
            
            ax1.plot(cumulative.index, cumulative.values, marker='o', linewidth=2, label=method)
        
        ax1.set_ylabel('Cumulative Selected Count', fontsize=12)
        ax1.set_title('Cumulative Number of Samples Selected by Method', fontsize=14)
        ax1.legend(title='Method', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot cumulative target counts
        for method in method_order:
            method_data = overall_counts[overall_counts['method'] == method]
            # Skip iteration 0 as it's usually the initial data
            if 0 in method_data['iteration'].values:
                method_data = method_data[method_data['iteration'] > 0]
            
            # Group by iteration and calculate mean across seeds
            iter_means = method_data.groupby('iteration')['target'].mean()
            cumulative = iter_means.cumsum()
            
            ax2.plot(cumulative.index, cumulative.values, marker='o', linewidth=2, label=method)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Cumulative Target Count', fontsize=12)
        ax2.set_title('Cumulative Number of Target Samples Selected by Method', fontsize=14)
        ax2.legend(title='Method', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('cumulative_counts_by_method.png', dpi=300)
        plt.close()
        
        # Create bar plot showing final counts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Final counts
        final_counts = overall_counts.groupby(['method', 'seed']).agg({
            'selected': 'sum', 
            'target': 'sum'
        }).reset_index()
        
        final_counts_avg = final_counts.groupby('method').agg({
            'selected': ['mean', 'std'],
            'target': ['mean', 'std']
        }).reset_index()
        
        # Sort by mean target count
        final_counts_avg = final_counts_avg.sort_values(('target', 'mean'), ascending=False)
        
        # Plot selected counts
        ax1.bar(final_counts_avg['method'], final_counts_avg[('selected', 'mean')], 
                yerr=final_counts_avg[('selected', 'std')], capsize=5,
                color=sns.color_palette("colorblind", len(final_counts_avg)))
        
        ax1.set_xlabel('Method', fontsize=12)
        ax1.set_ylabel('Total Selected Count', fontsize=12)
        ax1.set_title('Total Selected Samples by Method', fontsize=14)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Plot target counts
        ax2.bar(final_counts_avg['method'], final_counts_avg[('target', 'mean')], 
                yerr=final_counts_avg[('target', 'std')], capsize=5,
                color=sns.color_palette("colorblind", len(final_counts_avg)))
        
        ax2.set_xlabel('Method', fontsize=12)
        ax2.set_ylabel('Total Target Count', fontsize=12)
        ax2.set_title('Total Target Samples by Method', fontsize=14)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('total_counts_by_method.png', dpi=300)
        plt.close()
        
        print("Created count plots: cumulative_counts_by_method.png and total_counts_by_method.png")
        
    except Exception as e:
        print(f"Error creating count plots: {e}") 