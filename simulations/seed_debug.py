import pickle
import os
import pandas as pd
import numpy as np

# Define the directory containing the pickle files
directory = 'data/sim_results_0427/'

# Let's directly examine a few specific pickle files to check raw data
# Focus on one seed for multiple methods
seed = "42"
methods = ["Random", "Least Confidence", "NIHDAL", "DAL2", "Core Set"]

print("=== DETAILED RAW DATA ANALYSIS ===")
print(f"Examining seed {seed} for methods: {methods}")

for method in methods:
    filename = f"hate_speech_{method}_results_{seed}_unbiased.pkl"
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    print(f"\n=== METHOD: {method} ===")
    
    # Load the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Create a summary of the iterations
    print(f"Total iterations: {len(data)}")
    
    # Examine counts for each iteration
    iteration_counts = []
    
    for i, result in enumerate(data):
        target_count = 0
        selected_count = 0
        
        if 'counts' in result and result['counts'] is not None:
            if 'all' in result['counts']:
                selected_count = result['counts']['all'].get('selected', 0)
                target_count = result['counts']['all'].get('target', 0)
        
        iteration_counts.append({
            'iteration': i,
            'selected': selected_count,
            'target': target_count,
            'target_ratio': target_count / selected_count if selected_count > 0 else 0,
            'test_f1': result.get('Test F1', 'N/A')
        })
    
    # Convert to DataFrame for easier analysis
    counts_df = pd.DataFrame(iteration_counts)
    
    # Print summary statistics
    print("\nIteration-by-iteration counts:")
    print(counts_df[['iteration', 'selected', 'target', 'target_ratio', 'test_f1']])
    
    # Check if all target counts are the same after iteration 0
    if len(counts_df) > 1:
        active_df = counts_df[counts_df['iteration'] > 0]
        unique_targets = active_df['target'].unique()
        
        print(f"\nUnique target counts after iteration 0: {unique_targets}")
        
        if len(unique_targets) == 1:
            print(f"WARNING: All iterations after the initial one have the exact same target count: {unique_targets[0]}")
        
        # Check if target counts increase by the same amount each iteration
        if len(active_df) > 1:
            target_diffs = np.diff(active_df['target'].values)
            unique_diffs = np.unique(target_diffs)
            
            print(f"Differences between consecutive target counts: {target_diffs}")
            print(f"Unique differences: {unique_diffs}")
            
            if len(unique_diffs) == 1:
                print(f"WARNING: Target count increases by exactly {unique_diffs[0]} in each iteration")

# Now inspect the actual active learning code in the simulation file
print("\n=== EXAMINING ACTIVE LEARNING IMPLEMENTATION ===")

# Get the path to the simulation code file
code_file = "simulations/active_learning_simulation_hate_speech.py"

if os.path.exists(code_file):
    with open(code_file, 'r') as f:
        code = f.readlines()
    
    # Extract the key parts of the code that might be related to the issue
    relevant_sections = []
    
    # Look for the active learning loop and how counts are tracked
    for i, line in enumerate(code):
        if "active_learning_loop" in line or "selected_descr" in line or "counts" in line:
            context = "".join(code[max(0, i-5):min(len(code), i+15)])
            relevant_sections.append(f"Line {i+1}:\n{context}\n")
    
    print("Relevant code sections:")
    for section in relevant_sections:
        print(section)
else:
    print(f"Could not find simulation code file: {code_file}")

# Look for consistency in target sample selection
print("\n=== CONSISTENCY CHECK ACROSS ALL SEEDS ===")
all_files = [f for f in os.listdir(directory) if f.endswith("_unbiased.pkl")]

# Group by method and check consistency
method_consistency = {}

for method in methods:
    method_files = [f for f in all_files if f"hate_speech_{method}_results_" in f]
    
    print(f"\nMethod: {method}, Files: {len(method_files)}")
    
    # Store iteration target counts for all seeds
    all_seed_counts = []
    
    for file in method_files:
        try:
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            seed = file.split("_results_")[1].split("_")[0]
            iter_counts = []
            
            for i, result in enumerate(data):
                if i > 0 and 'counts' in result and result['counts'] is not None:
                    if 'all' in result['counts']:
                        target = result['counts']['all'].get('target', 0)
                        iter_counts.append(target)
            
            all_seed_counts.append({
                'seed': seed,
                'target_counts': iter_counts
            })
            
            print(f"  Seed {seed}: Target counts {iter_counts}")
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    # Check if target counts are consistent across seeds
    if all_seed_counts:
        count_lengths = [len(seed_data['target_counts']) for seed_data in all_seed_counts]
        if len(set(count_lengths)) > 1:
            print(f"  Different number of iterations across seeds")
        else:
            # Compare target counts for each iteration across seeds
            iterations = count_lengths[0]
            for iter_idx in range(iterations):
                iter_targets = [seed_data['target_counts'][iter_idx] if iter_idx < len(seed_data['target_counts']) else None 
                               for seed_data in all_seed_counts]
                unique_targets = set([t for t in iter_targets if t is not None])
                
                if len(unique_targets) == 1:
                    print(f"  ALERT: Iteration {iter_idx+1} has identical target count {next(iter(unique_targets))} across all seeds") 