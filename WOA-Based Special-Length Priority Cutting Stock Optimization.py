# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:43:27 2025

@author: user
"""

import numpy as np
import random
import pandas as pd
import time
from itertools import combinations_with_replacement

# Load discontinuous rebar lengths from the file
discontinuous_file = r"C:/Users/user/Desktop/Short bars.xlsx"
data = pd.read_excel(discontinuous_file)

# Ensure the required columns exist
if 'Length' not in data.columns or 'Quantity' not in data.columns:
    raise ValueError("Discontinuous file must contain 'Length' and 'Quantity' columns.")

# Convert lengths and quantities
required_lengths = data['Length'].tolist()
required_quantities = data['Quantity'].tolist()

# Display input data
print("\n--- Required Rebar Data (from Discontinuous) ---")
print(data.to_string(index=False))

def find_best_combination(stock_length, required_lengths, quantities):
    best_combination = None
    min_waste = stock_length
    all_combinations = []
    
    for r in range(1, min(20, len(required_lengths) + 1)):  
        for comb in combinations_with_replacement(required_lengths, r):
            if all(comb.count(x) <= quantities[required_lengths.index(x)] for x in comb):
                total_length = sum(comb)
                if total_length <= stock_length:
                    all_combinations.append((comb, total_length))
    
    # Sort by maximizing length utilization
    all_combinations.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
    
    for combination, total_length in all_combinations:
        waste = stock_length - total_length
        temp_quantities = quantities.copy()
        valid_combination = True

        for length in combination:
            idx = required_lengths.index(length)
            if temp_quantities[idx] > 0:
                temp_quantities[idx] -= 1
            else:
                valid_combination = False
                break

        if valid_combination and waste < min_waste:
            best_combination = combination
            min_waste = waste
    
    return best_combination if best_combination else [], min_waste

def calculate_waste(stock_length, required_lengths, quantities):
    total_waste = 0
    total_bars_needed = 0
    cutting_pattern = []
    remaining_quantities = quantities.copy()

    while sum(remaining_quantities) > 0:
        combination, waste = find_best_combination(stock_length, required_lengths, remaining_quantities)
        if not combination:
            break
        
        cutting_pattern.append({
            'Combination': [round(length, 2) for length in combination],
            'Cut from Length (m)': round(stock_length, 1),
            'Waste (m)': round(waste, 2)
        })
        
        total_waste += waste
        total_bars_needed += 1
        
        for length in combination:
            index = required_lengths.index(length)
            remaining_quantities[index] -= 1

    return total_waste, total_bars_needed, cutting_pattern, remaining_quantities

def fitness_function(stock_length):
    if any(stock_length < length for length in required_lengths):
        return float('inf')  # Infeasible solution
    
    total_waste, _, _, _ = calculate_waste(stock_length, required_lengths, required_quantities.copy())
    waste_penalty = total_waste * (0.5 + (random.uniform(0.1, 0.5)))  # Dynamic penalty factor
    return total_waste + waste_penalty

# WOA Parameters
num_whales = 30
num_iterations = 100  # Increased iterations for better convergence
whales = np.random.uniform(6.0, 12.0, num_whales)
best_whale = whales[0]
best_fitness = fitness_function(best_whale)

# Start timing the entire process
start_time = time.time()

for iteration in range(num_iterations):
    iter_start = time.time()  # Start time for this iteration

    a = 2 - iteration * (2 / num_iterations)  
    
    for i in range(num_whales):
        r = random.random()
        A = 2 * a * r - a
        C = 2 * r
        l = random.uniform(-1, 1)
        p = random.random()
        
        if p < 0.5:
            if abs(A) < 1:
                D_leader = abs(C * best_whale - whales[i])
                whales[i] = best_whale - A * D_leader
            else:
                random_whale_index = random.randint(0, num_whales - 1)
                D_random_whale = abs(C * whales[random_whale_index] - whales[i])
                whales[i] = whales[random_whale_index] - A * D_random_whale
        else:
            D_leader = abs(best_whale - whales[i])
            whales[i] = D_leader * np.exp(l) * np.cos(2 * np.pi * l) + best_whale
        
        whales[i] = round(np.clip(whales[i], 6.0, 12.0) * 10) / 10
        
        # Introduce occasional mutation to avoid premature convergence
        if iteration > num_iterations * 0.7 and random.random() < 0.1:
            whales[i] += random.uniform(-0.2, 0.2)
            whales[i] = round(np.clip(whales[i], 6.0, 12.0) * 10) / 10
        
        fitness = fitness_function(whales[i])
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_whale = whales[i]
    
    iter_end = time.time()  # End time for this iteration
    iter_time = iter_end - iter_start  # Compute iteration time

    print(f"Iteration {iteration + 1}/{num_iterations}, Best Waste: {best_fitness:.2f} m, Time: {iter_time:.4f} sec")

# End timing for the entire WOA process
end_time = time.time()
total_execution_time = end_time - start_time

optimal_stock_length = round(best_whale, 1)
print(f"\nBest Rebar Length: {optimal_stock_length} m")

total_waste, total_bars_needed, cutting_pattern, remaining_quantities = calculate_waste(
    optimal_stock_length, required_lengths, required_quantities.copy())

print(f"Total Waste: {round(total_waste, 2)} m")
print(f"Total Bars Needed: {total_bars_needed}")

cutting_df = pd.DataFrame(cutting_pattern)
print("\nCutting Pattern:")
print(cutting_df.to_string(index=False))

print("\nRemaining Quantities:")
for length, quantity in zip(required_lengths, remaining_quantities):
    if quantity > 0:
        print(f"Length {round(length, 2)}m: {quantity}")

# Display total execution time
print(f"\nTotal Computational Time: {total_execution_time:.4f} sec")

# Save the cutting pattern to an Excel file
output_file =  r"C:/Users/user/Desktop/Optimized_CuttingShort_H20.xlsx" 
cutting_df.to_excel(output_file, index=False)
print(f"\nCutting pattern saved to: {output_file}")
