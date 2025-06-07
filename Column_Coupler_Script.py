# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 13:38:37 2025

@author: Daniel
"""

import numpy as np
import random
import pandas as pd
import math
import time
from itertools import combinations_with_replacement
from prettytable import PrettyTable

# File path
file_path = r"C:\Users\Daniel\OneDrive\Desktop/Column Coupler case.xlsx"

# Step 1: Read both sheets
sheet1 = pd.read_excel(file_path, sheet_name=0)  # First sheet
sheet2 = pd.read_excel(file_path, sheet_name=1)  # Second sheet

# Step 2: Create a library from Sheet 1
sheet1['Description'] = sheet1['Description'].str.strip().str.lower()
sheet1_library = dict(zip(sheet1['Description'], sheet1['Contents']))

# Display library contents in the desired format
print("Library Created from Sheet 1:")
for key, value in sheet1_library.items():
    print(f"{key}: {value}")

# Retrieve values from the library
max_rebar_length = sheet1_library.get('maximum rebar length', 12000.0)
inner_gap_coupler = sheet1_library.get('inner gap of coupler', 0)
rebar_unit_weight = sheet1_library.get('rebar unit weight', None)
if rebar_unit_weight is None:
    raise ValueError("Error: Rebar Unit Weight is missing from the library.")

# Step 3: Calculate total lengths for all rebar groups
sheet2 = sheet2.fillna(0)
rebar_groups = sheet2['Rebar group']
total_floor_height = sheet2['Total floor height']
dowel_length = sheet2['Dowel length']
hook_anchorage_length = sheet2['Hook anchorage length']
top_girder_depth = sheet2['Top girder depth']

total_length_library = {}
for i in range(len(rebar_groups)):
    total_length = total_floor_height[i] + dowel_length[i] + hook_anchorage_length[i] - top_girder_depth[i]
    total_length_library[rebar_groups[i]] = total_length

print("\nTotal Lengths for All Rebar Groups:")
for group, length in total_length_library.items():
    print(f"{group}: {length}")

# Step 4: Calculate the special length
first_group = list(total_length_library.keys())[0]
first_group_total_length = total_length_library[first_group]
number_of_special_lengths = math.ceil(first_group_total_length / max_rebar_length)
end_bar_length = (first_group_total_length / number_of_special_lengths) - (inner_gap_coupler / 2)
middle_bar_length = (first_group_total_length / number_of_special_lengths) - inner_gap_coupler
purchasable_special_length = math.ceil(max(end_bar_length, middle_bar_length) / 100) * 100

print("\nBar Lengths and Purchasable Special Length:")
print(f"End Bar Length: {end_bar_length}")
print(f"Middle Bar Length: {middle_bar_length}")
print(f"Purchasable Special Length: {purchasable_special_length}")

# Step 5: Calculate number of rebars, special-length rebars, and residual lengths
special_rebar_summary = {}
residual_lengths_data = {}

# List to store updated purchasable special lengths for each group
updated_special_lengths = []

for i, (group, total_length) in enumerate(total_length_library.items()):
    number_of_rebars = math.ceil(total_length / purchasable_special_length)
    
    if i == 0:  # First group
        special_rebar_summary[group] = {
            'Total Length': total_length,
            'Number of Rebars': number_of_rebars,
            'Number of Special Length Rebars': number_of_rebars,
            'Residual Rebars': 0,
            'Residual Length': 0
        }
        updated_special_lengths.append(purchasable_special_length)  # Add to list
    else:
        # Initial number of special-length rebars
        number_of_special_length_rebars = number_of_rebars - 1
        number_of_residual_rebars = 1
        residual_length = total_length - (number_of_special_length_rebars * purchasable_special_length) - (inner_gap_coupler / 2)
        
        # Step 1: Check if the residual length is smaller than (market length - purchasable special length)
        if residual_length < (max_rebar_length - purchasable_special_length):
            # Step 2: Recalculate with market length as the new purchasable special length
            purchasable_special_length = max_rebar_length
            updated_special_lengths.append(purchasable_special_length)  # Add updated special length to list
            
            number_of_rebars = math.ceil(total_length / purchasable_special_length)
            number_of_special_length_rebars = number_of_rebars - 1
            residual_length = total_length - (number_of_special_length_rebars * purchasable_special_length) - (inner_gap_coupler / 2)
            
            # Update residual rebar data
            number_of_residual_rebars = 1  # Recalculate residual rebars count
            
        else:
            # Store the current purchasable special length for this group if no update occurs
            updated_special_lengths.append(purchasable_special_length)
            
        number_of_rebars_in_bundle = sheet2.loc[sheet2['Rebar group'] == group, 'No of rebar in bundle'].values[0]
        total_residual_lengths = number_of_rebars_in_bundle * number_of_residual_rebars

        # Store residual lengths data
        if residual_length in residual_lengths_data:
            residual_lengths_data[residual_length] += total_residual_lengths
        else:
            residual_lengths_data[residual_length] = total_residual_lengths

        # Update the special rebar summary
        special_rebar_summary[group] = {
            'Total Length': total_length,
            'Number of Rebars': number_of_rebars,
            'Number of Special Length Rebars': number_of_special_length_rebars,
            'Residual Rebars': number_of_residual_rebars,
            'Residual Length': residual_length
        }

# Check the length of the updated_special_lengths list and print the special lengths
print("\nUpdated Purchasable Special Lengths for Each Group:")
if len(updated_special_lengths) == len(total_length_library):
    for idx, group in enumerate(total_length_library.keys()):
        print(f"Group {group}: Updated Purchasable Special Length: {updated_special_lengths[idx]}")
else:
    print("Error: The length of updated_special_lengths does not match the number of rebar groups.")
    print(f"Length of updated_special_lengths: {len(updated_special_lengths)}")
    print(f"Number of rebar groups: {len(total_length_library)}")
    
print("\nRebar Group Summary:")
for group, values in special_rebar_summary.items():
    print(f"Group {group} - Total Length: {values['Total Length']:.2f} mm")
    print(f"  Total Number of Rebars: {values['Number of Rebars']}")
    print(f"  Number of Special Length Rebars: {values['Number of Special Length Rebars']}")
    print(f"  Number of Residual Rebars: {values['Residual Rebars']}")
    print(f"  Residual Length: {values['Residual Length']:.2f} mm")

# Step 6: Display and Save the Summary Table
summary_table = PrettyTable()
summary_table.field_names = [
    "Rebar Group",
    "Purchasable Special Length (mm)",
    "Total Special Length Rebars",
    "Number of Special Length Rebars",
    "Total Required Quantity/Weight (kg)",
    "Total Purchased Quantity/Weight (kg)"
]

summary_data = []
sum_total_special_length_rebars = 0
sum_number_of_special_length_rebars = 0
sum_total_required_quantity = 0
sum_total_purchased_quantity = 0

for group, values in special_rebar_summary.items():
    # Fetch the corresponding updated special length for this group
    current_special_length = updated_special_lengths.pop(0)
    
    # Calculate other summary fields as before
    number_of_special_length_rebars = values['Number of Special Length Rebars']
    number_of_rebars_in_bundle = sheet2.loc[sheet2['Rebar group'] == group, 'No of rebar in bundle'].values[0]
    total_special_length_rebars = number_of_special_length_rebars * number_of_rebars_in_bundle
    
    # Handle special case for 2 rebars
    if total_special_length_rebars == 2:
        total_end_bars = 2  # Both rebars are end bars
        total_middle_bars = 0
    else:
        total_end_bars = 2 * number_of_rebars_in_bundle  # Standard calculation for end bars
        total_middle_bars = total_special_length_rebars - total_end_bars  # Remaining are middle bars

    # Calculate weights
    required_end_bar = end_bar_length
    required_mid_bar = middle_bar_length 
    end_bar_quantity = total_end_bars * rebar_unit_weight * (required_end_bar / 1000)
    middle_bar_quantity = total_middle_bars * rebar_unit_weight * (required_mid_bar / 1000)
    
    total_required_quantity = end_bar_quantity + middle_bar_quantity
    total_purchased_quantity = total_special_length_rebars * rebar_unit_weight * (current_special_length / 1000)

    summary_table.add_row([group, current_special_length, total_special_length_rebars, number_of_special_length_rebars,
                           round(total_required_quantity, 2), round(total_purchased_quantity, 2)])
    
    summary_data.append([group, current_special_length, total_special_length_rebars, number_of_special_length_rebars,
                         round(total_required_quantity, 2), round(total_purchased_quantity, 2)])
    
    sum_total_special_length_rebars += total_special_length_rebars
    sum_number_of_special_length_rebars += number_of_special_length_rebars
    sum_total_required_quantity += total_required_quantity
    sum_total_purchased_quantity += total_purchased_quantity

# Add total row to summary table
summary_table.add_row(["Total", "-", sum_total_special_length_rebars, sum_number_of_special_length_rebars,
                       round(sum_total_required_quantity, 2), round(sum_total_purchased_quantity, 2)])

print("\nRebar Summary Table:")
print(summary_table)

# Step 7: Save to Excel
output_file_summary = r"C:\Users\Daniel\OneDrive\Desktop/Rebar_Summary_Column_Couplers.xlsx"
output_file_residuals = r"C:\Users\Daniel\OneDrive\Desktop/Residual_Lengths_Couplers.xlsx"

summary_df = pd.DataFrame(summary_data, columns=[
    "Rebar Group",
    "Purchasable Special Length (mm)",
    "Total Special Length Rebars",
    "Number of Special Length Rebars",
    "Total Required Quantity/Weight (kg)",
    "Total Purchased Quantity/Weight (kg)"
])

# Append total row to DataFrame
summary_df.loc[len(summary_df)] = ["Total", "-", sum_total_special_length_rebars, sum_number_of_special_length_rebars, round(sum_total_required_quantity, 2), round(sum_total_purchased_quantity, 2)]

summary_df.to_excel(output_file_summary, index=False)

print("Rebar summary saved to", output_file_summary)

residual_table = PrettyTable()
residual_table.field_names = ["Length", "Quantity"]
residual_data = []

for length, quantity in residual_lengths_data.items():
    residual_table.add_row([length / 1000, quantity])
    residual_data.append([length / 1000, quantity])

print("\nResidual Lengths Summary:")
print(residual_table)

residual_df = pd.DataFrame(residual_data, columns=["Length", "Quantity"])
residual_df.to_excel(output_file_residuals, index=False)

print("Residual lengths saved to", output_file_residuals)

#Step 8: Cutting Pattern  Optimization

# Load residual rebar lengths from Residual_Lengths.xlsx
residual_file = r"C:/Users/Daniel/OneDrive/Desktop/Residual_Lengths_Couplers.xlsx"
data = pd.read_excel(residual_file)

# Ensure the required columns exist
if 'Length' not in data.columns or 'Quantity' not in data.columns:
    raise ValueError("Residual file must contain 'Length' and 'Quantity' columns.")

# Convert lengths and quantities
required_lengths = data['Length'].tolist()
required_quantities = data['Quantity'].tolist()

# Display input data
print("\n--- Required Rebar Data (from Residual) ---")
print(data.to_string(index=False))

def find_best_combination(stock_length, required_lengths, quantities):
    best_combination = None
    min_waste = stock_length
    all_combinations = []
    
    for r in range(1, min(8, len(required_lengths) + 1)):  # Increased combination limit to 7
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

purchased_quantity_cutting = rebar_unit_weight * optimal_stock_length * total_bars_needed
required_quantity_cutting = purchased_quantity_cutting - (rebar_unit_weight * total_waste)
print(f"Required quantity: {round(required_quantity_cutting, 2)} kg")
print(f"Purchased quantity: {purchased_quantity_cutting} kg")
        
# Save the cutting pattern to an Excel file
output_file = r"C:/Users/Daniel/OneDrive/Desktop/Optimized_Cutting_Pattern_Couplers.xlsx"
cutting_df.to_excel(output_file, index=False)
print(f"\nCutting pattern saved to: {output_file}")