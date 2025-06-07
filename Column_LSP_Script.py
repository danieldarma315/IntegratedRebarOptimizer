# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 12:04:03 2025

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
file_path =  r"C:\Users\Daniel\OneDrive\Desktop/Column LSP case.xlsx"
# Step 1: Read both sheets
sheet1 = pd.read_excel(file_path, sheet_name=0)  # First sheet
sheet2 = pd.read_excel(file_path, sheet_name=1)  # Second sheet

# Step 2: Create a library from Sheet 1
# Standardize 'Description' for consistency (strip spaces, lowercase)
sheet1['Description'] = sheet1['Description'].str.strip().str.lower()

# Build a dictionary with 'Description' as keys and 'Contents' as values
sheet1_library = dict(zip(sheet1['Description'], sheet1['Contents']))

# Debug: Display the library
print("Library from Member and Rebar Specifications:")
for key, value in sheet1_library.items():
    print(f"{key}: {value}")

# Step 3: Retrieve Maximum Rebar Length from the library
max_rebar_length = sheet1_library.get('maximum rebar length')
max_rebar_length = int(max_rebar_length)

# Check if the value is available; if not, print a warning but don't stop the program
if pd.isna(max_rebar_length):
    print("Warning: 'Maximum Rebar Length' value is missing or NaN. Using a default value of 12000.0 meters.")
    max_rebar_length = 12000.0  # Default value

print(f"\nMaximum rebar length retrieved: {max_rebar_length:.2f} mm")

# Step 4: Calculate total lengths for all rebar groups and store in a library
# Replace NaN values with 0 before calculation
sheet2 = sheet2.fillna(0)  # Replace NaN with 0 in the entire sheet

rebar_groups = sheet2['Rebar group']  # Assuming a column exists to identify rebar groups
total_floor_height = sheet2['Total floor height']
dowel_length = sheet2['Dowel length']
hook_anchorage_length = sheet2['Hook anchorage length']
total_lapping_length = sheet2['Total lapping length']
top_girder_depth = sheet2['Top girder depth']
number_of_rebars_in_bundle = int(sheet2['No of rebar in bundle'].iloc[0])

# Initialize a dictionary to store total lengths
total_length_library = {}

# Calculate total length for each group
for i in range(len(rebar_groups)):
    total_length = total_floor_height[i] + dowel_length[i] + hook_anchorage_length[i] + total_lapping_length[i] - top_girder_depth[i]
    total_length_library[rebar_groups[i]] = total_length

# Display the total lengths
print("\nTotal Lengths for All Rebar Groups:")
for group, length in total_length_library.items():
    print(f"Group {group}: {length:.2f} mm")

# Step 5: Calculate the number of special lengths for the first rebar group
first_group = list(total_length_library.keys())[0]  # Get the first rebar group
first_group_total_length = total_length_library[first_group]  # Retrieve total length for the first group

if pd.isna(first_group_total_length) or first_group_total_length == 0:
    raise ValueError("The total length for the first rebar group is invalid (NaN or 0). Please check the input data.")

number_of_special_lengths = math.ceil(first_group_total_length / max_rebar_length)
print(f"\nNumber of special lengths required for Group {first_group}: {number_of_special_lengths}")

# Step 6: Adjust total lapping length for the first group
lapping_length = sheet1_library.get('lapping length', 0)
adjusted_total_lapping_length = lapping_length * (number_of_special_lengths - 1)

# Step 7: Recalculate adjusted total length for the first group
adjusted_total_length = total_floor_height[0] + dowel_length[0] + hook_anchorage_length[0] + adjusted_total_lapping_length - top_girder_depth[0]

print(f"\nAdjusted Total Lapping Length for Group {first_group}: {adjusted_total_lapping_length:.2f} mm")
print(f"Adjusted Total Length for Group {first_group}: {adjusted_total_length:.2f} mm")

# Step 8: Calculate the purchasable special length

# Step 8.1: Calculate the required special length (before ceiling)
required_special_length = adjusted_total_length / number_of_special_lengths
print(f"Required Special Length for Group {first_group}: {required_special_length:.0f} mm")

# Step 8.2: Calculate the purchasable special length (after ceiling)
purchasable_special_length = math.ceil(required_special_length / 100) * 100  # rounded up to the nearest 100 mm

print(f"Purchasable Special Length for Group {first_group}: {purchasable_special_length:.0f} mm")

# Step 9: Adjust total length and calculate number of rebars for all groups
print("\nAdjusted Total Lengths and Number of Rebars for All Groups:")
rebar_summary = {}
purchasable_special_lengths = {}
updated_purchasable_lengths = {}

for i, group in enumerate(rebar_groups):
    # Retrieve total length for the current group
    total_length = total_length_library[group]  # Corrected
    
    # Calculate the number of special lengths required for the current group
    group_number_of_special_lengths = math.ceil(total_length / max_rebar_length)
    
    # Adjust total lapping length based on the number of special lengths
    adjusted_lapping_length = lapping_length * (group_number_of_special_lengths - 1)

    # Adjusted total length for the current group
    adjusted_total_length = total_floor_height[i] + dowel_length[i] + hook_anchorage_length[i] + adjusted_lapping_length - top_girder_depth[i]
    purchasable_special_lengths[group] = purchasable_special_length
    updated_purchasable_lengths[group] = purchasable_special_length
    
    # Calculate the number of rebars (rounded up)
    number_of_rebars = math.ceil(adjusted_total_length / purchasable_special_length)
    
    # Store results
    rebar_summary[group] = {
        'Adjusted Total Length': adjusted_total_length, 
        'Number of Rebars': number_of_rebars,
        'Group Special Lengths': group_number_of_special_lengths  # Store for reference
    }
    
    # Print results
    print(f"Group {group}:")
    print(f"  Adjusted Total Length: {adjusted_total_length:.2f} mm")
    print(f"  Number of Rebars: {number_of_rebars}")
 
# Final summary
print("\nRebar Group Summary:")
for i, (group, values) in enumerate(rebar_summary.items()):
    # Base summary details
    print(f"Group {group} - Adjusted Total Length: {values['Adjusted Total Length']:.2f} mm, "
          f"Number of Rebars: {values['Number of Rebars']}")

    if i > 0:  # For groups other than the first group
        number_of_special_length_rebars = values['Number of Rebars'] - 1
        number_of_residual_rebars = 1
        
        # Add information about special and residual rebar counts
        print(f"  Number of Special Length Rebars: {number_of_special_length_rebars}")
        print(f"  Number of Residual Rebars: {number_of_residual_rebars}")

# Step 10: Calculate residual lengths and update purchasable + required special length
print("\nUpdated Residual Lengths for Remaining Rebar Groups with Market-Length Constraint:")

remaining_rebar_groups = {}

for i, group in enumerate(rebar_groups):
    if i == 0:  # Skip the first group
        continue
    
    adjusted_total_length = rebar_summary[group]['Adjusted Total Length']
    number_of_rebars = rebar_summary[group]['Number of Rebars']
    
    # Calculate initial residual length
    residual_length = adjusted_total_length - ((number_of_rebars - 1) * purchasable_special_length)
    
    if residual_length < (max_rebar_length - purchasable_special_length):
        print(f"Group {group}: Residual length ({residual_length:.2f} mm) is too small. Adjusting calculations...")
        
        # Update both values to maintain consistency
        updated_purchasable_length = max_rebar_length
        updated_required_length = max_rebar_length  
        
        number_of_rebars = math.ceil(adjusted_total_length / updated_purchasable_length)
        residual_length = adjusted_total_length - ((number_of_rebars - 1) * updated_purchasable_length)

        # Store updates
        rebar_summary[group]['Number of Rebars'] = number_of_rebars
        rebar_summary[group]['Updated Purchasable Length'] = updated_purchasable_length
        rebar_summary[group]['Updated Required Length'] = updated_required_length  # New addition
        updated_purchasable_lengths[group] = updated_purchasable_length
    
    # Save residual length data
    number_of_rebars_in_bundle = sheet2.loc[sheet2['Rebar group'] == group, 'No of rebar in bundle'].values[0]
    total_residual_lengths = number_of_rebars_in_bundle * 1
    
    if residual_length in remaining_rebar_groups:
        remaining_rebar_groups[residual_length] += total_residual_lengths
    else:
        remaining_rebar_groups[residual_length] = total_residual_lengths

    print(f"Group {group}:")
    print(f"  Updated Purchasable Length: {rebar_summary[group].get('Updated Purchasable Length', purchasable_special_length):.2f} mm")
    print(f"  Updated Required Length: {rebar_summary[group].get('Updated Required Length', required_special_length):.2f} mm")  # New printout
    print(f"  Residual Length: {residual_length:.2f} mm")
    print(f"  Number of Rebars: {number_of_rebars}")
    print(f"  Total Number of Residual Lengths: {total_residual_lengths}")

# Step 11: Display summary table for all rebar groups

# Retrieve the rebar unit weight (kg/m) from the library (Sheet 1)
rebar_unit_weight = sheet1_library.get('rebar unit weight', None)
if rebar_unit_weight is None:
    raise ValueError("Error: Rebar Unit Weight is missing from the library.")

# Purchasable special length (convert from mm to meters)
purchasable_length_m = purchasable_special_length / 1000  # Convert to meters

# Create PrettyTable instance for the summary table
summary_table = PrettyTable()

# Define the table's header with the new "Total Required Quantity" column
summary_table.field_names = [
    "Rebar Group", 
    "Purchasable Special Length (mm)", 
    "Number of Special Length Rebars", 
    "Total Special Length Rebars", 
    "Total Required Quantity/Weight (kg)",  # New column
    "Total Purchased Quantity/Weight (kg)"  # Revised column
]

# Prepare summary data (same as before)
summary_data = []  # Store data for Excel export
total_special_length_rebars = 0
total_required_quantity = 0
total_purchased_quantity = 0


for i, (group, values) in enumerate(rebar_summary.items()):
    adjusted_total_length = values['Adjusted Total Length']
    number_of_rebars = values['Number of Rebars']
    
    number_of_rebars_in_current_group = sheet2.loc[sheet2['Rebar group'] == group, 'No of rebar in bundle'].values[0]
    purchasable_length = updated_purchasable_lengths.get(group, purchasable_special_length)
    
    # Apply the correct required length:
    if i == 0:  
        # Group 1 always uses the required special length
        required_length = required_special_length
    else:
        # Use the updated required length if available
        required_length = values.get('Updated Required Length', required_special_length)

    # Calculate total quantities
    number_of_special_length_rebars = number_of_rebars - 1 if i > 0 else number_of_rebars
    total_special_length = number_of_special_length_rebars * number_of_rebars_in_current_group
    required_quantity = total_special_length * rebar_unit_weight * (required_length / 1000)
    purchased_quantity = total_special_length * rebar_unit_weight * (purchasable_length / 1000)
    
    summary_table.add_row([
        group,
        purchasable_length,
        number_of_special_length_rebars,
        total_special_length,
        round(required_quantity, 2),
        round(purchased_quantity, 2)
    ])
    
    summary_data.append([
        group,
        purchasable_length,
        number_of_special_length_rebars,
        total_special_length,
        round(required_quantity, 2),
        round(purchased_quantity, 2)
    ])
    
    total_special_length_rebars += total_special_length
    total_required_quantity += required_quantity
    total_purchased_quantity += purchased_quantity

# Add final total row to summary table
summary_table.add_row([
    "Total",
    "-",  # No sum for special length column
    "-",  # No sum for this column
    total_special_length_rebars,
    round(total_required_quantity, 2),
    round(total_purchased_quantity, 2)
])


# Print the PrettyTable in the console
print("\nRebar Summary (Visualized):")
print(summary_table)  # Ensures the table is displayed

# Convert summary data to a Pandas DataFrame
summary_df = pd.DataFrame(summary_data, columns=summary_table.field_names)
# Append total row to DataFrame
summary_df.loc[len(summary_df)] = ["Total", "-", "-", total_special_length_rebars, round(total_required_quantity, 2), round(total_purchased_quantity, 2)]


# Define file path for saving
summary_file_path =  r"C:\Users\Daniel\OneDrive\Desktop\Rebar_Summary_Column_LSP.xlsx"

# Save the DataFrame to an Excel file
summary_df.to_excel(summary_file_path, index=False)

# Confirm that the file was saved
print(f"\nRebar summary saved to: {summary_file_path}")

# Step 11b: Save residual lengths and total numbers to an Excel file

# Create a PrettyTable for visualizing residual lengths
residual_table = PrettyTable()
residual_table.field_names = ["Length", "Quantity"]  # Corrected field names

# Prepare data for Excel
residual_data = []

# Iterate through remaining rebar groups and collect residual lengths
for length, quantity in remaining_rebar_groups.items():
    residual_table.add_row([length / 1000, quantity])  # Add to PrettyTable
    residual_data.append([length / 1000, quantity])  # Add to list for Excel

# Print the PrettyTable in the console
print("\nResidual Lengths Summary:")
print(residual_table)

# Convert residual data to a Pandas DataFrame
residual_df = pd.DataFrame(residual_data, columns=["Length", "Quantity"])  # Corrected headers

# Define file path for saving
residual_file_path = r"C:\Users\Daniel\OneDrive\Desktop\Residual_Lengths.xlsx"

# Save to Excel
residual_df.to_excel(residual_file_path, index=False)

# Confirm file save
print(f"\nResidual lengths saved to: {residual_file_path}")

#Step 12: Cutting Pattern  Optimization
# Load discontinuous rebar lengths from the file
discontinuous_file = r"C:\Users\Daniel\OneDrive\Desktop\Residual_Lengths.xlsx"
data = pd.read_excel(discontinuous_file)

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

purchased_quantity_cutting = round(rebar_unit_weight * optimal_stock_length * total_bars_needed,2)
required_quantity_cutting = round(purchased_quantity_cutting - (rebar_unit_weight * total_waste),2)
print(f"Required quantity: {round(required_quantity_cutting, 2)} kg")
print(f"Purchased quantity: {purchased_quantity_cutting} kg")
        
# Save the cutting pattern to an Excel file
output_file = r"C:/Users/Daniel/OneDrive/Desktop/Optimized_Cutting_Pattern.xlsx" 
cutting_df.to_excel(output_file, index=False)
print(f"\nCutting pattern saved to: {output_file}")