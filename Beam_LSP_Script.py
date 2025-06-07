# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:01:45 2024

@author: Daniel
"""
import pandas as pd
import math
from prettytable import PrettyTable

# File path
file_path = r"C:\Users\Daniel\OneDrive\Desktop/Beam LSP case.xlsx"

# Step 1: Read both sheets
sheet1 = pd.read_excel(file_path, sheet_name=0)  # First sheet
sheet2 = pd.read_excel(file_path, sheet_name=1)  # Second sheet

# Step 2: Create a library from Sheet 1
sheet1['Description'] = sheet1['Description'].str.strip().str.lower()  # Normalize descriptions
sheet1_library = dict(zip(sheet1['Description'], sheet1['Contents']))  # Create dictionary

# Debug: Display the library
print("Library from Sheet 1:")
for key, value in sheet1_library.items():
    print(f"{key}: {value}")

# Step 3: Retrieve Maximum Rebar Length from the library
max_rebar_length = sheet1_library.get('maximum rebar length', 12000.0)  # Default to 12000.0 if missing

if pd.isna(max_rebar_length):
    print("Warning: 'Maximum Rebar Length' is missing or NaN. Using default value of 12000.0 mm.")
    max_rebar_length = 12000.0

print(f"\nMaximum Rebar Length: {max_rebar_length:.2f} mm")

# Step 4: Calculate total lengths and number of special length rebars for all sections
sheet2 = sheet2.fillna(0)  # Replace NaN with 0

sections = sheet2['Section']  # List of sections
total_span_length = sheet2['Total span length']
hook_anchorage_length = sheet2['Hook anchorage length']
total_lapping_length = sheet2['Total lapping length']
column_width_left = sheet2['Column width at the left end']
column_width_right = sheet2['Column width at the right end']
number_of_rebars_in_bundle = int(sheet2['No of rebar in bundle'].iloc[0])

# Dictionaries to store calculations
total_length_library = {}
number_of_special_length = {}

# Calculate total lengths and number of special lengths
for i in range(len(sections)):
    total_length = (
        total_span_length[i]
        + (2 * hook_anchorage_length[i])
        + total_lapping_length[i]
        - ((column_width_left[i] + column_width_right[i]) / 2)
    )
    total_length_library[i] = total_length
    number_of_special_length[i] = math.ceil(total_length / max_rebar_length)

# Display results
print("\nTotal Rebar Lengths and Number of Special Length Rebars:")
for i in range(len(sections)):
    print(f"Section {sections[i]}: Length = {total_length_library[i]:.2f} mm, "
          f"Special Rebars = {number_of_special_length[i]}")

# Step 5: Adjust total lapping length for all sections
lapping_length = sheet1_library.get('lapping length', 0)  # Default to 0 if missing
adjusted_lapping_length = {}

for i in range(len(sections)):
    adjusted_lapping_length[i] = (number_of_special_length[i] - 1) * lapping_length

# Display adjusted lapping lengths
print("\nAdjusted Lapping Lengths:")
for i in range(len(sections)):
    print(f"Section {sections[i]}: {adjusted_lapping_length[i]:.2f} mm")

# Step 6: Recalculate adjusted total lengths
adjusted_total_length = {}

for i in range(len(sections)):
    adjusted_total_length[i] = (
        total_span_length[i]
        + (2 * hook_anchorage_length[i])
        + adjusted_lapping_length[i]
        - ((column_width_left[i] + column_width_right[i]) / 2)
    )

# Display adjusted total lengths
print("\nAdjusted Total Lengths:")
for i in range(len(sections)):
    print(f"Section {sections[i]}: {adjusted_total_length[i]:.2f} mm")

# Step 7: Calculate required and purchasable special lengths
required_special_length = {}
purchasable_special_length = {}

for i in range(len(sections)):
    required_special_length[i] = adjusted_total_length[i] / number_of_special_length[i]
    purchasable_special_length[i] = math.ceil(required_special_length[i] / 100) * 100

# Display required and purchasable special lengths
print("\nRequired and Purchasable Special Lengths:")
for i in range(len(sections)):
    print(
        f"Section {sections[i]}: "
        f"Required = {required_special_length[i]:.2f} mm, "
        f"Purchasable = {purchasable_special_length[i]:.2f} mm"
    )

# Step 8: Calculate required and purchasable weights for each section 

# Retrieve unit weight of rebar from the library (default to 3.04 kg/m if missing)
rebar_unit_weight = sheet1_library.get('rebar unit weight', 3.04)  # Default unit weight for 22 mm rebar
if pd.isna(rebar_unit_weight):
    print("Warning: 'Rebar Unit Weight' value is missing or NaN. Using a default value of 3.04 kg/m.")
    rebar_unit_weight = 3.04

# Initialize dictionaries for total number of rebars, required weights, and purchasable weights
total_number_of_special_lengths = {}
required_weight = {}
purchasable_weight = {}

# Calculate total rebars, required weights, and purchasable weights
for i in range(len(sections)):
    # Retrieve the number of rebars in a bundle for the current section
    rebars_in_bundle = sheet2.loc[i, 'No of rebar in bundle']
    
    # Handle missing or invalid values
    if pd.isna(rebars_in_bundle) or rebars_in_bundle == 0:
        print(f"Warning: Missing or invalid 'Number of rebars in bundle' for section {sections[i]}. Defaulting to 1.")
        rebars_in_bundle = 1  # Default to 1 if no data available

    # Total number of special lengths for the section
    total_number_of_special_lengths[i] = (
        number_of_special_length[i] * rebars_in_bundle
    )

    # Required weight for the section
    required_weight[i] = (
        rebar_unit_weight
        * (required_special_length[i] / 1000)  # Convert mm to meters
        * total_number_of_special_lengths[i]
    )

    # Purchasable weight for the section
    purchasable_weight[i] = (
        rebar_unit_weight
        * (purchasable_special_length[i] / 1000)  # Convert mm to meters
        * total_number_of_special_lengths[i]
    )

# Display the calculated weights
print("\nRequired and Purchasable Weights for Each Section:")
for i in range(len(sections)):
    print(
        f"Section {sections[i]}: "
        f"Required Weight = {required_weight[i]:.2f} kg, "
        f"Purchasable Weight = {purchasable_weight[i]:.2f} kg"
    )

# Summary Table
summary_table = PrettyTable()
summary_table.field_names = [
    "Section",
    "Purchasable Special Length (mm)",
    "Total Number of Special Length Rebars",
    "Total Required Quantity (kg)",
    "Total Purchased Quantity (kg)",
]

summary_data =[]
total_special_rebar = 0
total_required_weight= 0
total_purchasable_weight = 0

for i, section in enumerate(sections):
    summary_table.add_row([
        section,
        purchasable_special_length[i],
        total_number_of_special_lengths[i],
        round(required_weight[i], 2),
        round(purchasable_weight[i], 2),
    ])
    
    summary_data.append([
        section,
        purchasable_special_length[i],
        total_number_of_special_lengths[i],
        round(required_weight[i], 2),
        round(purchasable_weight[i], 2),
        ])
    
    total_special_rebar += total_number_of_special_lengths[i]
    total_required_weight += required_weight[i]
    total_purchasable_weight += purchasable_weight [i]
    
# Add total row to summary table
summary_table.add_row([
    "Total",
    "-",
    total_special_rebar, 
    round(total_required_weight, 2),
    round(total_purchasable_weight, 2),
])

print("\nSummary Table:")
print(summary_table)

output_file_summary = r"C:\Users\Daniel\OneDrive\Desktop/Rebar_Summary_Beam_LSP.xlsx"

summary_df = pd.DataFrame(summary_data, columns=[
    "Section",
    "Purchasable Special Length (mm)",
    "Total Number of Special Length Rebars",
    "Total Required Quantity (kg)",
    "Total Purchased Quantity (kg)",
])

# Append total row to DataFrame
summary_df.loc[len(summary_df)] = ["Total", "-", total_special_rebar, round(total_required_weight, 2), round(total_purchasable_weight, 2)]

summary_df.to_excel(output_file_summary, index=False)

print("Rebar summary saved to", output_file_summary)

#Step 9: Optimization for discontinuous rebars
import numpy as np
import random
# Load discontinuous rebar lengths from the file
discontinuous_file = r"C:/Users/Daniel/OneDrive/Desktop/Discontinuous rebar_beam.xlsx"
data = pd.read_excel(discontinuous_file)

# Ensure the required columns exist
if 'Length' not in data.columns or 'Quantity' not in data.columns:
    raise ValueError("Discontinuous file must contain 'Length (m)' and 'Quantity' columns.")

# Convert lengths and quantities
required_lengths = data['Length'].tolist()
required_quantities = data['Quantity'].tolist()

# Display input data
print("\n--- Required Rebar Data (from Discontinuous) ---")
print(data.to_string(index=False))

def find_best_combination(stock_length, required_lengths, quantities):
    """Find the best cutting combination to minimize waste."""
    best_combination = []
    min_waste = stock_length
    
    sorted_indices = sorted(range(len(required_lengths)), key=lambda k: required_lengths[k], reverse=True)
    
    def backtrack(index, current_combination, remaining_length):
        nonlocal best_combination, min_waste
        
        if remaining_length < min_waste:
            best_combination = current_combination.copy()
            min_waste = remaining_length
        
        if index == len(required_lengths):
            return
        
        for i in range(index, len(required_lengths)):
            length = required_lengths[sorted_indices[i]]
            qty = quantities[sorted_indices[i]]
            
            if length <= remaining_length and qty > 0:
                current_combination.append(length)
                quantities[sorted_indices[i]] -= 1
                backtrack(i, current_combination, remaining_length - length)
                current_combination.pop()
                quantities[sorted_indices[i]] += 1
    
    backtrack(0, [], stock_length)
    return best_combination, min_waste

def calculate_waste(stock_length, required_lengths, quantities):
    """Calculate total waste and determine the cutting pattern."""
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
    """Evaluate the fitness of a stock length (minimizing waste)."""
    if any(stock_length < length for length in required_lengths):
        return float('inf')  # Infeasible solution
    
    total_waste, _, _, _ = calculate_waste(stock_length, required_lengths, required_quantities.copy())
    penalty_factor = sum(q for q in required_quantities if q > 0) * stock_length  # Penalize uncut lengths

    return total_waste + penalty_factor

# WOA Parameters
num_whales = 50
num_iterations = 100

# Initialize whale positions within stock length range
whales = np.random.uniform(6.0, 12.0, num_whales)
best_whale = whales[0]
best_fitness = fitness_function(best_whale)

for iteration in range(num_iterations):
    a = 2 - iteration * (2 / num_iterations)  # Linearly decreasing factor
    
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
        
        fitness = fitness_function(whales[i])
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_whale = whales[i]
    
    print(f"Iteration {iteration + 1}/{num_iterations}, Best Waste: {best_fitness:.2f} m")

optimal_stock_length = round(best_whale, 1)
print(f"\nBest Rebar Length: {optimal_stock_length} m")

# Compute the final cutting pattern
total_waste, total_bars_needed, cutting_pattern, remaining_quantities = calculate_waste(
    optimal_stock_length, required_lengths, required_quantities.copy())

# Display final results
print(f"Total Waste: {round(total_waste, 2)} m")
print(f"Total Bars Needed: {total_bars_needed}")

cutting_df = pd.DataFrame(cutting_pattern)
print("\nCutting Pattern:")
print(cutting_df.to_string(index=False))

print("\nRemaining Quantities:")
for length, quantity in zip(required_lengths, remaining_quantities):
    if quantity > 0:
        print(f"Length {round(length, 2)}m: {quantity}")

purchased_quantity_cutting = rebar_unit_weight * optimal_stock_length * total_bars_needed
required_quantity_cutting = purchased_quantity_cutting - (rebar_unit_weight * total_waste)
print(f"Required quantity: {round(required_quantity_cutting, 2)} kg")
print(f"Purchased quantity: {round(purchased_quantity_cutting, 2)} kg")

# Save the cutting pattern to an Excel file
output_file = r"C:/Users/Daniel/OneDrive/Desktop/Optimized_Cutting_Pattern_Beam_LSP.xlsx"
cutting_df.to_excel(output_file, index=False)
print(f"\nCutting pattern saved to: {output_file}")