# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:01:45 2024

@author: Daniel
"""
import pandas as pd
import math
from prettytable import PrettyTable

# File path
file_path = r"C:\Users\Daniel\OneDrive\Desktop/Wall LSP case.xlsx"

# Step 1: Read both sheets
sheet1 = pd.read_excel(file_path, sheet_name=0)  # First sheet
sheet2 = pd.read_excel(file_path, sheet_name=1)  # Second sheet

# Step 2: Create a library from Sheet 1
# Standardize 'Description' for consistency (strip spaces, lowercase)
sheet1['Description'] = sheet1['Description'].str.strip().str.lower()

# Build a dictionary with 'Description' as keys and 'Contents' as values
sheet1_library = dict(zip(sheet1['Description'], sheet1['Contents']))

# Debug: Display the library
print("Library from Rebar Specifications:")
for key, value in sheet1_library.items():
    print(f"{key}: {value}")

# Step 3: Retrieve Maximum Rebar Length from the library
max_rebar_length = sheet1_library.get('maximum rebar length')

# Check if the value is available; if not, print a warning but don't stop the program
if pd.isna(max_rebar_length):
    print("Warning: 'Maximum Rebar Length' value is missing or NaN. Using a default value of 12000.0 meters.")
    max_rebar_length = 12000.0  # Default value

print(f"\nMaximum rebar length retrieved: {max_rebar_length:.2f} mm")

# Step 4: Calculate total lengths and number of special length rebars for all rebar groups and store in a library
# Replace NaN values with 0 before calculation
sheet2 = sheet2.fillna(0)  # Replace NaN with 0 in the entire sheet

wall_types = sheet2['Wall type']  # Assuming a wall exists to identify rebar groups
total_floor_height = sheet2['Total floor height']
net_wall_span = sheet2['Net wall span']
dowel_length = sheet2['Dowel length']
hook_anchorage_length_vertical = sheet2['Hook anchorage length vertical']
hook_anchorage_length_horizontal = sheet2['Hook anchorage length horizontal']
total_lapping_length_vertical = sheet2['Total lapping length vertical']
total_lapping_length_horizontal = sheet2['Total lapping length horizontal']
top_slab_depth = sheet2['Top slab depth']

# Initialize dictionaries to store total lengths
total_length_vertical_library = {}
total_length_horizontal_library = {}
number_of_special_lengths_vertical = {}
number_of_special_lengths_horizontal = {}

# Calculate total length for each type
for i in range(len(wall_types)):
    # Vertical lengths
    total_length_vertical = (
        total_floor_height[i] 
        + dowel_length[i] 
        + hook_anchorage_length_vertical[i] 
        + total_lapping_length_vertical[i] 
        - top_slab_depth[i]
    )
    total_length_vertical_library[wall_types[i]] = total_length_vertical
    
    # Horizontal lengths
    total_length_horizontal = (
        net_wall_span[i] 
        + (2 * hook_anchorage_length_horizontal[i]) 
        + total_lapping_length_horizontal[i]
    )
    total_length_horizontal_library[wall_types[i]] = total_length_horizontal
    
    # Calculate number of special lengths (ceil division by max rebar length)
    number_of_special_lengths_vertical[wall_types[i]] = math.ceil(total_length_vertical / max_rebar_length)
    number_of_special_lengths_horizontal[wall_types[i]] = math.ceil(total_length_horizontal / max_rebar_length)

# Display the total lengths
print("\nTotal Vertical Rebar Lengths for All Wall Types:")
for type, length in total_length_vertical_library.items():
    print(f"Type {type}: {length:.2f} mm")

print("\nTotal Horizontal Rebar Lengths for All Wall Types:")
for type, length in total_length_horizontal_library.items():
    print(f"Type {type}: {length:.2f} mm")

# Display the number of special lengths
print("\nNumber of Special Length Rebars for Each Wall Type (Vertical):")
for type, count in number_of_special_lengths_vertical.items():
    print(f"Type {type}: {count} rebars")

print("\nNumber of Special Length Rebars for Each Wall Type (Horizontal):")
for type, count in number_of_special_lengths_horizontal.items():
    print(f"Type {type}: {count} rebars")

# Step 5: Adjust total lapping length for all types
lapping_length = sheet1_library.get('lapping length')

# Initialize dictionaries for adjusted lapping lengths
adjusted_lapping_length_vertical = {}
adjusted_lapping_length_horizontal = {}

# Calculate adjusted lapping lengths for each wall type
for wall_type in wall_types:
    # Vertical direction
    adjusted_lapping_length_vertical[wall_type] = (
        (number_of_special_lengths_vertical[wall_type] - 1) * lapping_length
    )
    
    # Horizontal direction
    adjusted_lapping_length_horizontal[wall_type] = (
        (number_of_special_lengths_horizontal[wall_type] - 1) * lapping_length
    )

# Display the adjusted lapping lengths
print("\nAdjusted Lapping Lengths for Each Wall Type (Vertical):")
for type, length in adjusted_lapping_length_vertical.items():
    print(f"Type {type}: {length:.2f} mm")

print("\nAdjusted Lapping Lengths for Each Wall Type (Horizontal):")
for type, length in adjusted_lapping_length_horizontal.items():
    print(f"Type {type}: {length:.2f} mm")

# Step 6: Recalculate adjusted total length for all types

# Initialize dictionaries for adjusted total lengths
adjusted_total_length_vertical = {}
adjusted_total_length_horizontal = {}

# Calculate adjusted total lengths for each wall type
for i, wall_type in enumerate(wall_types):
    # Vertical direction
    adjusted_total_length_vertical[wall_type] = (
        total_floor_height[i] 
        + dowel_length[i] 
        + hook_anchorage_length_vertical[i] 
        + adjusted_lapping_length_vertical[wall_type] 
        - top_slab_depth[i]
    )
    
    # Horizontal direction
    adjusted_total_length_horizontal[wall_type] = (
        net_wall_span[i] 
        + (2 * hook_anchorage_length_horizontal[i]) 
        + adjusted_lapping_length_horizontal[wall_type]
    )

# Display the adjusted total lengths
print("\nAdjusted Total Lengths for Each Wall Type (Vertical):")
for type, length in adjusted_total_length_vertical.items():
    print(f"Type {type}: {length:.2f} mm")

print("\nAdjusted Total Lengths for Each Wall Type (Horizontal):")
for type, length in adjusted_total_length_horizontal.items():
    print(f"Type {type}: {length:.2f} mm")
    
# Step 7: Calculate required and purchasable special lengths for both directions

# Initialize dictionaries for required and purchasable special lengths
required_special_length_vertical = {}
purchasable_special_length_vertical = {}
required_special_length_horizontal = {}
purchasable_special_length_horizontal = {}

# Calculate the lengths
for wall_type in wall_types:
    # Vertical direction
    required_special_length_vertical[wall_type] = (
        adjusted_total_length_vertical[wall_type] / number_of_special_lengths_vertical[wall_type]
    )
    purchasable_special_length_vertical[wall_type] = math.ceil(required_special_length_vertical[wall_type] / 100) * 100
    
    # Horizontal direction
    required_special_length_horizontal[wall_type] = (
        adjusted_total_length_horizontal[wall_type] / number_of_special_lengths_horizontal[wall_type]
    )
    purchasable_special_length_horizontal[wall_type] = math.ceil(required_special_length_horizontal[wall_type] / 100) * 100

# Display the results
print("\nRequired and Purchasable Special Lengths for Each Wall Type (Vertical):")
for type in wall_types:
    print(f"Type {type}: Required Length = {required_special_length_vertical[type]:.2f} mm, Purchasable Length = {purchasable_special_length_vertical[type]} mm")

print("\nRequired and Purchasable Special Lengths for Each Wall Type (Horizontal):")
for type in wall_types:
    print(f"Type {type}: Required Length = {required_special_length_horizontal[type]:.2f} mm, Purchasable Length = {purchasable_special_length_horizontal[type]} mm")

# Step 8: Calculate required and purchasable weights for each type in both directions

# Retrieve unit weight of rebar from the library (convert to kg/m if necessary)
rebar_unit_weight = sheet1_library.get('rebar unit weight')
if pd.isna(rebar_unit_weight):
    print("Warning: 'Rebar Unit Weight' value is missing or NaN. Using a default value of 0.785 kg/m.")
    rebar_unit_weight = 0.56  # Default unit weight for 10 mm rebar in kg/m

# Initialize dictionaries for total number of rebars, required weights, and purchasable weights
total_number_of_special_lengths_vertical = {}
total_number_of_special_lengths_horizontal = {}
required_weight_vertical = {}
purchasable_weight_vertical = {}
required_weight_horizontal = {}
purchasable_weight_horizontal = {}

# Calculate total rebars, required, and purchasable weights
for i, wall_type in enumerate(wall_types):
    # Retrieve number of rebars in bundle for vertical and horizontal directions
    rebars_in_bundle_vertical = sheet2.loc[i, 'No of rebar in bundle vertical']
    rebars_in_bundle_horizontal = sheet2.loc[i, 'No of rebar in bundle horizontal']
    
    # Handle missing or invalid values
    if pd.isna(rebars_in_bundle_vertical) or rebars_in_bundle_vertical == 0:
        print(f"Warning: Missing or invalid 'Number of rebars in bundle (Vertical)' for wall type {wall_type}. Defaulting to 1.")
        rebars_in_bundle_vertical = 1  # Default to 1 if no data available
    if pd.isna(rebars_in_bundle_horizontal) or rebars_in_bundle_horizontal == 0:
        print(f"Warning: Missing or invalid 'Number of rebars in bundle (Horizontal)' for wall type {wall_type}. Defaulting to 1.")
        rebars_in_bundle_horizontal = 1  # Default to 1 if no data available

    # Total number of special lengths
    total_number_of_special_lengths_vertical[wall_type] = (
        number_of_special_lengths_vertical[wall_type] * rebars_in_bundle_vertical
    )
    total_number_of_special_lengths_horizontal[wall_type] = (
        number_of_special_lengths_horizontal[wall_type] * rebars_in_bundle_horizontal
    )

    # Required and purchasable weights for vertical rebars
    required_weight_vertical[wall_type] = (
        rebar_unit_weight 
        * (required_special_length_vertical[wall_type] / 1000)  # Convert mm to meters
        * total_number_of_special_lengths_vertical[wall_type]
    )
    purchasable_weight_vertical[wall_type] = (
        rebar_unit_weight 
        * (purchasable_special_length_vertical[wall_type] / 1000)  # Convert mm to meters
        * total_number_of_special_lengths_vertical[wall_type]
    )

    # Required and purchasable weights for horizontal rebars
    required_weight_horizontal[wall_type] = (
        rebar_unit_weight 
        * (required_special_length_horizontal[wall_type] / 1000)  # Convert mm to meters
        * total_number_of_special_lengths_horizontal[wall_type]
    )
    purchasable_weight_horizontal[wall_type] = (
        rebar_unit_weight 
        * (purchasable_special_length_horizontal[wall_type] / 1000)  # Convert mm to meters
        * total_number_of_special_lengths_horizontal[wall_type]
    )
    
# Step 9: Summary Table for all types
# Vertical Summary Table
vertical_table = PrettyTable()
vertical_table.field_names = [
    "Wall Type",
    "Purchasable Special Length (mm)",
    "Total Number of Special Length Rebars",
    "Total Required Quantity (kg)",
    "Total Purchased Quantity (kg)",
]

vertical_data = []
sum_total_required_quantity_v = 0
sum_total_purchased_quantity_v = 0

for wall_type in wall_types:
    vertical_table.add_row([
        wall_type,
        purchasable_special_length_vertical[wall_type],
        total_number_of_special_lengths_vertical[wall_type],
        round(required_weight_vertical[wall_type], 2),
        round(purchasable_weight_vertical[wall_type], 2),
    ])

    vertical_data.append([
        wall_type,
        purchasable_special_length_vertical[wall_type],
        total_number_of_special_lengths_vertical[wall_type],
        round(required_weight_vertical[wall_type], 2),
        round(purchasable_weight_vertical[wall_type], 2),
        ]) 
    
    sum_total_required_quantity_v += required_weight_vertical[wall_type]
    sum_total_purchased_quantity_v += purchasable_weight_vertical[wall_type]
    
# Add total row to summary table
vertical_table.add_row([
    "Total",
    "-",
    "-",
    round(sum_total_required_quantity_v, 2),
    round(sum_total_purchased_quantity_v, 2)
]) 
   
print("\nVertical Summary Table:")
print(vertical_table)

# Horizontal Summary Table
horizontal_table = PrettyTable()
horizontal_table.field_names = [
    "Wall Type",
    "Purchasable Special Length (mm)",
    "Total Number of Special Length Rebars",
    "Total Required Quantity (kg)",
    "Total Purchased Quantity (kg)",
]

horizontal_data = []
sum_total_required_quantity_h = 0
sum_total_purchased_quantity_h = 0

for wall_type in wall_types:
    horizontal_table.add_row([
        wall_type,
        purchasable_special_length_horizontal[wall_type],
        total_number_of_special_lengths_horizontal[wall_type],
        round(required_weight_horizontal[wall_type], 2),
        round(purchasable_weight_horizontal[wall_type], 2),
    ])

    horizontal_data.append([
        wall_type,
        purchasable_special_length_horizontal[wall_type],
        total_number_of_special_lengths_horizontal[wall_type],
       round(required_weight_horizontal[wall_type], 2),
       round(purchasable_weight_horizontal[wall_type], 2),
        ]) 
    
    sum_total_required_quantity_h += required_weight_horizontal[wall_type]
    sum_total_purchased_quantity_h += purchasable_weight_horizontal[wall_type]
    
# Add total row to summary table
horizontal_table.add_row([
    "Total",
    "-",
    "-",
    round(sum_total_required_quantity_h, 2),
    round(sum_total_purchased_quantity_h, 2)
]) 

print("\nHorizontal Summary Table:")
print(horizontal_table)

# Convert the vertical summary table to a DataFrame
vertical_df = pd.DataFrame(vertical_data, columns=[
    "Wall Type",
    "Purchasable Special Length (mm)",
    "Total Number of Special Length Rebars",
    "Total Required Quantity (kg)",
    "Total Purchased Quantity (kg)"
])

# Add total row to DataFrame
total_row_v = pd.DataFrame([[
    "Total",
    "-",
    "-",
    round(sum_total_required_quantity_v, 2),
    round(sum_total_purchased_quantity_v, 2)
]], columns=vertical_df.columns)

vertical_df = pd.concat([vertical_df, total_row_v], ignore_index=True)

# Convert the horizontal summary table to a DataFrame
horizontal_df = pd.DataFrame(horizontal_data, columns=[
    "Wall Type",
    "Purchasable Special Length (mm)",
    "Total Number of Special Length Rebars",
    "Total Required Quantity (kg)",
    "Total Purchased Quantity (kg)"
])

# Add total row to DataFrame
total_row_h = pd.DataFrame([[
    "Total",
    "-",
    "-",
    round(sum_total_required_quantity_h, 2),
    round(sum_total_purchased_quantity_h, 2)
]], columns=horizontal_df.columns)

horizontal_df = pd.concat([horizontal_df, total_row_h], ignore_index=True)

# Save to Excel
output_file = "Rebar_summary_wall_LSP.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    vertical_df.to_excel(writer, sheet_name="Vertical Summary", index=False)
    horizontal_df.to_excel(writer, sheet_name="Horizontal Summary", index=False)

print(f"Rebar summary saved to {output_file}")