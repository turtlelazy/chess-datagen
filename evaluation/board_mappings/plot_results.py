import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import sys

def plot_score_distribution(df, output_file="score_distribution.png"):
    # Define bucket ranges
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.0–0.5', '0.5–0.6', '0.6–0.7', '0.7–0.8', '0.8–0.9', '0.9–1.0']

    # Create buckets
    df['Bucket'] = pd.cut(df['passed'], bins=bins, labels=labels, include_lowest=True)

    # Count the number of scores in each bucket
    bucket_counts = df['Bucket'].value_counts().sort_index()
    print(bucket_counts)

    # Plot
    plt.figure(figsize=(8, 5))
    bucket_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Score Distribution by Bucket')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save to file
    plt.savefig(output_file, dpi=300)

def plot_score_distribution_percentage(df, output_file="score_distribution_percentage.png"):
    # Define bucket ranges
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['0.0–0.5', '0.5–0.6', '0.6–0.7', '0.7–0.8', '0.8–0.9', '0.9–1.0']

    # Create buckets
    df['Bucket'] = pd.cut(df['passed'], bins=bins, labels=labels, include_lowest=True)

    # Count the number of scores in each bucket
    bucket_counts = df['Bucket'].value_counts().sort_index()

    # Calculate percentages
    bucket_percentages = (bucket_counts / bucket_counts.sum()) * 100
    print(bucket_percentages)

    # Plot as percentages
    plt.figure(figsize=(8, 5))
    bucket_percentages.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Score Distribution by Bucket (%)')
    plt.xlabel('Score Range')
    plt.ylabel('Percentage of Images')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save percentage plot
    plt.savefig(output_file, dpi=300)

def angle_from_xy(rotation_matrix):
    """
    Calculates the angle (in degrees) between the camera direction and the XY plane.
    Assumes the Z-component of the forward direction is at rot[2][2].
    """
    forward_z = rotation_matrix[2][2]
    forward_z = np.clip(forward_z, -1.0, 1.0)
    angle_from_z = np.arccos(forward_z) * 180 / np.pi
    angle_from_xy = 90 - angle_from_z
    return angle_from_xy

def plot_pass_rate_by_angle_bin(json_file, csv_file, output_file="pass_rate_by_angle.png"):
    # Load CSV
    score_df = pd.read_csv(csv_file)
    score_df['file_name'] = score_df['file_name'].apply(lambda x: x.split('/')[-1])  # normalize for merging

    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Collect (filename, angle_from_xy)
    angle_data = []
    for img_path, entry in data.items():
        img_name = img_path.split('/')[-1]  # extract file name only
        rot = entry["cam"]["rot"]
        angle = angle_from_xy(rot)
        angle_data.append((img_name, angle))

    angle_df = pd.DataFrame(angle_data, columns=['file_name', 'Angle'])
    print(angle_df)
    # Merge with score_df
    merged = score_df.merge(angle_df, on='file_name')

    # Convert 'passed' to boolean if not already
    merged['passed'] = merged['passed'].astype(bool)

    # Bin by angle (10° increments)
    bins = np.arange(0, 100, 10)
    labels = [f"{i}–{i+10}" for i in bins[:-1]]
    merged['AngleBin'] = pd.cut(merged['Angle'], bins=bins, labels=labels, include_lowest=True)

    # Group by angle bin and compute average passed rate
    avg_pass_rate = merged.groupby('AngleBin')['passed'].mean() * 100  # convert to percentage

    # Plot
    plt.figure(figsize=(8, 5))
    avg_pass_rate.plot(kind='bar', color='mediumseagreen', edgecolor='black')
    plt.title('Pass Rate by Camera Angle from XY Plane')
    plt.xlabel('Camera Angle (°)')
    plt.ylabel('Pass Rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)




if __name__ == "__main__":
    # Load CSV
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py <csv_file> <json_file>")
        sys.exit(1)

    csv = sys.argv[1]
    json_file = sys.argv[2]
    
    df = pd.read_csv(csv)

    # plot_score_distribution(df)
    # plot_score_distribution_percentage(df)
    plot_pass_rate_by_angle_bin(json_file, csv)


