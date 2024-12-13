"""
This script scales the X,Y,Z and X2,Y2,Z2 columns (if any) in a CSV file
from one resolution to another.
"""

import readline
from collections import namedtuple
from pathlib import Path

import pandas as pd

# Create a named tuple for 3D vectors
Vec3D = namedtuple("Vec3D", ["x", "y", "z"])


def input_or_default(prompt, value):
    response = input(f"{prompt} [{value}]: ")
    if response == "":
        response = value
    return response


def input_vec3D(prompt="", default=None):
    while True:
        s = input(prompt + (f" [{default.x}, {default.y}, {default.z}]" if default else "") + ": ")
        if s == "" and default:
            return default
        try:
            x, y, z = map(float, s.replace(",", " ").split())
            return Vec3D(x, y, z)
        except:
            print("Enter x, y, and z values separated by commas or spaces.")


def get_scale_factors(current_res, desired_res):
    """Calculate scale factors for each dimension"""
    return Vec3D(
        current_res.x / desired_res.x, current_res.y / desired_res.y, current_res.z / desired_res.z
    )


def scale_coordinates(df, scale_factors):
    """Scale the coordinates in the dataframe"""
    # Scale X, Y, Z columns
    df["X"] *= scale_factors.x
    df["Y"] *= scale_factors.y
    df["Z"] *= scale_factors.z

    # Scale X2, Y2, Z2 columns if they exist
    if all(col in df.columns for col in ["X2", "Y2", "Z2"]):
        df["X2"] *= scale_factors.x
        df["Y2"] *= scale_factors.y
        df["Z2"] *= scale_factors.z

    return df


def main():
    # Get input file path
    input_path: str | Path = ""
    while True:
        input_path = input("Enter path to input CSV file: ").strip()
        if Path(input_path).is_file():
            break
        print("File not found. Please enter a valid path.")

    # Generate default output path
    input_path = Path(input_path)
    default_output = str(input_path.parent / f"{input_path.stem}_scaled{input_path.suffix}")

    # Get current and desired resolutions
    current_res = input_vec3D("Enter current resolution (x, y, z)")
    desired_res = input_vec3D("Enter desired resolution (x, y, z)")

    # Get output path
    output_path = input_or_default("\nEnter output file path", default_output)

    try:
        # Read the CSV file
        df = pd.read_csv(input_path)

        # Calculate scale factors
        scale_factors = get_scale_factors(current_res, desired_res)

        # Scale the coordinates
        df = scale_coordinates(df, scale_factors)

        # Write the scaled data to output file
        df.to_csv(output_path, index=False)

        print(f"\nScaled coordinates have been written to: {output_path}")
        print(f"Scale factors used (current/desired):")
        print(f"X: {scale_factors.x:.4f}")
        print(f"Y: {scale_factors.y:.4f}")
        print(f"Z: {scale_factors.z:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
