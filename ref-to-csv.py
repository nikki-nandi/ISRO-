import os
import rasterio
import numpy as np
import csv
from itertools import compress
from multiprocessing import Pool, cpu_count
import time

folder_path = 'data/reflectance'
output_csv = 'data/reflectance.csv'

# Get all .tif files with full path
all_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.endswith('.tif')]

def process_file(file_path):
    try:
        with rasterio.open(file_path) as src:
            band = src.read(1)
            if src.crs is None:
                return []

            rows, cols = np.indices(band.shape)
            lons, lats = rasterio.transform.xy(src.transform, rows, cols)
            reflectance = band.ravel()
            mask = ~np.isnan(reflectance)

            if np.count_nonzero(mask) == 0:
                return []

            filename = os.path.basename(file_path)
            return list(zip(
                [filename] * np.count_nonzero(mask),
                compress(np.ravel(lats), mask),
                compress(np.ravel(lons), mask),
                compress(reflectance, mask)
            ))

    except Exception as e:
        print(f"❌ Error with {file_path}: {e}")
        return []

if __name__ == "__main__":
    start = time.time()
    print(f"\n📂 Found {len(all_files)} images. Using {cpu_count()} CPU cores.\n")

    with Pool(processes=cpu_count()) as pool:
        all_rows = pool.map(process_file, all_files)

    # Flatten and write to CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'latitude', 'longitude', 'reflectance'])
        for rows in all_rows:
            if rows:
                writer.writerows(rows)

    elapsed = time.time() - start
    print(f"\n✅ Completed in {elapsed:.2f} seconds. CSV saved to: {output_csv}")
