# Preprocessing of coral reef NC data (Band 350): Convert data type (float32); normalize (divide by 255);
# perform incremental PCA dimensionality reduction (incremental PCA was used due to device memory limitations)
import numpy as np
import xarray as xr
import time
from sklearn.decomposition import IncrementalPCA

train_start_time = time.time()
print(train_start_time)

# 读取数据
nc_file_path = '../data/cr_005/transect_005.nc'

dataset_img = xr.open_dataset(nc_file_path)
print(dataset_img)
print()
#
data_images = dataset_img['cube']
print(data_images)

# data_img = data_images.values
# # print(data_img.shape)
# print(type(data_img))
# print()

sliced_cube01 = data_images.values.astype(np.float32)
sliced_cube01 = sliced_cube01 / 255

sliced_cube01_2d = sliced_cube01.reshape(-1, sliced_cube01.shape[2])
print(sliced_cube01_2d.shape)
hight = sliced_cube01_2d.shape[0]

# Parameter Settings
n_components = 100  # Number of dimensions after dimensionality reduction
batch_size = 500000  # The size of each data block

# Initialize IncrementalPCA
ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

# Perform PCA dimensionality reduction in blocks
for i in range(0, hight, batch_size):
    end_index = min(i + batch_size, hight)
    print(end_index)
    batch_data = sliced_cube01_2d[i:end_index]
    ipca.partial_fit(batch_data)

print("PCA model fitting complete")

# Retrieve the dimension-reduced data (transformation matrix)
trans_size = 500000
transformed_chunks = None
for t in range(0, hight, trans_size):
    end_indext = min(t + trans_size, hight)
    print(end_indext)
    trans_data = sliced_cube01_2d[t:end_indext]
    transed_data = ipca.transform(trans_data)

    if transformed_chunks is None:
        transformed_chunks = transed_data
    else:

        transformed_chunks = np.concatenate((transformed_chunks, transed_data), axis=0)
    print(transformed_chunks.shape)

print("Data conversion complete")
transformed_pca_data = transformed_chunks.reshape(sliced_cube01.shape[0], sliced_cube01.shape[1], -1)
transformed_pca_data = transformed_pca_data.astype(np.float32)

# Create a new DataArray
new_img_pca = xr.DataArray(transformed_pca_data, dims=("Y", "X", "W"), name="cube",
    attrs={'description': 'Data after PCA', 'n_components': n_components, 'original_bands': sliced_cube01.shape[2],
           'normalization': 'Normalize to the range [0,1]'})
print(new_img_pca)

# Set compression encoding
encoding = {'cube': {'zlib': True, 'complevel': 5, 'shuffle': True, 'dtype': 'float32'}}
# Save the compressed data
output_path = '../data/cr_005/transect_005_PCA100.nc'
print(f"Save data to: {output_path}")
new_img_pca.to_netcdf(output_path, encoding=encoding)

# # Do not compress; save directly. This uses a lot of memory.
# new_img_pca.to_netcdf('../data/cr_005/transect_005_PCA100.nc')