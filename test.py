import numpy as np
# import cupy as cp
import time
from test_ddj import cosine_sim_torch, cosine_sim_slice
import torch

# Load the matrices using NumPy
recognition_embeddings = np.load('recognition_embeddings.npy')
registry_embeddings = np.load('registry_embeddings.npy')

start_time = time.time()
sim = cosine_sim_torch(recognition_embeddings, registry_embeddings)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# # Convert the matrices to CuPy arrays
# recognition_embeddings_gpu = cp.asarray(recognition_embeddings)
# registry_embeddings_gpu = cp.asarray(registry_embeddings)

# # Perform matrix multiplication using CuPy
# start_time = time.time()
# result_gpu = cp.matmul(recognition_embeddings_gpu, registry_embeddings_gpu)
# end_time = time.time()

# # Convert the result back to a NumPy array
# result = cp.asnumpy(result_gpu)

# # Print the execution time
# print(f"Execution time: {end_time - start_time} seconds")