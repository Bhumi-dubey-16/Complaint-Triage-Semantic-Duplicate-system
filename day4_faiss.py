import time
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS is not installed. Run 'pip install faiss-cpu' to install it.")

# 1. Generate or simulate embedding datasets of dimension 384
dim = 384
print("Generating vector datasets for speed benchmarking...")
database_10k = np.random.random((10000, dim)).astype('float32')
database_50k = np.random.random((50000, dim)).astype('float32')
query_vector = np.random.random((1, dim)).astype('float32')

# 2. Benchmark NumPy brute-force search at 10k vs 50k vectors
print("\n--- NumPy Brute-Force Benchmarking ---")
start = time.time()
np_sims_10k = np.dot(database_10k, query_vector.T)
print(f"NumPy 10,000 vectors search time: {time.time() - start:.6f} seconds")

start = time.time()
np_sims_50k = np.dot(database_50k, query_vector.T)
print(f"NumPy 50,000 vectors search time: {time.time() - start:.6f} seconds")

# 3. Benchmark FAISS Flat Index if installed
if FAISS_AVAILABLE:
    print("\n--- FAISS Flat Index Benchmarking ---")
    
    # 1. Benchmark FAISS at 10k scale
    db_10k_faiss = database_10k.copy()
    q_faiss = query_vector.copy()
    faiss.normalize_L2(db_10k_faiss)
    faiss.normalize_L2(q_faiss)
    
    index_10k = faiss.IndexFlatIP(dim)
    index_10k.add(db_10k_faiss)
    
    start = time.time()
    distances, indices = index_10k.search(q_faiss, k=5)
    print(f"FAISS Flat Index (10,000 vectors) search time: {time.time() - start:.6f} seconds")

    # 2. Benchmark FAISS at 50k scale
    db_50k_faiss = database_50k.copy()
    faiss.normalize_L2(db_50k_faiss)
    
    index_50k = faiss.IndexFlatIP(dim)
    index_50k.add(db_50k_faiss)
    
    start = time.time()
    distances, indices = index_50k.search(q_faiss, k=5)
    print(f"FAISS Flat Index (50,000 vectors) search time: {time.time() - start:.6f} seconds")