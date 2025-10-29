from typing import Dict, List, Annotated
import numpy as np
import os
import struct
from sklearn.cluster import KMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
N_CLUSTERS = 256          # Number of IVF clusters
N_HASH_PLANES = 8         # Number of random hyperplanes for LSH
PROBE_CLUSTERS = 10     # How many nearest clusters to search
TOP_K_DEFAULT = 5

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat",
                 new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        # Load index headers only (small metadata)
        if os.path.exists(self.index_path):
            self._load_index_header()

    # --------------------------- DB CREATION --------------------------- #
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    # --------------------------- RETRIEVAL --------------------------- #
    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            print(f"Error in get_one_row: {e}")
            return None

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=TOP_K_DEFAULT):
        """
        Disk-based retrieval using IVF + LSH hybrid:
          1. Find top PROBE_CLUSTERS nearest centroids.
          2. For each cluster, hash query and probe matching buckets.
          3. Load vectors from those buckets, compute cosine similarity.
        """
        # Step 1: load index header if not loaded
        if not hasattr(self, "centroids"):
            self._load_index_header()

        # Step 2: choose top cluster centroids by cosine sim
        sims = np.dot(self.centroids, query.T).flatten() / (
            np.linalg.norm(self.centroids, axis=1) * np.linalg.norm(query)
        )
        nearest_cluster_ids = np.argsort(-sims)[:PROBE_CLUSTERS]

        best_scores = []
        for cid in nearest_cluster_ids:
            cluster_file = os.path.join(self.index_dir, f"cluster_{cid}.bin")
            if not os.path.exists(cluster_file):
                continue
            with open(cluster_file, "rb") as f:
                # Each record = ID (int) + vector (70 * float32)
                record_size = 4 + DIMENSION * ELEMENT_SIZE
                f.seek(0, os.SEEK_END)
                num_records = f.tell() // record_size
                f.seek(0)
                for _ in range(num_records):
                    rec_id = struct.unpack("i", f.read(4))[0]
                    # vec = np.frombuffer(f.read(DIMENSION * ELEMENT_SIZE), dtype=np.float32)
                    vec = self.get_one_row(rec_id)
                    score = self._cal_score(query, vec)
                    best_scores.append((score, rec_id))

        best_scores = sorted(best_scores, key=lambda x: -x[0])[:top_k]
        return [rid for (_, rid) in best_scores]

    # --------------------------- INDEX BUILDING --------------------------- #
    def _build_index(self):
        print("Building index...")
        vectors = self.get_all_rows()
        db_size=len(vectors)
        num_records = len(vectors)
        record_ids = np.arange(num_records, dtype=np.int32)

        # KMeans clustering (IVF centroids)
        kmeans = KMeans(n_clusters= min(N_CLUSTERS, db_size // 2), random_state=DB_SEED_NUMBER, n_init=5)
        labels = kmeans.fit_predict(vectors)
        centroids = kmeans.cluster_centers_.astype(np.float32)

        # Save metadata
        self.centroids = centroids
        self.index_dir = self.index_path.replace(".dat", "_clusters")
        os.makedirs(self.index_dir, exist_ok=True)

        # Save each cluster to disk as binary file
        for cid in range(min(N_CLUSTERS, db_size // 2)):
            mask = labels == cid
            if not np.any(mask):
                continue
            cluster_vecs = vectors[mask]
            cluster_ids = record_ids[mask]
            cluster_file = os.path.join(self.index_dir, f"cluster_{cid}.bin")
            with open(cluster_file, "wb") as f:
                for rid in (cluster_ids):
                    f.write(struct.pack("i", int(rid)))
                    # f.write(vec.astype(np.float32).tobytes())

        # Save centroid header
        np.save(self.index_path.replace(".dat", "_centroids.npy"), centroids)
        print("Index built successfully.")

    def _load_index_header(self):
        self.index_dir = self.index_path.replace(".dat", "_clusters")
        centroids_path = self.index_path.replace(".dat", "_centroids.npy")
        self.centroids = np.load(centroids_path)

    # --------------------------- UTILITIES --------------------------- #
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
