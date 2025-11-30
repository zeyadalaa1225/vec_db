from typing import Annotated
import numpy as np
import os
from sklearn.cluster import  MiniBatchKMeans
import heapq

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
DB_SEED_NUMBER = 42


# n_clusters based on database size
IVF_CONFIGS = {
    1_000_000: 4096,
    10_000_000: 8192,
    20_000_000: 16384
}


# n_probe based on database size
PROBE_CLUSTERS_CONFIGS = {
    1_000_000: 30,
    10_000_000: 15,
    20_000_000: 15
}

# Batch Size of vector assignment to centroid during build
CHUNK_SIZE = 5_000

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat",
                 new_db=True, db_size=None, new_index = False) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.num_records = self._get_num_records()
        os.makedirs(self.index_path, exist_ok=True)
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            self.db_size = db_size
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        if new_index:
            self._build_index()

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

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    # --------------------------- INDEX BUILD: IVF --------------------------- #
    def _choose_n_clusters(self, num_records):
        # pick nearest config
        closest = min(IVF_CONFIGS.keys(), key=lambda x: abs(x - num_records))
        return IVF_CONFIGS[closest]

    def _build_index(self):

        num_records = self._get_num_records()
        if num_records == 0:
            raise RuntimeError("No vectors to index")

        n_clusters = self._choose_n_clusters(num_records)

        print(f"Building IVF: N={num_records}, clusters={n_clusters}")

        # # -------------------- step 1: load data --------------------
        db = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))

        db_magnitude = np.linalg.norm(db, axis=1, keepdims=True)
        non_zero_mask = (db_magnitude[:,0] != 0)
        db = db[non_zero_mask] / db_magnitude[non_zero_mask]

        # -------------------- step 2: train IVF centroids --------------------
        print("Training MiniBatchKMeans for centroids...")

        kmeans_ivf = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10_000, n_init="auto")
        kmeans_ivf.fit(db)
        centroids = kmeans_ivf.cluster_centers_.astype(np.float32)

        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / c_norms

        centroids.tofile(os.path.join(self.index_path, "centroids.bin"))
        print("Centroids saved.")

        # -------------------- step 3: assign vectors in chunks --------------------
        cluster_files = [
            open(os.path.join(self.index_path, f"cluster_{cid}.bin"), "ab")
            for cid in range(n_clusters)
        ]

        for start in range(0, num_records, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_records)


            chunk_vectors = db[start:end]

            similarity_matrix = chunk_vectors @ centroids.T
            assigned_clusters = np.argmax(similarity_matrix, axis=1)

            for local_idx, cid in enumerate(assigned_clusters):
                gid = start + local_idx
                cluster_files[cid].write(np.uint32(gid).tobytes())


        for f in cluster_files:
            f.close()

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k):

        # Initialize Parameters
        num_records = self.num_records
        n_clusters = IVF_CONFIGS[num_records]
        PROBE_CLUSTERS = PROBE_CLUSTERS_CONFIGS[num_records]

        # Normalize Query
        q = query.reshape(-1)
        q /= np.linalg.norm(q)

      
        # Batch load Centroids and compute similarity between batch and query
        BATCH = 1600
        top_clusters = []
        for start in range(0, n_clusters, BATCH):
            end = min(start + BATCH, n_clusters)
            batch = np.memmap(
                self.index_path + "/centroids.bin",
                dtype=np.float32,
                mode="r",
                shape=(end - start, DIMENSION),
                offset = start * DIMENSION * 4
            )
            scores = batch @ q

            # Keep only the highest n_probe (PROBE_CLUSTERS) centroid scores usnig minheap
            for local_idx, score in enumerate(scores):
                cid = start + local_idx
                heapq.heappush(top_clusters, (score, cid))
                if len(top_clusters) > PROBE_CLUSTERS:
                    heapq.heappop(top_clusters)

        top_vectors = []
        all_ids = []

        # Probe the top n_probe centroids and retrieve their vector ids
        for _, cid in top_clusters:
            cluster_path = f"{self.index_path}/cluster_{cid}.bin"
            ids = np.fromfile(cluster_path, dtype='<u4')
            all_ids.append(ids)


        # Sort for sequential I/O

        all_ids = np.concatenate(all_ids)
        all_ids.sort()
        n_ids = len(all_ids)


        # Batch the ids in a given range so that they belong to the same page
        i = 0
        while i < n_ids:
            start_id = all_ids[i]
            j = np.searchsorted(all_ids, start_id + BATCH, side='left')
            batch_ids = all_ids[i:j]
            offset = np.int64(start_id) * DIMENSION * 4
            length = (batch_ids[-1] - start_id + 1)
            mmap_batch = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode='r',
                offset=offset,
                shape=(length, DIMENSION)
            )

            batch = mmap_batch[(batch_ids - start_id)]

            batch_scores = batch @ q

            # keep the vectors with the top k scores only
            for score, vec_id in zip(batch_scores, batch_ids):
                heapq.heappush(top_vectors, (score, vec_id))
                if len(top_vectors) > top_k:
                    heapq.heappop(top_vectors)

            i = j

        top_vectors.sort(key=lambda x: x[0], reverse=True)

        top_ids = [id for _, id in top_vectors]
        return top_ids

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
