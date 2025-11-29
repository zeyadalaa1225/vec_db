from typing import Annotated, Dict
import numpy as np
import os
import struct
from sklearn.cluster import KMeans, MiniBatchKMeans
import heapq
import gc

ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
DB_SEED_NUMBER = 42


# IVF number of clusters based on db size
IVF_CONFIGS = {
    1_000_000: 4096,
    10_000_000: 8192,
    20_000_000: 16384
}


## de mesh mofeda zeyadetha asl wana batrain batrain 3ala 500000 kda kda fa mesh hyfe2 we wana baretrieve be retreive 3ala probe_cluster fe 3add el el data ele gowa kol cluster
# ele howa 8aleban bardo 3add sabet 34an 3add el ivf clusters byzed ma3a el size
# Number of clusters to probe
PROBE_CLUSTERS_CONFIGS = {
    1_000_000: 10,
    10_000_000: 10,
    20_000_000: 10
}

# Chunking and sampling for build phase
CHUNK_SIZE = 5_000

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat",
                 new_db=True, db_size=None, new_index = False) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
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

    # --------------------------- INDEX BUILD: IVF + PQ --------------------------- #
    def _choose_n_clusters(self, num_records):
        # pick nearest config
        closest = min(IVF_CONFIGS.keys(), key=lambda x: abs(x - num_records))
        return IVF_CONFIGS[closest]

    def _build_index(self):
        """
        Build IVF + PQ index from on-disk vectors.
        - Centroids saved to index_path_centroids.npy
        - PQ codebooks saved to index_path_pq.npy (shape: list of arrays or stacked)
        - Per-cluster inverted lists saved to index_path_cluster_{cid}.bin
          Each entry in cluster file: struct of M uint8 (code) followed by uint32 (vector id)
        - cluster_counts saved to index_path_cluster_counts.npy
        """
        num_records = self._get_num_records()
        if num_records == 0:
            raise RuntimeError("No vectors to index")

        n_clusters = self._choose_n_clusters(num_records)

        print(f"Building IVF: N={num_records}, clusters={n_clusters}")

        # -------------------- step 1: load data --------------------
        db = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))

        db_magnitude = np.linalg.norm(db, axis=1, keepdims=True)
        # non_zero_mask = (db_magnitude[:,0] != 0)
        # db = db[non_zero_mask] / db_magnitude[non_zero_mask]
        db_magnitude[db_magnitude == 0] = 1
        db = db / db_magnitude
        
        # -------------------- step 2: train IVF centroids --------------------
        print("Training MiniBatchKMeans for centroids...")

        kmeans_ivf = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10_000, n_init="auto")
        kmeans_ivf.fit(db)
        centroids = kmeans_ivf.cluster_centers_.astype(np.float32)

        # kmeans_ivf = KMeans(
        #   n_clusters=n_clusters,
        #   random_state=DB_SEED_NUMBER,
        #   n_init="auto",
        #   max_iter=300,
        #   algorithm="lloyd"
        # )
        # kmeans_ivf.fit(sample)
        # centroids = kmeans_ivf.cluster_centers_.astype(np.float32)

        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / c_norms

        centroids.tofile(os.path.join(self.index_path, "centroids.bin"))
        print("Centroids saved.")

        # # -------------------- step 3: assign vectors in chunks --------------------
        # centroids = np.fromfile(self.index_path + "/centroids.bin", dtype=np.float32).reshape((n_clusters, DIMENSION))
        cluster_files = [
            open(os.path.join(self.index_path, f"cluster_{cid}.bin"), "ab")
            for cid in range(n_clusters)
        ]

        for start in range(0, num_records, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_records)
            # chunk_size = end - start
            # offset = start * DIMENSION * 4
            # chunk_vectors = np.memmap(
            #     self.db_path,
            #     dtype=np.float32,
            #     mode="r",
            #     shape=(chunk_size, DIMENSION),
            #     offset = offset
            # )

            chunk_vectors = db[start:end]
            # norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
            # non_zero_mask = (norms[:,0] != 0)
            # chunk_vectors = chunk_vectors[non_zero_mask] / norms[non_zero_mask]

            similarity_matrix = chunk_vectors @ centroids.T
            assigned_clusters = np.argmax(similarity_matrix, axis=1)

            for local_idx, cid in enumerate(assigned_clusters):
                gid = start + local_idx
                cluster_files[cid].write(np.uint32(gid).tobytes())


        for f in cluster_files:
            f.close()

    # --------------------------- RETRIEVE (NO CACHING) --------------------------- #
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k):
        """
        Disk-only retrieval using IVF+PQ + ADC.
        No caching: loads centroids & PQ codebooks per call (tiny), reads cluster files sequentially.
        """
        num_records = self._get_num_records()
        n_clusters = IVF_CONFIGS[num_records]
        PROBE_CLUSTERS = PROBE_CLUSTERS_CONFIGS[num_records]


        # TODO: keep only reshape(-1)
        q = query.astype(np.float32).copy().reshape(-1)
        # normalize query
        qnorm = np.linalg.norm(q)
        if qnorm == 0:
            qnorm = 1.0
        q = q / qnorm

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
            for local_idx, score in enumerate(scores):
                cid = start + local_idx
                heapq.heappush(top_clusters, (score, cid))
                if len(top_clusters) > PROBE_CLUSTERS:
                    heapq.heappop(top_clusters)

        vectors = np.memmap(
            self.db_path,
            dtype=np.float32,
            mode='r',
            shape=(num_records, DIMENSION)
        )
        top_vectors = []
        all_ids = []
        for _, cid in top_clusters:
            cluster_path = f"{self.index_path}/cluster_{cid}.bin"
            with open(cluster_path, "rb") as fh:
                raw = fh.read()
            ids = np.frombuffer(raw, dtype='<u4')
            # ids.sort()
            all_ids.append(ids)
            # candidate_vectors = vectors[ids]

            # scores = candidate_vectors @ q
            # for score, id in zip(scores,ids):
            #     if id == 19852040:
            #       print(score)
            #     heapq.heappush(top_vectors, (score, id))

            #     if len(top_vectors) > top_k:
            #         heapq.heappop(top_vectors)

            # for id in ids:

            #     candidate_vector = self.get_one_row(np.int64(id))

            #     score = candidate_vector @ q
            #     heapq.heappush(top_vectors, (score, id))

            #     if len(top_vectors) > top_k:
            #         heapq.heappop(top_vectors)



        all_ids = np.concatenate(all_ids)
        all_ids.sort()
        n_ids = len(all_ids)


        i = 0
        while i < n_ids:
            start_id = all_ids[i]
            batch_ids = [start_id]
            j = i + 1
            while j < n_ids and all_ids[j] - start_id < BATCH:
                batch_ids.append(all_ids[j])
                j += 1

            offset = np.int64(start_id) * DIMENSION * 4
            length = (batch_ids[-1] - start_id + 1)
            mmap_batch = np.memmap(
                self.db_path,
                dtype=np.float32,
                mode='r',
                offset=offset,
                shape=(length, DIMENSION)
            )

            local_indices = [idx - start_id for idx in batch_ids]
            batch = mmap_batch[local_indices]

            scores = batch @ q
            for score, vec_id in zip(scores, batch_ids):
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
