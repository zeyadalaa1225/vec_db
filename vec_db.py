from typing import Annotated, Dict
import numpy as np
import os
import struct
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64
TOP_K_DEFAULT = 5

# IVF number of clusters based on db size
IVF_CONFIGS = {
    1_000_000: 4096,      
    10_000_000: 8192,    
    15_000_000: 12288,   
    20_000_000: 16384     
}

# PQ M (Dimensionality of vector after applying PQ) choice
PQ_M_CONFIGS = {
    1_000_000: 32,      
    10_000_000: 8,
    15_000_000: 32,
    20_000_000: 32       
}
## de mesh mofeda zeyadetha asl wana batrain batrain 3ala 500000 kda kda fa mesh hyfe2 we wana baretrieve be retreive 3ala probe_cluster fe 3add el el data ele gowa kol cluster
# ele howa 8aleban bardo 3add sabet 34an 3add el ivf clusters byzed ma3a el size
# Number of clusters to probe
PROBE_CLUSTERS = 60

# Chunking and sampling for build phase
CHUNK_SIZE = 10_000
SAMPLE_SIZE = 500_000


class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat",
                 new_db=True, db_size=None, new_index = True) -> None:
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

    def _choose_pq_m(self, num_records):
        closest = min(PQ_M_CONFIGS.keys(), key=lambda x: abs(x - num_records))
        return PQ_M_CONFIGS[closest]

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
        M = self._choose_pq_m(num_records)

        # divide into M subvectors each with (DIMENSION//M) dimensionality
        # to account for M not divisible by DIMENSION add 1 from the remainder to each subvector dimensionality
        subdims = []
        base = DIMENSION // M
        rem = DIMENSION % M
        for i in range(M):
            subdims.append(base + (1 if i < rem else 0))

        print(f"Building IVF+PQ: N={num_records}, clusters={n_clusters}, PQ M={M}, subdims={subdims}")

        # -------------------- step 1: sample data --------------------
        samp = min(SAMPLE_SIZE, num_records)

        #TODO: Expirement with different sampling methods instead of random
        rng = np.random.default_rng(DB_SEED_NUMBER)
        sample_idxs = rng.choice(num_records, size=samp, replace=False)

        #TODO: GPT picked sample size as to make it fit in memory see if you can increase it 

        # load sample
        mm = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        sample = np.array(mm[sample_idxs])  # fits in memory (SAMPLE_SIZE x DIM)

        # normalize sample rows (so PQ approximates cosine reasonably)
        sample_norms = np.linalg.norm(sample, axis=1, keepdims=True)
        sample_norms[sample_norms == 0] = 1.0
        sample = sample / sample_norms

        '''
        ana hena ba5tar random sample_size rows we ba3mel le kol wa7ed normalization ele howa
        ba2sem 3al length aw el magnitude 34an wana bamatch fel pq a2der a3mel dot product 3ala tol

        '''

        # -------------------- step 2: train IVF centroids --------------------
        print("Training MiniBatchKMeans for centroids...")

        #TODO: Try to increase batch_size
        kmeans_ivf = MiniBatchKMeans(n_clusters=n_clusters, random_state=DB_SEED_NUMBER, batch_size=4096, n_init="auto")
        kmeans_ivf.fit(sample)
        centroids = kmeans_ivf.cluster_centers_.astype(np.float32)
        # normalize centroids
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1.0
        centroids = centroids / c_norms
        
        np.save(os.path.join(self.index_path,"centroids.npy"), centroids)
        print("Centroids saved.")

        # -------------------- step 3: train PQ codebooks (per subspace) --------------------
        print("Training PQ codebooks...")
        pq_codebooks = []
        start_idx = 0
        for sd in subdims:
            subspace = sample[:, start_idx:start_idx + sd]

            #TODO: Try to increase the number of clusters and batch_size
            kmeans_pq = MiniBatchKMeans(n_clusters=256, random_state=DB_SEED_NUMBER, batch_size=4096, n_init="auto")
            kmeans_pq.fit(subspace)
            codebook = kmeans_pq.cluster_centers_.astype(np.float32)
            #TODO: Try and normalize the codebooks

            pq_codebooks.append(codebook)
            start_idx += sd

        # Save PQ codebooks as a single .npz for convenience
        np.savez(os.path.join(self.index_path, "pq_codebooks.npz"), *pq_codebooks)
        print("PQ codebooks saved.")#(265,32)

        # -------------------- step 4: create empty cluster files and counts --------------------
        cluster_counts = np.zeros(n_clusters, dtype=np.uint32)   
        cluster_paths = [os.path.join(self.index_path,f"cluster_{i}.bin") for i in range(n_clusters)]
        # create (truncate) files
        for p in cluster_paths:
            open(p, "wb").close()

        # -------------------- step 5: assign vectors in chunks and encode PQ codes --------------------
        print("Assigning and encoding vectors in chunks...")
        mm_all = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        # iterate over all rows of db in chunks
        #TODO: Try and increase chunk size
        for start in range(0, num_records, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_records)
            # array of vectors
            chunk = np.array(mm_all[start:end], dtype=np.float32)  

            # normalize chunk
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            chunk_norm = chunk / norms  

            # compute similarity to centroids (dot product because normalized)
            similarity_matrix = chunk_norm @ centroids.T  # shape (chunk_size, n_clusters) (how similar is each vector to all centroids)
            chunk_cluster_ids = np.argmax(similarity_matrix, axis=1)  # Gets the nearest cluster for each vector (axis = 1) as an id (argmax)

            # encode PQ codes for chunk (vectorized)
            # Similar to finding neerest cluster but we do it for each subvector within the vector and we compute euclidean distance instead of dot product since pq clusters arent normalized
            chunk_codes = np.empty((end - start, M), dtype=np.uint8)
            s = 0
            for m_idx, sd in enumerate(subdims):
                subvecs = chunk_norm[:, s:s + sd]  # shape (chunk_size, sd)

                #the clusters for this subvector
                codebook = pq_codebooks[m_idx]  # shape (256, sd)

                # compute euclidean distance between codebook and subvector
                # For vectorized efficiency: use (a-b)^2 = a^2 + b^2 - 2a.b (av)
                a2 = np.sum(subvecs ** 2, axis=1, keepdims=True)  # (chunk,1)
                b2 = np.sum(codebook ** 2, axis=1)  # (256,)
                ab = subvecs @ codebook.T  # (chunk,256)
                # squared distances:
                dists = a2 + b2 - 2.0 * ab  # (chunk,256)
                nearest_codes = np.argmin(dists, axis=1).astype(np.uint8)
                chunk_codes[:, m_idx] = nearest_codes
                s += sd

            # group bytes to write per cluster to minimize file calls
            buffers: Dict[int, bytearray] = {}
            counts_local: Dict[int, int] = {}

            for local_idx, cid in enumerate(chunk_cluster_ids):
                global_id = start + local_idx
                code_bytes = chunk_codes[local_idx].tobytes()  # M bytes
                id_bytes = struct.pack('<I', int(global_id))  # 4 bytes, little-endian unsigned int
                if cid not in buffers:
                    buffers[cid] = bytearray()
                    counts_local[cid] = 0
                buffers[cid].extend(code_bytes)
                buffers[cid].extend(id_bytes)
                counts_local[cid] += 1

            # write buffers to files (append)
            for cid, buff in buffers.items():
                with open(cluster_paths[cid], "ab") as fh:
                    fh.write(buff)
                cluster_counts[cid] += counts_local[cid]

        # Save counts
        np.save(os.path.join(self.index_path,"cluster_counts.npy"), cluster_counts)
        print("Index build complete. Cluster counts saved.")

    # --------------------------- RETRIEVE (NO CACHING) --------------------------- #
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=TOP_K_DEFAULT):
        """
        Disk-only retrieval using IVF+PQ + ADC.
        No caching: loads centroids & PQ codebooks per call (tiny), reads cluster files sequentially.
        """
        
        # TODO: keep only reshape(-1)
        q = query.astype(np.float32).copy().reshape(-1)
        # normalize query
        qnorm = np.linalg.norm(q)
        if qnorm == 0:
            qnorm = 1.0
        q = q / qnorm

        # Load centroids (tiny) and pq codebooks (tiny) and counts
        centroids = np.load(self.index_path + "/centroids.npy")  # (n_clusters, DIM)

        # TODO: try using pq_npz directly
        pq_npz = np.load(self.index_path + "/pq_codebooks.npz")
        pq_codebooks = [pq_npz[key] for key in pq_npz.files]  # list of (256, subdim)
        cluster_counts = np.load(self.index_path + "/cluster_counts.npy")

        #TODO: Remove this line
        n_clusters = centroids.shape[0]

        M = len(pq_codebooks)

        # compute similarity to centroids and pick top PROBE_CLUSTERS
        similarity_to_centroids = centroids @ q  # (n_clusters,)
        top_clusters = np.argsort(similarity_to_centroids)[-PROBE_CLUSTERS:][::-1]

        # Precompute ADC tables T[m, 256]: distances between query subvector and PQ centroids
        # define subdims same as during build
        # compute subdims from pq_codebooks shapes

        #TODO: Try to not do this whole block and just use codebook and q directly
        subdims = [codebook.shape[1] for codebook in pq_codebooks]
        # split query into subspaces
        q_subs = []
        s = 0
        for sd in subdims:
            q_subs.append(q[s:s + sd])
            s += sd

        # compute T
        #TODO: make it a numpy array mel awel
        T = []
        for m_idx in range(M):
            codebook = pq_codebooks[m_idx]  # (256, sd)
            # compute squared L2 distances between q_subs[m] and all 256 centroids
            a2 = np.sum(q_subs[m_idx] ** 2)
            b2 = np.sum(codebook ** 2, axis=1)  # (256,)
            ab = codebook @ q_subs[m_idx]  # (256,)
            dists = a2 + b2 - 2.0 * ab  # (256,)
            T.append(dists.astype(np.float32))
        T = np.array(T)  # shape (M, 256)

        # Gather candidates
        candidates_ids = []
        candidates_scores = []  # lower = closer in L2; we'll pick smallest distances

        # For each probed cluster, read its file and compute ADC distances vectorized
        # TODO: Try to completely remove cluster_counts 
        for cid in top_clusters:
            cluster_path = f"{self.index_path}/cluster_{cid}.bin"
            if not os.path.exists(cluster_path):
                continue
            count = int(cluster_counts[cid])
            if count == 0:
                continue
            # read full content (sequential read)
            with open(cluster_path, "rb") as fh:
                raw = fh.read()
            # interpret as structured array: codes (M x uint8) + id (uint32)
            #TODO: try to store as np format to begin with
            dtype = np.dtype([('code', np.uint8, M), ('id', np.uint32)])
            try:
                arr = np.frombuffer(raw, dtype=dtype)
            except Exception as e:
                # in case of alignment / corruption, skip
                continue
            if arr.size == 0:
                continue

            codes = arr['code']  # shape (count, M), dtype uint8
            ids = arr['id'].astype(np.int64)

            # ADC: for each candidate, sum T[m, code[m]] across m
            # Vectorized: build (count, M) from indexing
            # T is (M, 256). For each m, T[m, codes[:, m]] -> (count,)
            dist_sum = np.zeros(codes.shape[0], dtype=np.float32)
            for m_idx in range(M):
                dist_sum += T[m_idx][codes[:, m_idx]]

            candidates_ids.append(ids)
            candidates_scores.append(dist_sum)

        if len(candidates_ids) == 0:
            return []

        # concatenate arrays
        all_ids = np.concatenate(candidates_ids)
        all_scores = np.concatenate(candidates_scores)

        # get top_k smallest distances
        if all_scores.size <= top_k:
            top_idx = np.argsort(all_scores)
        else:
            top_idx = np.argpartition(all_scores, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(all_scores[top_idx])]

        top_ids = all_ids[top_idx].tolist()
        return top_ids

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
