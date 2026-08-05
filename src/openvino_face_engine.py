"""
OpenVINO FP16 Face Feature Extraction & Cosine Similarity Matcher
Generates 512-dimensional L2-normalized embeddings and performs sub-millisecond vector matching.
"""

import os
import cv2
import numpy as np
import openvino as ov


class OpenVINOFaceEngine:
    """
    Inference engine for facial feature extraction using OpenVINO FP16.
    Produces 512-dimensional L2-normalized facial embeddings.
    """
    def __init__(self, model_path=None, device="CPU"):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "openvino_fp16", "facenet_vggface2.xml")
            
        self.core = ov.Core()
        
        config = {
            ov.properties.hint.performance_mode(): ov.properties.hint.PerformanceMode.LATENCY,
            ov.properties.hint.execution_mode(): ov.properties.hint.ExecutionMode.PERFORMANCE
        }
        
        self.compiled_model = self.core.compile_model(model_path, device, config)
        self.infer_request = self.compiled_model.create_infer_request()
        print(f"[OpenVINOFaceEngine] Loaded FP16 Feature Extractor on {device}")

    def extract_embedding(self, aligned_bgr_or_rgb_face, is_bgr=True):
        """
        Extract 512-d L2 normalized embedding for a single aligned face (160x160).
        """
        if is_bgr:
            rgb_face = cv2.cvtColor(aligned_bgr_or_rgb_face, cv2.COLOR_BGR2RGB)
        else:
            rgb_face = aligned_bgr_or_rgb_face
            
        if rgb_face.shape[:2] != (160, 160):
            rgb_face = cv2.resize(rgb_face, (160, 160), interpolation=cv2.INTER_LINEAR)
            
        # Normalization
        inp = (rgb_face.astype(np.float32) - 127.5) * 0.0078125
        inp = np.ascontiguousarray(inp.transpose(2, 0, 1)[None, ...], dtype=np.float32)
        
        # OpenVINO inference
        res = self.infer_request.infer({0: inp})
        embedding = list(res.values())[0][0]
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
            
        return embedding.astype(np.float32)

    def extract_embeddings_batch(self, aligned_faces, is_bgr=True):
        """
        Extract embeddings for a list or batch of aligned faces (160x160).
        """
        if len(aligned_faces) == 0:
            return np.empty((0, 512), dtype=np.float32)
            
        batch = []
        for face in aligned_faces:
            if is_bgr:
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            else:
                rgb = face
            if rgb.shape[:2] != (160, 160):
                rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
            inp = (rgb.astype(np.float32) - 127.5) * 0.0078125
            batch.append(inp.transpose(2, 0, 1))
            
        batch_tensor = np.ascontiguousarray(np.stack(batch), dtype=np.float32)
        res = self.infer_request.infer({0: batch_tensor})
        embeddings = list(res.values())[0]
        
        # Batch L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        embeddings = embeddings / norms
        return embeddings.astype(np.float32)


class FaceMatcher:
    """
    High-speed Cosine Similarity Face Matcher.
    Stores and matches against enrolled employee templates.
    """
    def __init__(self, db_path=None, similarity_threshold=0.60):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.employee_ids = []
        self.employee_names = []
        self.templates = np.empty((0, 512), dtype=np.float32)
        
        if db_path and os.path.exists(db_path):
            self.load_database(db_path)

    def load_database(self, db_path):
        """Load enrolled templates from .npz or .npy database file"""
        self.db_path = db_path
        data = np.load(db_path, allow_pickle=True)
        self.employee_ids = list(data["ids"])
        self.employee_names = list(data["names"]) if "names" in data else list(data["ids"])
        self.templates = data["templates"].astype(np.float32)
        
        # Ensure templates are L2 normalized
        norms = np.linalg.norm(self.templates, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        self.templates = self.templates / norms
        
        print(f"[FaceMatcher] Loaded database '{db_path}' with {len(self.employee_ids)} enrolled templates.")

    def save_database(self, db_path):
        """Save enrolled templates to .npz file"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        np.savez_compressed(
            db_path,
            ids=np.array(self.employee_ids),
            names=np.array(self.employee_names),
            templates=self.templates
        )
        print(f"[FaceMatcher] Saved {len(self.employee_ids)} templates to '{db_path}'")

    def add_template(self, employee_id, employee_name, embedding):
        """Add an enrolled embedding to memory"""
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm
        embedding = embedding.reshape(1, 512).astype(np.float32)
        
        self.employee_ids.append(employee_id)
        self.employee_names.append(employee_name)
        if len(self.templates) == 0:
            self.templates = embedding
        else:
            self.templates = np.vstack([self.templates, embedding])

    def match(self, query_embedding):
        """
        Match a query embedding against the database.
        Returns:
            (best_id, best_name, similarity, is_match)
        """
        if len(self.templates) == 0:
            return "UNKNOWN", "Unknown Person", 0.0, False
            
        # Cosine similarity is simply dot product between L2 normalized vectors
        sims = np.dot(self.templates, query_embedding)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        
        if best_sim >= self.similarity_threshold:
            return self.employee_ids[best_idx], self.employee_names[best_idx], best_sim, True
        else:
            return "UNKNOWN", "Unknown Person", best_sim, False

    def match_batch(self, query_embeddings):
        """
        Batch match query embeddings against the database.
        Returns list of (best_id, best_name, similarity, is_match)
        """
        if len(self.templates) == 0 or len(query_embeddings) == 0:
            return [("UNKNOWN", "Unknown Person", 0.0, False)] * len(query_embeddings)
            
        sims_matrix = np.dot(query_embeddings, self.templates.T)  # Shape (N_queries, N_templates)
        results = []
        for i in range(len(query_embeddings)):
            best_idx = int(np.argmax(sims_matrix[i]))
            best_sim = float(sims_matrix[i, best_idx])
            if best_sim >= self.similarity_threshold:
                results.append((self.employee_ids[best_idx], self.employee_names[best_idx], best_sim, True))
            else:
                results.append(("UNKNOWN", "Unknown Person", best_sim, False))
        return results
