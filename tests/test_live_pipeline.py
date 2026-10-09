import unittest
import cv2
import numpy as np
import os
import glob
from src.runtime.live_pipeline_v4 import LiveInferencePipelineV4
from src.runtime.live_swap import NativeInSwapper

class TestLivePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = LiveInferencePipelineV4()
        
        # We must use ONLY real images that exist in the repo.
        self.alice_images = glob.glob("sample_data/lfw/Angelina_Jolie/*.jpg")[:5]
        for img_path in self.alice_images:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Required test fixture missing: {img_path}")
                
        # Bob (Andre Agassi)
        self.bob_images = glob.glob("sample_data/lfw/Andre_Agassi/*.jpg")[:5]
        for img_path in self.bob_images:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Required test fixture missing: {img_path}")

        # Enroll Alice
        self.pipeline.identity_pipeline.enroll("Alice", self.alice_images)
        
        # Set Alice as source face
        alice_img = cv2.imread(self.alice_images[0])
        faces = self.pipeline.identity_pipeline.engine.extract_faces(alice_img)
        self.alice_face = faces[0]
        self.pipeline.swapper.set_source_face(self.alice_face)

        # Bob test frame
        self.bob_image = cv2.imread(self.bob_images[0])
        self.alice_test_image = cv2.imread(self.alice_images[0])

    def assertFrontendSchema(self, res, expect_face=True):
        """Validates that the returned dictionary exactly matches the frontend InferenceResult schema"""
        expected_keys = {
            "timestamp", "session_id", "frame_id", "condition", "face_detected",
            "swap_applied", "raw_model_a_similarity", "ema_model_a_similarity",
            "model_a_threshold", "model_a_raw_accept", "matched_identity",
            "raw_c_class", "raw_c_probs", "smoothed_class", "ema_c_probs",
            "raw_predicted_state", "final_state", "swap_latency", "model_a_latency",
            "model_c_latency", "total_latency", "fps", "status", "error"
        }
        self.assertTrue(expected_keys.issubset(set(res.keys())), f"Missing keys: {expected_keys - set(res.keys())}")
        
        if expect_face:
            self.assertIsNotNone(res["raw_model_a_similarity"])
            self.assertIsNotNone(res["final_state"])
        else:
            self.assertIsNone(res["raw_model_a_similarity"])
            self.assertEqual(res["final_state"], "UNKNOWN")

    def test_missing_face_handling(self):
        # Explicit test for invalid/no-face handling
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.pipeline.process_frame(empty_frame, condition="genuine", frame_id=0)
        self.assertEqual(res["status"], "NO_FACE")
        self.assertEqual(res["raw_predicted_state"], "UNKNOWN")
        self.assertFrontendSchema(res, expect_face=False)

    def test_genuine_condition_alice(self):
        # Alice in front of camera (should accept)
        self.pipeline.start_session("test_genuine_alice")
        res = self.pipeline.process_frame(self.alice_test_image, condition="genuine", frame_id=1)
        self.assertTrue(res["face_detected"])
        self.assertTrue(res["model_a_raw_accept"])
        self.assertFrontendSchema(res, expect_face=True)

    def test_genuine_condition_unknown(self):
        # Bob in front of camera without swap (identity gate should reject)
        self.pipeline.start_session("test_genuine")
        res = self.pipeline.process_frame(self.bob_image, condition="genuine", frame_id=1)
        self.assertTrue(res["face_detected"])
        
        # It must reject because Bob != Alice
        self.assertFalse(res["model_a_raw_accept"])
        self.assertEqual(res["raw_predicted_state"], "UNKNOWN")
        self.assertFrontendSchema(res, expect_face=True)
        
        # Ensure threshold is exactly the frozen 0.244529
        self.assertAlmostEqual(res["model_a_threshold"], 0.244529, places=5)

    def test_impersonation_condition(self):
        # Alice swapped onto Bob
        self.pipeline.start_session("test_impersonation")
        res = self.pipeline.process_frame(self.bob_image, condition="impersonation", frame_id=1)
        self.assertTrue(res["face_detected"])
        
        self.assertIn(res["raw_predicted_state"], ["VERIFIED", "UNKNOWN", "SUSPECTED_IMPERSONATION"])
        self.assertFrontendSchema(res, expect_face=True)
        
    def test_temporal_smoothing(self):
        self.pipeline.reset_temporal_state()
        self.pipeline.ema_alpha = 0.5
        self.pipeline.ema_similarity = 0.8
        self.pipeline.initialized_ema = True
        
        # Feed Bob image
        res = self.pipeline.process_frame(self.bob_image, condition="genuine", frame_id=1)
        
        # Raw should be very low (since Bob != Alice)
        self.assertTrue(res["raw_model_a_similarity"] < 0.2)
        # But EMA should be blended with the initialized 0.8
        self.assertTrue(res["ema_model_a_similarity"] > 0.3)
        self.assertFrontendSchema(res, expect_face=True)

if __name__ == "__main__":
    unittest.main()
