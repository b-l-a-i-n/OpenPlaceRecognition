import logging
import paddleocr.tools.infer.predict_rec as predict_rec
import paddleocr.tools.infer.predict_det as predict_det
import paddleocr.tools.infer.predict_cls as predict_cls
from paddleocr.tools.infer.predict_system import TextSystem
from paddleocr.ppocr.utils.logging import get_logger
import numpy as np
import onnxruntime as ort


logger = get_logger()


class PaddleOcrPipeline(TextSystem):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def patch_onnx_povider(self, provider: str):
        """
        Switch CPU onnxruntive provider to CUDA or Tensorrt 
        
        provider: str [cuda, tensorrt]
        
        """
        
        if provider not in ["cuda", "tensorrt"]:
            print("Invalid provider! Check docstring")
            return 
        
        det_model_dir = self.args.det_model_dir
        rec_model_dir = self.args.rec_model_dir
        
        available_providers = ort.get_available_providers()
        
        provider = "CUDAExecutionProvider" if provider == "cuda" else "TensorrtExecutionProvider"
        
        if provider not in available_providers:
            print("Provider is not supported!")
            print(f"Available providers: {', '.join(available_providers)}")
            return

        if det_model_dir.endswith(".onnx"):
            try:
                det_sess = ort.InferenceSession(det_model_dir)
                det_sess.set_providers([provider])
            except Exception as e:
                print(e)
        else:
            print("Detection model is not in onnx format!")

        if rec_model_dir.endswith(".onnx"):
            try:
                rec_sess = ort.InferenceSession(rec_model_dir)
                rec_sess.set_providers([provider])
            except Exception as e:
                print(e)
        else:
            print("Recognition model is not in onnx format!")
        
        self.text_detector.predictor = det_sess
        self.text_recognizer.predictor = rec_sess