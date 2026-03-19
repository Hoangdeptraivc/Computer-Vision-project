from typing import List,Tuple
import time
import cv2
import numpy as np
import onnxruntime


class Detection:
    def __init__(self, model_path: str, device: str = "cpu",conf_threshold :float = 0.5,iou_threshold :float = 0.5,
                 score_threshold: float = 0.1,
                 original_size: Tuple[int, int] = (640, 640),):
        self.model = model_path
        self.classes = ["enemy", "head_enemy"]
        self.device = device
        self.create_session()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.image_width, self.image_height = original_size

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = ['CPUExecutionProvider']
        if self.device.casefold() != "cpu":
            providers.insert(0, "CUDAExecutionProvider")
        session = onnxruntime.InferenceSession(self.model, providers=providers)

        self.session = session
        self.model_input = self.session.get_inputs()
        self.input_names = [self.model_input[i].name for i in range(len(self.model_input))]
        self.input_shape = self.model_input[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

    def preprocess(self, img_path: str) -> np.array:
        # Đọc ảnh từ đường dẫn
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ path: {img_path}")

        # Tiền xử lý
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def detect(self, img_path: str) -> List:
        # Tiền xử lý
        input_tensor = self.preprocess(img_path)
        init_start = time.time()
        # Inference
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_tensor}
        )
        init_time = time.time() - init_start
        return outputs,init_time
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    def postpocess(self,outputs):

        predictions =np.squeeze(outputs).T


        class_ids = predictions[:,5]
        head_mask = class_ids >= 0.5
        filtered_predictions = predictions[head_mask]
        scores = filtered_predictions[:, 4]


        boxes = filtered_predictions[:,:4]

        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        boxes = boxes.astype(np.int32)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=self.score_threshold,
                                   nms_threshold=self.iou_threshold)

        detections = []
        for bbox, score, label in zip(self.xywh2xyxy(boxes[indices]), scores[indices],class_ids[indices]):
            detections.append({
                "class_index": label,
                "confidence": score,
                "box": bbox,

            })
        return detections



    def draw_detetection(self,img, detections: List):

        image = cv2.imread(img)

        for detection in detections:
            bbox = detection['box']
            confidence = detection['confidence']
            class_score = detection['class_index']

            # Lấy tọa độ từ bbox
            x1, y1, x2, y2 = bbox

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image,(x1,y1),(x2,y2),(0.255,0),2)
            label = "head_enemy"
            cv2.putText(image,label ,(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image





if __name__ == "__main__":
    # Đường dẫn tới model và ảnh
    model_path = r"D:\game\Myproject\toolvalorant\best.onnx"
    image_path = r"D:\data_set\valorant_selected_frames\fn.v1i.yolov8\test\images\clip2-681-_jpg.rf.9ca894b70b0d2ceca17db797ca9a3488.jpg"
    # Khởi tạo detector
    detector = Detection(model_path, device="cpu")

    total_start = time.time()



    results,time = detector.detect(image_path)
    print("time:",time)


    predict = detector.postpocess(results)

    draw = detector.draw_detetection(image_path,predict)
    cv2.imshow('Detections', draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

