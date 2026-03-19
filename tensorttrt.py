import threading

import tensorrt as trt
import random
import pycuda.driver as cuda
import numpy as np
import time
import cv2
import os

CONF_THRESH = 0.0
IOU_THRESHOLD = 0.4
POSE_NUM = 0
DET_NUM = 6
SEG_NUM = 0
OBB_NUM = 0
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret
class TensorRTInfer:
    def __init__(self, engine_file_path):
        cuda.init()
        self.categories = categories
        self.ctx = cuda.Device(0).make_context()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Load engine
        with open(engine_file_path, "rb") as f:
            engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)

        # Create context
        context = engine.create_execution_context()
        stream = cuda.Stream()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.tensor_names = {"input": None, "output": None}
        self.context = context

        for binding in engine:
            mode = engine.get_tensor_mode(binding)
            shape = engine.get_tensor_shape(binding)
            size = trt.volume(shape)
            tensor_name = binding
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if mode == trt.TensorIOMode.INPUT:
                self.input_w = shape[-1]
                self.input_h = shape[-2]
                self.tensor_names["input"] = tensor_name
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                self.context.set_tensor_address(tensor_name, int(cuda_mem))

            else:
                self.tensor_names["output"] = tensor_name
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                self.context.set_tensor_address(tensor_name, int(cuda_mem))


        self.stream = stream

        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs

        self.batch_size = 1
        self.det_output_length = host_outputs[0].shape[0]


    def infer(self,raw_image_generator):

        try:
            self.ctx.push()
            stream = self.stream
            context = self.context
            host_inputs = self.host_inputs
            cuda_inputs = self.cuda_inputs
            host_outputs = self.host_outputs
            cuda_outputs = self.cuda_outputs


            batch_image_raw = []
            batch_origin_h = []
            batch_origin_w = []
            batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])

            for i, image_raw in enumerate(raw_image_generator):
                input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
                batch_image_raw.append(image_raw)
                batch_origin_w.append(origin_w)
                batch_origin_h.append(origin_h)
                np.copyto(batch_input_image[i], input_image[0])
            batch_input_image = np.ascontiguousarray(batch_input_image)

            np.copyto(host_inputs[0], batch_input_image.ravel())
            start = time.time()
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            context.execute_async_v3(
                stream_handle=stream.handle
            )
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            stream.synchronize()
            end = time.time()
            output = self.host_outputs[0]


            result_boxes, result_scores, result_classid =self.print_output_info(output, batch_origin_h, batch_origin_w)
            print(result_boxes)
            print(result_classid)
            print(result_scores)
            if len(result_boxes) > 0:
                for j in range(len(result_boxes)):
                    box = result_boxes[j]
                    plot_one_box(
                        box,
                        batch_image_raw[0],  # Lấy ảnh đầu tiên (và duy nhất)
                        label="{}:{:.2f}".format(
                            categories[int(result_classid[j])], result_scores[j]
                        ),
                    )

                # Lưu ảnh sau khi vẽ
                cv2.imwrite("result.jpg", batch_image_raw[0])
            return batch_image_raw, end - start

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return batch_image_raw, 0

        finally:
            # Pop context
            try:
                self.ctx.pop()
            except:
                pass

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        # Copy để không ảnh hưởng input
        y = x.copy()

        # Tính tỷ lệ resize
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h

        # Điều chỉnh tọa độ theo padding và tỷ lệ
        if r_h > r_w:
            # Pad theo chiều dọc
            pad = (self.input_h - r_w * origin_h) / 2
            y[:, 0] = x[:, 0]  # center_x
            y[:, 1] = x[:, 1] - pad  # center_y - pad
            y[:, 2] = x[:, 2]  # width
            y[:, 3] = x[:, 3]  # height
            y[:, :4] /= r_w  # Chia tỷ lệ
        else:
            # Pad theo chiều ngang
            pad = (self.input_w - r_h * origin_w) / 2
            y[:, 0] = x[:, 0] - pad  # center_x - pad
            y[:, 1] = x[:, 1]  # center_y
            y[:, 2] = x[:, 2]  # width
            y[:, 3] = x[:, 3]  # height
            y[:, :4] /= r_h  # Chia tỷ lệ

        # BƯỚC QUAN TRỌNG: Chuyển từ [center_x, center_y, w, h] sang [x1, y1, x2, y2]
        result = np.zeros_like(y)
        result[:, 0] = y[:, 0] - y[:, 2] / 2  # x1 = center_x - w/2
        result[:, 1] = y[:, 1] - y[:, 3] / 2  # y1 = center_y - h/2
        result[:, 2] = y[:, 0] + y[:, 2] / 2  # x2 = center_x + w/2
        result[:, 3] = y[:, 1] + y[:, 3] / 2  # y2 = center_y + h/2

        # Copy confidence và class_id nếu có
        if y.shape[1] > 4:
            result[:, 4:] = y[:, 4:]

        return result

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = (np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None)
                      * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None))
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.2, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH

        boxes = prediction[prediction[:, 4] >= conf_thres]
        print(boxes.shape)
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        print("boxes",boxes)
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        print(boxes)
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def print_output_info(self, output,batch_origin_h, batch_origin_w):
        """Print detailed information about the output"""
        print("\n" + "=" * 60)
        print("DETECTION RESULTS:")
        print("=" * 60)

        # Với output shape (50400,) - YOLOv8 format
        try:

            result_boxes = np.array([])
            result_scores = np.array([])
            result_classid = np.array([])
            # Reshape theo format YOLOv8 (1, 6, 8400)
            if len(output) == 50400:  # 1 * 6 * 8400 = 50400
                output_3d = output.reshape(1, 6, 8400)

                detections = output_3d[0]
                predictions = detections.T

                class_ids = predictions[:, 5]
                head_mask = class_ids >= 0.5
                filtered_predictions = predictions[head_mask]

                origin_h = batch_origin_h[0]  # Đây là số, không phải list
                origin_w = batch_origin_w[0]
                if len(filtered_predictions) > 0:
                    print("Applying NMS...")
                    boxes = self.non_max_suppression(
                        filtered_predictions,
                        origin_h,
                        origin_w,
                        conf_thres=CONF_THRESH,
                        nms_thres=IOU_THRESHOLD
                    )

                    if len(boxes) > 0:
                        result_boxes = boxes[:, :4]
                        result_scores = boxes[:, 4]
                        result_classid = boxes[:, 5]

                        print(f"Boxes after NMS: {len(boxes)}")
                        print(f"Class IDs: {result_classid}")
                else:
                    print("No detections with class_id >= 0.5")

                return result_boxes, result_scores, result_classid

        except Exception as e:
                print(f"Error in print_output_info: {e}")
                import traceback
                traceback.print_exc()
                return np.array([]), np.array([]), np.array([])

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)


    def preprocess_image(self,raw_bgr_image):

        image_raw = raw_bgr_image
        h,w,c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def get_raw_image(self, image_path_batch):
        """Read images from paths"""
        for img_path in image_path_batch:
            img = cv2.imread(img_path)
            if img is not None:
                yield img
            else:
                print(f"Warning: Could not read image {img_path}")
                yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Clean up CUDA resources in correct order"""
        try:
            # 1. Đầu tiên, hủy stream và context
            if hasattr(self, 'stream'):
                try:
                    self.stream = None
                except:
                    pass

            if hasattr(self, 'context'):
                try:
                    self.context = None
                except:
                    pass

            if hasattr(self, 'engine'):
                try:
                    self.engine = None
                except:
                    pass

            # 2. Giải phóng CUDA memory
            if hasattr(self, 'cuda_inputs'):
                for mem in self.cuda_inputs:
                    try:
                        mem.free()
                    except:
                        pass
                self.cuda_inputs = None

            if hasattr(self, 'cuda_outputs'):
                for mem in self.cuda_outputs:
                    try:
                        mem.free()
                    except:
                        pass
                self.cuda_outputs = None

            # 3. Pop và detach context CUDA
            if hasattr(self, 'ctx'):
                try:
                    self.ctx.pop()
                except:
                    pass
                try:
                    self.ctx.detach()
                except:
                    pass
                self.ctx = None

        except Exception as e:
            print(f"Cleanup warning: {e}")
class inferThread(threading.Thread):
    def __init__(self, yolov8_wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time = self.yolov8_wrapper.infer(self.yolov8_wrapper.get_raw_image(self.image_path_batch))
        for i, img_path in enumerate(self.image_path_batch):
            parent, filename = os.path.split(img_path)
            save_name = os.path.join('output', filename)
            # Save image
            cv2.imwrite(save_name, batch_image_raw[i])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))




class warmUpThread(threading.Thread):
    def __init__(self, yolov8_wrapper):
        threading.Thread.__init__(self)
        self.yolov8_wrapper = yolov8_wrapper

    def run(self):
        try:
            batch_image_raw, use_time = self.yolov8_wrapper.infer(
                self.yolov8_wrapper.get_raw_image_zeros()
            )
            print(f'Warmup time: {use_time * 1000:.2f}ms')
        except Exception as e:
            print(f"Error in warmup thread: {e}")

if __name__ == "__main__":
    engine_file_path = r"D:\game\Myproject\toolvalorant\bạckend\native\best.engine"
    categories = [" enemy_head","head"]
    # Check if engine file exists
    if not os.path.exists(engine_file_path):
        print(f"Error: Engine file not found at {engine_file_path}")
        exit(1)

    print(f"Loading engine from: {engine_file_path}")

    yolov8_wrapper = TensorRTInfer(engine_file_path)
    try:
        print('batch size is', yolov8_wrapper.batch_size)
        image_dir = r"D:\data_set\valorant_selected_frames\fn.v1i.yolov8\test\test"
        for i in range(5):
            # create a new thread to do warm_up
            thread1 = warmUpThread(yolov8_wrapper)
            thread1.start()
            thread1.join()

        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:
                for i in range(min(3, len(image_files))):
                    test_image = os.path.join(image_dir, image_files[i])
                    print(f'\nTesting image {i + 1}: {test_image}')

                    thread1 = inferThread(yolov8_wrapper, [test_image])
                    thread1.start()
                    thread1.join()
            else:
                print("No images found in directory")
    finally:
        # destroy the instance
        if yolov8_wrapper:
            yolov8_wrapper.cleanup()
        print("\nDone!")