import logging
import time
import cv2
import mediapipe

from math import floor, exp
from cv2.typing import MatLike
from threading import Condition, Event, Lock, Thread

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.face_detector import FaceDetector, FaceDetectorOptions, FaceDetectorResult

from src.stream.camera.CameraFrame import CameraFrame
from src.stream.core.StreamReadOnly import StreamReadOnly
from src.stream.core.StreamWriteOnly import StreamWriteOnly
from src.stream.core.components.WriteStreamSplitter import WriteStreamSplitter

_logger = logging.getLogger(__name__)


def expDecay(a: float, b: float, decay: float, dt: float):
    return b + ( a - b ) * exp( -decay * dt )


class MediaPipeCropStream:
    """
    Unstable when recreated, try to avoid any reinitialization
    """

    def __init__(self, image_stream: StreamReadOnly[CameraFrame], model_asset_data: bytes,
                 frame_timeout: float | None = 1.0, frame_lost_timeout: float = 1.0):

        self.lastCrop = {
            "t": -1,
            "x": 0.0,
            "y": 0.0,
            "w": 0.0,
            "h": 0.0
        }

        self.__image_stream: StreamReadOnly[CameraFrame] = image_stream
        self.__frame_timeout: float | None = frame_timeout
        self.__frame_lost_timeout: float = frame_lost_timeout

        self.__detector = self.__create_detector(model_asset_data)

        self.__close_event = Event()
        self.__condition_lock = Condition(Lock())
        self.__callback_lock = Lock()

        self.__last_frame: CameraFrame | None = None
        self.__last_packet_time_ms: int = time.perf_counter_ns() // 1_000_000
        self.__last_callback_time_ms: int = time.perf_counter_ns() // 1_000_000

        self.__stream_root = WriteStreamSplitter[CameraFrame]()

        self.__fps_limiter_time: int = time.perf_counter_ns()
        self.__fps_limit_ns: int | None = None

        self.__thread = Thread(target=self.__loop, daemon=True, name="MediaPipeCrop Thread")
        self.__thread.start()

    def register_stream(self, stream: StreamWriteOnly[CameraFrame]) -> None:
        self.__stream_root.register_stream(stream)

    def unregister_stream(self, stream: StreamWriteOnly[CameraFrame]) -> None:
        self.__stream_root.unregister_stream(stream)

    def set_fps_limit(self, fps_limit: int | None):
        if fps_limit is None:
            self.__fps_limit_ns = None
        else:
            if fps_limit <= 0:
                raise ValueError("fps_limit must be positive")

            self.__fps_limit_ns = 1_000_000_000 // fps_limit

    def close(self):
        self.__close_event.set()
        self.__stream_root.close()

        with self.__condition_lock:
            self.__condition_lock.notify_all()

        try:
            self.__thread.join(self.__frame_timeout * 2.0)
        except Exception:
            _logger.warning("Failed to join MediaPipeCrop thread", exc_info=True, stack_info=True)

        self.__detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __loop(self):
        while not self.__close_event.is_set():
            try:
                self.__last_frame = self.__image_stream.poll(self.__frame_timeout)
                packet_time_ms = self.__last_frame.timestamp_ns // 1_000_000

                if self.__last_packet_time_ms - packet_time_ms >= 0:
                    continue  # System lag

                mp_image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=self.__last_frame.frame)

                self.__detector.detect_async(mp_image, packet_time_ms)

                with self.__condition_lock:  # not ideal back-pressure, we can load more to achieve more FPS, but latency will increase
                    self.__condition_lock.wait(self.__frame_lost_timeout)

                fps_limit = self.__fps_limit_ns
                if fps_limit is not None:
                    target_frame_completion_time_ns = self.__fps_limiter_time + fps_limit
                    current_actual_time_ns = time.perf_counter_ns()
                    sleep_duration_ns = target_frame_completion_time_ns - current_actual_time_ns
                    if sleep_duration_ns > 0:
                        self.__close_event.wait(sleep_duration_ns / 1_000_000_000)
                        self.__fps_limiter_time = target_frame_completion_time_ns
                    else:
                        self.__fps_limiter_time = current_actual_time_ns

                self.__last_packet_time_ms = packet_time_ms

            except TimeoutError:
                continue
            except InterruptedError:
                return
            except Exception:
                _logger.warning("Exception in MediaPipeCrop loop", exc_info=True, stack_info=True)
                self.__close_event.wait(0.001)

    def __async_result(self, result: FaceDetectorResult, image, timestamp_ms):
        last_packet = self.__last_frame

        with self.__condition_lock:
            self.__condition_lock.notify()

        if result.detections:
            try:
                with self.__callback_lock:
                    if self.__last_callback_time_ms - timestamp_ms > 0:
                        return

                    cropped_frame = self.crop_face(last_packet.frame, result.detections, timestamp_ms)

                    self.__last_callback_time_ms = timestamp_ms
                    self.__stream_root.put(CameraFrame(cropped_frame, last_packet.timestamp_ns))

            except InterruptedError:
                return
            except Exception:
                _logger.warning("Exception in MediaPipeCrop callback", exc_info=True, stack_info=True)

    def __create_detector(self, model_asset_data: bytes) -> FaceDetector:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_buffer=model_asset_data), #, delegate=BaseOptions.Delegate.GPU
            running_mode=RunningMode.LIVE_STREAM,
            result_callback=self.__async_result)

        return FaceDetector.create_from_options(options)

    def crop_face(self, image, detections, timestamp_ms, crop_padding = 80, crop_offset_y = 20) -> MatLike:
        output_image = image.copy()
        height, width, _ = image.shape
        aspect = width / height

        cropW = width
        cropH = height
        cropCenterX = width * 0.5
        cropCenterY = height * 0.5

        if len(detections) > 0:
            bbox = detections[0].bounding_box

            cropW = bbox.width
            cropH = bbox.height
            cropCenterX = max(cropW * 0.5, bbox.origin_x + cropW * 0.5)
            cropCenterY = max(cropH * 0.5, bbox.origin_y - crop_offset_y + cropH * 0.5)

            cropW = (cropW + crop_padding) * aspect
            cropH = cropH + crop_padding

        last = self.lastCrop

        if last["t"] > 0:
            dt = (timestamp_ms - last["t"]) / 1000.0

            cropCenterX = expDecay(last["x"], cropCenterX, 3.0, dt)
            cropCenterY = expDecay(last["y"], cropCenterY, 3.0, dt)

            cropW = expDecay(last["w"], cropW, 3.0, dt)
            cropH = expDecay(last["h"], cropH, 3.0, dt)

        last["t"] = timestamp_ms
        last["x"] = cropCenterX
        last["y"] = cropCenterY
        last["w"] = cropW
        last["h"] = cropH

        top = max(0, floor(cropCenterY - cropH * 0.5))
        left = max(0, floor(cropCenterX - cropW * 0.5))
        bottom = min(height, floor(cropCenterY + cropH * 0.5))
        right = min(width, floor(cropCenterX + cropW * 0.5))

        return cv2.resize(output_image[top:bottom, left:right], dsize=(width, height), interpolation=cv2.INTER_CUBIC)
