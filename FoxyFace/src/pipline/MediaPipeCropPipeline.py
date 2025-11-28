import logging
from typing import Any, Callable

from AppConstants import AppConstants

from src.pipline.CameraPipeline import CameraPipeline
from src.stream.camera.CameraFrame import CameraFrame
from src.stream.camera.CameraProcessing import CameraProcessing
from src.stream.core.StreamWriteOnly import StreamWriteOnly
from src.stream.core.components.SingleBufferStream import SingleBufferStream
from src.stream.core.components.WriteCpsCounter import WriteCpsCounter
from src.stream.mediapipe.core.MediaPipeCropStream import MediaPipeCropStream

_logger = logging.getLogger(__name__)


class MediaPipeCropPipeline:
    def __init__(self, camera_pipeline: CameraPipeline):
        self.__camera_pipeline = camera_pipeline
        self.__buffer = SingleBufferStream[CameraFrame]()
        self.__camera_pipeline.register_stream(self.__buffer)

        processed_stream = CameraProcessing(self.__buffer, self.__camera_pipeline.get_processing_options())

        self.__stream: MediaPipeCropStream = MediaPipeCropStream(processed_stream, MediaPipeCropPipeline.__read_media_pipe_model())

        self.__fps_counter = WriteCpsCounter()
        self.__stream.register_stream(self.__fps_counter)

    def register_stream(self, stream: StreamWriteOnly[CameraFrame]) -> None:
        self.__stream.register_stream(stream)

    def unregister_stream(self, stream: StreamWriteOnly[CameraFrame]) -> None:
        self.__stream.unregister_stream(stream)

    def get_fps(self):
        return self.__fps_counter.get_cps()

    def close(self):
        self.__camera_pipeline.unregister_stream(self.__buffer)

        self.__stream.unregister_stream(self.__fps_counter)
        self.__stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __update_fps_limit(self):
        self.__stream.set_fps_limit(30)

    @staticmethod
    def __read_media_pipe_model() -> bytes:
        return (AppConstants.get_application_root() / 'Assets' / 'blaze_face_short_range.tflite').read_bytes()
