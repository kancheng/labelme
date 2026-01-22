from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numpy.typing import NDArray

try:
    import osam
except (ImportError, OSError, RuntimeError):
    osam = None  # type: ignore[assignment]

if TYPE_CHECKING:
    if osam is not None:
        from osam.types import GenerateResponse, Model, ImageEmbedding, Prompt, GenerateRequest
    else:
        GenerateResponse = None  # type: ignore[assignment, misc]
        Model = None  # type: ignore[assignment, misc]
        ImageEmbedding = None  # type: ignore[assignment, misc]
        Prompt = None  # type: ignore[assignment, misc]
        GenerateRequest = None  # type: ignore[assignment, misc]


class OsamSession:
    _model_name: str
    _model: "Model | None"  # type: ignore[name-defined]
    _embedding_cache: collections.deque[tuple[str, "ImageEmbedding"]]  # type: ignore[name-defined]

    def __init__(
        self,
        model_name: str = "sam2:latest",
        embedding_cache_size: int = 3,
    ) -> None:
        if osam is None:
            raise RuntimeError(
                "osam is not available. Please ensure onnxruntime is properly installed."
            )
        logger.debug("Initializing OsamSession with model_name={!r}", model_name)
        self._model_name = model_name
        self._model = None
        self._embedding_cache = collections.deque(maxlen=embedding_cache_size)
        logger.debug("Initialized OsamSession with model_name={!r}", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def run(
        self,
        image: NDArray[np.uint8],
        image_id: str,
        points: NDArray[np.floating] | None = None,
        point_labels: NDArray[np.intp] | None = None,
        texts: list[str] | None = None,
    ) -> "GenerateResponse":  # type: ignore[name-defined]
        image_embedding: "ImageEmbedding | None"  # type: ignore[name-defined]
        try:
            image_embedding = self._get_or_compute_embedding(
                image=image, image_id=image_id
            )
        except NotImplementedError:
            image_embedding = None

        prompt: "Prompt"  # type: ignore[name-defined]
        if points is not None and point_labels is not None:
            prompt = osam.types.Prompt(
                points=points,
                point_labels=point_labels,
            )
        elif texts is not None:
            prompt = osam.types.Prompt(
                texts=texts,
                iou_threshold=1.0,
                score_threshold=0.01,
                max_annotations=1000,
            )
        else:
            raise ValueError(
                "Either points and point_labels, or texts must be provided."
            )

        model: "Model" = self._get_or_load_model()  # type: ignore[name-defined]
        return model.generate(
            request=osam.types.GenerateRequest(
                model=model.name,
                image=image,
                image_embedding=image_embedding,
                prompt=prompt,
            )
        )

    def _get_or_compute_embedding(
        self, image: NDArray[np.uint8], image_id: str
    ) -> "ImageEmbedding":  # type: ignore[name-defined]
        for key, embedding in self._embedding_cache:
            if key == image_id:
                return embedding

        model: "Model" = self._get_or_load_model()  # type: ignore[name-defined]
        logger.debug("Computing embedding for cache_key={!r}", image_id)
        embedding: "ImageEmbedding" = model.encode_image(image=image)  # type: ignore[name-defined]
        self._embedding_cache.append((image_id, embedding))
        logger.debug("Cached embedding for cache_key={!r}", image_id)
        return embedding

    def _get_or_load_model(self) -> "Model":  # type: ignore[name-defined]
        if self._model is None:
            logger.debug("Loading model with name={!r}", self._model_name)
            self._model = osam.apis.get_model_type_by_name(self._model_name)()
            logger.debug("Loaded model with name={!r}", self._model_name)
        return self._model
