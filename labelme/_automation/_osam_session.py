from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np
try:
    import osam
except (ImportError, OSError, RuntimeError):
    osam = None  # type: ignore[assignment]
from loguru import logger
from numpy.typing import NDArray

if TYPE_CHECKING:
    try:
        from osam.types import GenerateResponse, ImageEmbedding, Model
    except ImportError:
        GenerateResponse = None  # type: ignore[assignment, misc]
        ImageEmbedding = None  # type: ignore[assignment, misc]
        Model = None  # type: ignore[assignment, misc]
else:
    GenerateResponse = None  # type: ignore[assignment, misc]
    ImageEmbedding = None  # type: ignore[assignment, misc]
    Model = None  # type: ignore[assignment, misc]


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
                "AI features are not available. onnxruntime failed to load. "
                "Please ensure Visual C++ Redistributable is installed and "
                "onnxruntime is properly configured."
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
    ) -> "GenerateResponse | None":  # type: ignore[name-defined]
        image_embedding: "ImageEmbedding | None"  # type: ignore[name-defined]
        try:
            image_embedding = self._get_or_compute_embedding(
                image=image, image_id=image_id
            )
        except NotImplementedError:
            image_embedding = None

        if osam is None:
            raise RuntimeError("osam is not available")
        
        prompt = osam.types.Prompt(  # type: ignore[attr-defined]
            points=points,
            point_labels=point_labels,
        ) if points is not None and point_labels is not None else osam.types.Prompt(  # type: ignore[attr-defined]
            texts=texts,
            iou_threshold=1.0,
            score_threshold=0.01,
            max_annotations=1000,
        ) if texts is not None else None
        
        if prompt is None:
            raise ValueError(
                "Either points and point_labels, or texts must be provided."
            )

        model = self._get_or_load_model()
        return model.generate(
            request=osam.types.GenerateRequest(  # type: ignore[attr-defined]
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

        model = self._get_or_load_model()
        logger.debug("Computing embedding for cache_key={!r}", image_id)
        embedding = model.encode_image(image=image)
        self._embedding_cache.append((image_id, embedding))
        logger.debug("Cached embedding for cache_key={!r}", image_id)
        return embedding

    def _get_or_load_model(self) -> "Model":  # type: ignore[name-defined]
        if self._model is None:
            logger.debug("Loading model with name={!r}", self._model_name)
            self._model = osam.apis.get_model_type_by_name(self._model_name)()  # type: ignore[attr-defined]
            logger.debug("Loaded model with name={!r}", self._model_name)
        return self._model
