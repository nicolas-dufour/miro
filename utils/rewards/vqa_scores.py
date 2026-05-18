import t2v_metrics
import tempfile
import os
import re
from PIL import Image
from torchvision.utils import save_image
import torch
from typing import Union, List


class VQAScores(t2v_metrics.VQAScore):
    """
    Patching because the original VQAScore class is the worst spaghetti code I've ever seen.
    They for loop through the images and texts, instead of vectorizing properly...
    Also they only accept paths as inputs for some reason?
    To score 10k samples it takes 1h instead of 33h.
    """

    def __init__(self):
        super().__init__(model="clip-flant5-xxl")

    def forward(self, images, texts):
        if isinstance(images, Image.Image):
            with tempfile.TemporaryDirectory() as temp_dir:
                image_file = os.path.join(temp_dir, "image.jpg")
                images.save(image_file)
                scores = self.score([image_file], texts)
        elif isinstance(images, list) and all(
            isinstance(img, Image.Image) for img in images
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths = []
                # Need os for path joining, assuming it's imported elsewhere or standard enough
                import os

                for i, img in enumerate(images):
                    # Create a unique path for each image within the temporary directory
                    img_path = os.path.join(temp_dir, f"image_{i}.jpg")
                    # Save the PIL Image to the temporary file
                    img.save(img_path)
                    image_paths.append(img_path)

                # Call the VQA score model with the list of temporary image paths
                print(len(image_paths), len(texts), texts)
                scores = self.score(image_paths, texts)
            # The temporary directory and all its contents are automatically removed here
        elif isinstance(images, str) or (
            isinstance(images, list) and all(isinstance(img, str) for img in images)
        ):
            scores = self.score(images, texts)
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")
        return scores

    def score(
        self, images: Union[str, List[str]], texts: Union[str, List[str]], **kwargs
    ) -> torch.Tensor:
        """Return the similarity score(s) between the image(s) and the text(s)
        If there are m images and n texts, return a m x n tensor
        """
        if type(images) == str:
            images = [images]
        if type(texts) == str:
            texts = [texts]

        # Remove fenced code blocks delimited by triple backticks (including content)
        # Also remove unmatched opening fences and strip remaining backtick characters
        def _strip_code_fences(text: str) -> str:
            if not isinstance(text, str):
                return text
            text = re.sub(r"```[\s\S]*?```", "", text)
            text = re.sub(r"```[\s\S]*$", "", text)
            # Remove stray XML-like tags such as <image>, <binary-content>, etc.
            text = re.sub(r"<[^>]+>", " ", text)
            # Remove any remaining backticks and collapse whitespace
            text = text.replace("`", "")
            text = re.sub(r"\s{2,}", " ", text).strip()
            return text

        texts = [_strip_code_fences(t) for t in texts]
        scores = self.model.forward(images, texts, **kwargs)
        return scores
