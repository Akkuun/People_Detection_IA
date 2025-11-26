import torch
from diffusers import StableDiffusionImg2ImgPipeline

class MugshotGeneratorAI:
    def __init__(self, model_path="runwayml/stable-diffusion-v1-5"):
        print("Initialisation du modèle IA Mugshot CPU…")

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,   # IMPORTANT: CPU friendly
            safety_checker=None,         # évite erreurs safety
            feature_extractor=None       # évite erreurs CLIP
        )

        self.pipe = self.pipe.to("cpu")
        print("✔ Modèle SD1.5 chargé (CPU, Img2Img OK).")

    def generate_mugshot(self, image, orientation="front"):
        prompt = (
            f"professional ID photo, {orientation} view, neutral expression, "
            f"sharp lighting, gray background, realistic skin texture"
        )
        negative = "blur, distortion, extra face, deformed, ugly"

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=image,
            strength=0.45,
            guidance_scale=7.5
        )

        return result.images[0]
