# Usage

**Author: Chong Kiat Lim**

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended) or CPU (slower inference)
- Hugging Face account (for model downloads)

## Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd generative-ai-image-inpainting-generation
```

2. **Create and activate a virtual environment**

```bash
python -m venv .genai_inpainting_venv
# Windows
.\.genai_inpainting_venv\Scripts\activate
# Linux/macOS
source .genai_inpainting_venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy or edit the `.env` file and add your Hugging Face token:

```env
HF_TOKEN=hf_your_token_here
SAM_MODEL_NAME=facebook/sam-vit-base
INPAINTING_MODEL_NAME=runwayml/stable-diffusion-inpainting
```

Get your token at: https://huggingface.co/settings/tokens

## Running the Application

```bash
python GenAI_Image_InPainting_application.py
```

The application will launch a Gradio web interface (with a public share link).

## How to Use

![Application GUI](images/example_gui.jpg)

### Step-by-Step

1. **Upload an image** — Click on the Input canvas or drag and drop an image
2. **Select the subject** — Click on the object you want to segment. SAM runs automatically and shows the mask in the SAM result canvas. Add more points to refine the mask if needed
3. **Write a prompt** — Describe what you want to generate in the masked area (e.g., "Construction zone on the road, with traffic barriers")
4. **Adjust settings** (optional):
   - **Classifier-Free Guidance Scale** (0–50): Higher values follow the prompt more closely
   - **Random Seed**: For reproducible results
   - **Number of Generation Steps** (1–999): More steps = higher quality but slower
   - **Infill Subject instead of Background**: Toggle to replace the subject rather than the background
   - **Add Generative AI Watermark**: Adds an "AI Generated Image" label to the output
5. **Click "Run Inpaint"** — Wait for generation (up to a few minutes depending on image size and hardware)
6. **Download the result** — Right-click the Output canvas to save

### Tips

- Use a **Negative Prompt** to avoid unwanted features (e.g., "low quality, distortion, artifacts")
- Click **🎲 Generate Random Seed** to try different variations
- Use **Reset** to clear everything and start over
- For best results, use images under 1024×1024 pixels
