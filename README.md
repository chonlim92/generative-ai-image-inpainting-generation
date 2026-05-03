# GenAI Image Inpainting & Generation

**Author: Chong Kiat Lim**

An application that applies Visual-Language-Model in Generative AI to modify a given input image using a natural language prompt.

![Application GUI](documents/images/example_gui.jpg)

---

## Overview

This project combines **Meta's Segment Anything Model (SAM)** for object segmentation with **Stable Diffusion Inpainting** for text-guided image generation. Users can select objects in an image and replace either the subject or background using natural language prompts.

---

## Implementation

### Architecture

```
User Input (Image + Click Points)
        │
        ▼
┌─────────────────────────┐
│   SAM (Meta)            │  ← Object segmentation via point prompts
│   facebook/sam-vit-base │
└──────────┬──────────────┘
           │ Binary Mask
           ▼
┌──────────────────────────────────────────┐
│   Stable Diffusion Inpainting            │  ← Text-guided image generation
│   runwayml/stable-diffusion-inpainting   │
└──────────┬───────────────────────────────┘
           │ Generated Image
           ▼
     Output (with optional AI watermark)
```

### Components

| File | Purpose |
|------|---------|
| `GenAI_Image_InPainting_application.py` | Main entry point — loads models, defines inference functions, launches app |
| `app.py` | Gradio UI class and layout — handles user interactions and rendering |
| `.env` | Environment variables for tokens and model configuration |
| `requirements.txt` | Python dependencies |

### Key Details

- **Device handling**: Automatically selects CUDA GPU if available, falls back to CPU
- **Resolution handling**: Dimensions rounded to multiples of 8 for diffusion model compatibility
- **Mask inversion**: Infill subject or background by toggling a checkbox
- **Watermark**: AI-generated content watermark with automatic contrast adaptation
- **Environment variables**: Model names and HF token loaded from `.env` with defaults

---

## Usage

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) or CPU
- Hugging Face account

### Installation

```bash
# Clone and enter directory
git clone <repository-url>
cd generative-ai-image-inpainting-generation

# Create virtual environment
python -m venv .genai_inpainting_venv

# Activate (Windows)
.\.genai_inpainting_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `.env` with your Hugging Face token:

```env
HF_TOKEN=hf_your_token_here
SAM_MODEL_NAME=facebook/sam-vit-base
INPAINTING_MODEL_NAME=runwayml/stable-diffusion-inpainting
```

### Running

```bash
python GenAI_Image_InPainting_application.py
```

### How to Use

1. **Upload an image** — Click the Input canvas or drag and drop
2. **Select the subject** — Click on the object to segment (SAM runs automatically)
3. **Write a prompt** — Describe what to generate in the masked area
4. **Adjust settings** — CFG scale, seed, steps, infill mode, watermark toggle
5. **Click "Run Inpaint"** — Wait for generation
6. **Download the result** — Save from the Output canvas

---

## Tech Stack

| Skill (AI Focus) | Description |
|------------------|-------------|
| Image Segmentation (SAM) | Zero-shot object segmentation using point prompts with Meta's Segment Anything Model |
| Diffusion-based Image Inpainting | Text-guided image generation in masked regions using Stable Diffusion |
| Prompt Engineering | Positive and negative prompts to control generation quality and content |
| Classifier-Free Guidance | Adjustable CFG scale to balance prompt adherence vs. image diversity |
| NSFW Content Filtering | Built-in safety guard that detects and blocks unsafe generated content |
| AI Content Watermarking | Automatic watermark overlay with adaptive contrast for generated images |
| Interactive ML Web Application | Real-time Gradio-based UI for model inference with point-and-click interaction |
| GPU/CPU Device Management | Automatic hardware detection and model placement for optimal inference |
| Foundation Model Integration | Combining multiple pre-trained foundation models in a single pipeline |
