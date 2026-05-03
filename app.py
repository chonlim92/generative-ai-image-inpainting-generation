# Chong Kiat Lim, chong_kiat.lim@mercedes-benz.com

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import random

class InpaintGenAI:
    """
    InpaintGenAI class to handle image segmentation and inpainting using Generative AI.
    """
    def __init__(self, get_processed_inputs, inpaint):
        """
        Initialize the InpaintGenAI class with necessary functions.

        Args:
            get_processed_inputs (function): Function to process inputs for SAM.
            inpaint (function): Function to perform inpainting.
        """
        self.get_processed_inputs = get_processed_inputs
        self.inpaint = inpaint
        self.input_points = []
        self.input_image = None
        self.original_width = None
        self.original_height = None

    def get_points(self, img, evt: gr.SelectData):
        """
        Handle the event when points are selected on the image.

        Args:
            img (PIL.Image): The input image.
            evt (gr.SelectData): Event data containing the selected points.

        Returns:
            tuple: SAM output and the image with drawn points.
        """
        # Initialize input image if no points have been selected yet
        if len(self.input_points) == 0:
            self.input_image = img.copy()
            self.original_width, self.original_height = img.size
        
        print(f"Event data: {evt}")
        print(f"Event index: {evt.index}")
        
        # Validate event data and extract coordinates
        if isinstance(evt.index, (list, tuple)) and len(evt.index) == 2:
            x, y = evt.index
        else:
            raise gr.Error("Invalid event data. Expected a tuple with two values (x, y coordinates).")
        
        self.input_points.append([x, y])
        
        # Run SAM and draw points on the image
        sam_output = self.run_sam()
        
        draw = ImageDraw.Draw(img)
        size = 10
        for point in self.input_points:
            x, y = point
            draw.line((x - size, y, x + size, y), fill="green", width=5)
            draw.line((x, y - size, x, y + size), fill="green", width=5)
        
        return sam_output, img

    def run_sam(self):
        """
        Run SAM to generate the segmentation mask.

        Returns:
            tuple: Input image and segmentation mask.
        """
        if self.input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")
        
        try:
            mask = self.get_processed_inputs(self.input_image, [self.input_points])
            res_mask = np.array(mask)
            return (
                self.input_image, 
                [
                    (res_mask, "background"), 
                    (~res_mask, "subject")
                ]
            )
        except Exception as e:
            raise gr.Error(str(e))

    def run(self, prompt, negative_prompt, cfg, seed, invert, num_inference_steps, add_watermark):
        """
        Run the inpainting process.

        Args:
            prompt (str): Prompt for infill.
            negative_prompt (str): Negative prompt.
            cfg (float): Classifier-Free Guidance Scale.
            seed (int): Random seed.
            invert (bool): Whether to infill the subject instead of the background.
            num_inference_steps (int): Number of generation steps.
            add_watermark (bool): Whether to add watermark to the output image.

        Returns:
            PIL.Image: Inpainted image.
        """
        if self.input_image is None:
            raise gr.Error("No points provided. Click on the image to select the object to segment with SAM")
        
        amask = self.run_sam()[1][0][0]
        
        if bool(invert):
            what = 'subject'
            amask = ~amask
        else:
            what = 'background'
        
        gr.Info(f"Inpainting {what}... (this will take up to a few minutes)")
        try:
            inpainted = self.inpaint(self.input_image, amask, prompt, negative_prompt, seed, cfg, num_inference_steps, self.original_width, self.original_height)
        except Exception as e:
            raise gr.Error(str(e))

        # Ensure the output image retains the original resolution
        inpainted = inpainted.resize((self.original_width, self.original_height), Image.LANCZOS)

        # Check if the output image is empty or black
        if np.array(inpainted).mean() == 0:
            draw = ImageDraw.Draw(inpainted)
            font_size = 40  # Start with a large font size
            font = ImageFont.load_default()
            text = "Safe Guard: Potential Not-Safe-To-Work (NSFW) content was detected in the generated image. A black image will be returned instead. Try again with a different prompt and/or seed."
            
            # Wrap text to fit within the image
            max_width = self.original_width - 20  # Padding
            lines = []
            words = text.split()
            while words:
                line = ''
                while words and font.getbbox(line + words[0])[2] <= max_width:
                    line += (words.pop(0) + ' ')
                lines.append(line)
            
            # Adjust font size if necessary
            while font.getbbox(lines[0])[3] * len(lines) > self.original_height * 0.5:
                font_size -= 2
                font = ImageFont.load_default()
                lines = []
                words = text.split()
                while words:
                    line = ''
                    while words and font.getbbox(line + words[0])[2] <= max_width:
                        line += (words.pop(0) + ' ')
                    lines.append(line)
            
            # Draw text on the image
            y = (self.original_height - font.getbbox(lines[0])[3] * len(lines)) // 2
            for line in lines:
                width, height = font.getbbox(line)[2], font.getbbox(line)[3]
                x = (self.original_width - width) // 2
                draw.text((x, y), line, fill="white", font=font)
                y += height
        else:
            if add_watermark:
                # pre-load the black watermark
                black_watermark_path = 'resource/watermark/ai_generated_content_b.png'
                watermark = Image.open(black_watermark_path).convert("RGBA")
                
                # Calculate the maximum size for the watermark (3% of the image area)
                watermark_percentage_size = 3
                max_area = self.original_width * self.original_height * watermark_percentage_size / 100
                aspect_ratio = watermark.width / watermark.height
                watermark_height = int((max_area / aspect_ratio) ** (1 / 2))
                watermark_width = int(watermark_height * aspect_ratio)

                # Add watermark based on the brightness of the bottom-left corner
                crop_width = watermark_width + 3
                crop_height = watermark_height + 3
                bottom_left_corner = np.array(inpainted.crop((0, self.original_height - crop_height, crop_width, self.original_height)))
                # If the output image has dark bottom left corner, change the watermark to white
                if bottom_left_corner.mean() < 128:  
                    white_watermark_path = 'resource/watermark/ai_generated_content_w.png'
                    watermark = Image.open(white_watermark_path).convert("RGBA")

                
                # Resize the watermark
                watermark = watermark.resize((watermark_width, watermark_height), Image.LANCZOS)
                
                inpainted = inpainted.convert("RGBA")
                
                # Calculate position for the watermark
                start_watermark_position = 1
                position = (start_watermark_position, self.original_height - watermark.height - start_watermark_position)
                
                # Create a transparent layer for the watermark
                transparent = Image.new('RGBA', inpainted.size, (0, 0, 0, 0))
                transparent.paste(inpainted, (0, 0))
                transparent.paste(watermark, position, mask=watermark)
                inpainted = transparent.convert("RGB")

        return inpainted
    
    def reset_points(self, *args):
        """
        Reset the selected points and set checkboxes to True.
        """
        self.input_points.clear()
        return True, True  # Returning True for both checkboxes
    
    def preprocess(self, input_img):
        """
        Preprocess the input image.

        Args:
            input_img (PIL.Image): The input image.

        Returns:
            PIL.Image: Preprocessed image.
        """
        # No resizing to square, keep original dimensions
        self.original_width, self.original_height = input_img.size
        return input_img

def generate_app(get_processed_inputs, inpaint):
    """
    Generate the Gradio application.

    Args:
        get_processed_inputs (function): Function to process inputs for SAM.
        inpaint (function): Function to perform inpainting.

    Returns:
        gr.Blocks: Gradio Blocks object.
    """
    app = InpaintGenAI(get_processed_inputs, inpaint)

    title = "Prototype: GenAI Image Generation & Modification"
    
    js_func = """
    function refresh() {
        const url = new URL(window.location);

        if (!url.searchParams.has('__theme')) {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    with gr.Blocks(js=js_func, title=title) as genaiagent:
        gr.Markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <h1 style="font-size: 75px; margin-left: 10px; background: linear-gradient(45deg, #854F06, #A67C00, #854F06, #B8860B, #854F06); -webkit-background-clip: text; color: transparent;">{title}</h1>
        </div>
        <p>
            <strong>Creator: Chong Kiat Lim, RD/ASP</strong>
        </p>

        <br> 

        <p> 
            <strong>Instruction: </strong>
            <br><br>
            1. Upload an image by clicking on the input canvas or drag an image onto the input canvas.
            <br>
            2. Click on the subject on the uploaded image you would like to mask. SAM will immediately be run and you will see the masking results on the SAM result canvas. If you
            are happy with those results move to the next step, otherwise add more points to refine your mask.
            <br>
            3. Write a Prompt (and optionally a Negative Prompt) for what you want to generate for the infilling. 
            Adjust the CFG Scale and the Random Seed and the Number of Generation Steps if needed. You can also invert the mask, i.e., infill the subject 
            instead of the background by toggling the relative checkmark.
            <br>
            4. Click on "Run Inpaint" and wait for up to two minutes, depending on the image size and GPU availability. If you are not happy with the result, 
            change your prompts and/or the settings (CFG Scale, Random Seed, Number of Generation Steps) and click "Run Inpaint" again.
            <br>
            5. Download the generated image in the output canvas
        </p>

        <p>
            <strong><span style="color: red;">DISCLAIMER: Images in this application are created by Generative AI. Please handle them with correct legal and ethical manner!</span></strong>
        </p>

        """)

        with gr.Row():
            display_img = gr.Image(label="Input", interactive=True, type='pil')
            sam_mask = gr.AnnotatedImage(label="SAM result", color_map={"background": "#a89a00"})
            result = gr.Image(label="Output", interactive=False, type='pil')
        
        display_img.select(app.get_points, inputs=[display_img], outputs=[sam_mask, display_img])
        display_img.clear(app.reset_points)
        display_img.upload(app.preprocess, inputs=[display_img], outputs=[display_img])
        
        with gr.Row():
            cfg = gr.Slider(label="Classifier-Free Guidance Scale", minimum=0.0, maximum=50.0, value=7, step=0.05)
            with gr.Column():
                random_seed = gr.Number(label="Random Seed", value=20000, precision=0)
                dice_button = gr.Button(value="🎲 Generate Random Seed")
            num_inference_steps = gr.Slider(label="Number of Generation Steps", minimum=1, maximum=999, value=100, step=1)
            subject_checkbox = gr.Checkbox(label="Infill Subject instead of Background", value=True)
            watermark_checkbox = gr.Checkbox(label="Add Generative AI Watermark", value=True)

        with gr.Row():
            prompt = gr.Textbox(label="Prompt for Infill")
            neg_prompt = gr.Textbox(label="Negative Prompt")
            reset_points_b = gr.ClearButton(value="Reset", components=[display_img, sam_mask, result, prompt, neg_prompt, subject_checkbox, watermark_checkbox])
            reset_points_b.click(
                fn=lambda: app.reset_points(),
                outputs=[subject_checkbox, watermark_checkbox]
            )
            submit_inpaint = gr.Button(value="Run Inpaint")

        with gr.Row():
            examples = gr.Examples(
                [
                    ["example_image/example/mercedes-benz-cla.jpg", "moon landing", "artifacts, low quality, distortion", 1850922883, 12],
                    ["example_image/example/monalisa.png", "oil painted US President", "artifacts, low quality, colorful, distortion", 434709320, 7],
                    ["example_image/example/dragon.jpeg", "a dragon in a medieval village", "artifacts, low quality, distortion", 97, 7]
                ],
                inputs=[display_img, prompt, neg_prompt, random_seed, cfg]
            )

        def generate_random_seed():
            return random.randint(0, 4294967295)

        dice_button.click(fn=generate_random_seed, outputs=random_seed)
        submit_inpaint.click(fn=app.run, inputs=[prompt, neg_prompt, cfg, random_seed, subject_checkbox, num_inference_steps, watermark_checkbox], outputs=[result])

    genaiagent.queue(max_size=1).launch(share=True, debug=True)

    return genaiagent