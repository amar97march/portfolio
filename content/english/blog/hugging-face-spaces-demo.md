---
title: "Demoing Your AI Model with Hugging Face Spaces"
date: 2028-10-02T10:00:00+05:30
draft: false
description: "Learn how to create interactive demos for your AI models using Hugging Face Spaces. A live demo transforms a static project into an experience that hiring managers and collaborators can interact with directly."
tags: ["Hugging Face", "AI Portfolio", "Gradio", "Model Deployment", "Demo"]
categories: ["AI Portfolio"]
image: "/images/blogs/pool-portfolio/1.jpg"
keywords: ["Hugging Face Spaces", "AI model demo", "Gradio demo", "deploy ML model", "interactive AI demo", "portfolio demo"]
---

There is one thing that elevates an AI portfolio project from good to memorable: a live demo. When someone can interact with your model — type in text, upload an image, and see results — the impact is fundamentally different from reading a README or scrolling through a notebook.

Hugging Face Spaces makes this trivially easy. You can go from a trained model to a live, shareable demo in under an hour, completely free.

### Why Live Demos Matter

**Demos make your work tangible.** A model accuracy of 94% is an abstract number. Watching someone type a sentence and see the model correctly classify its sentiment in real-time makes the achievement concrete.

**Demos are shareable.** You can paste a Hugging Face Spaces URL into a resume, a LinkedIn post, a job application, or an email. The recipient can try your model without installing anything.

**Demos reveal honesty.** When your model is live, people can test edge cases you might not have mentioned. This actually works in your favor — if your model handles edge cases well, it demonstrates robustness. If it fails gracefully on out-of-distribution inputs, it shows mature engineering.

**Demos start conversations.** In interviews, being able to pull up a live demo and say "let me show you" is enormously more powerful than describing your project verbally.

### What is Hugging Face Spaces?

Hugging Face Spaces is a free hosting platform for machine learning demos. It supports three frameworks:

- **Gradio**: The easiest option for building ML interfaces. A few lines of Python create a fully functional web app.
- **Streamlit**: A more general-purpose app framework with more layout control.
- **Docker**: For custom applications that do not fit the other frameworks.

For most AI demos, Gradio is the best choice. It is specifically designed for ML interfaces and handles common patterns (text input/output, image upload, audio recording) with minimal code.


![Developer showcasing an interactive AI project demonstration](/images/blogs/pool-portfolio/3.jpg)

### Building Your First Gradio Demo

Let us build a complete demo for a text classification model:

```python
# app.py — This single file is all you need for a Hugging Face Space
import gradio as gr
from transformers import pipeline

# Load model
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None  # Return all class probabilities
)

def classify_text(text):
    """Classify text sentiment and return formatted results."""
    if not text.strip():
        return "Please enter some text to analyze."

    results = classifier(text)
    output = []
    for result in results[0]:
        label = result['label']
        score = result['score']
        bar = "█" * int(score * 20)
        output.append(f"{label}: {score:.1%} {bar}")

    return "\n".join(output)

# Build the interface
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(
        label="Enter text",
        placeholder="Type a movie review, product review, or any text...",
        lines=4
    ),
    outputs=gr.Textbox(label="Sentiment Analysis"),
    title="Sentiment Analyzer",
    description=(
        "Analyze the sentiment of any English text using a fine-tuned "
        "DistilBERT model. The model classifies text as POSITIVE or "
        "NEGATIVE with a confidence score."
    ),
    examples=[
        ["This movie was absolutely brilliant! The acting was superb."],
        ["Terrible experience. The product broke after one day."],
        ["The food was okay, nothing special but not bad either."],
    ],
    theme=gr.themes.Soft()
)

demo.launch()
```

### Deploying to Hugging Face Spaces

The deployment process is straightforward:

**Step 1**: Create a Hugging Face account at huggingface.co.

**Step 2**: Create a new Space. Choose Gradio as the SDK.

**Step 3**: Push your code. A Space is essentially a Git repository. You need:

```
your-space/
├── app.py              # Your Gradio application
├── requirements.txt    # Python dependencies
└── README.md           # Space metadata (auto-generated)
```

Your `requirements.txt`:
```
gradio>=4.0.0
transformers
torch
```

**Step 4**: Push to the Space repository:
```bash
git clone https://huggingface.co/spaces/yourusername/sentiment-analyzer
cd sentiment-analyzer
# Copy your files
git add .
git commit -m "Initial demo"
git push
```

Within minutes, your demo is live at `https://huggingface.co/spaces/yourusername/sentiment-analyzer`.


![Illustration of a live model deployment accessible via web interface](/images/blogs/pool-portfolio/5.jpg)

### Demo Patterns for Different Model Types

**Image classification**:
```python
import gradio as gr
from transformers import pipeline

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def classify_image(image):
    results = classifier(image)
    return {r['label']: r['score'] for r in results}

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classifier"
)
```

**Text generation**:
```python
import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate(prompt, max_length, temperature):
    result = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1
    )
    return result[0]['generated_text']

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(50, 500, value=150, label="Max Length"),
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text")
)
```

**Question answering with context**:
```python
import gradio as gr
from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(context, question):
    result = qa(question=question, context=context)
    return f"{result['answer']} (confidence: {result['score']:.1%})"

demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Context", lines=8,
                   placeholder="Paste a paragraph of text..."),
        gr.Textbox(label="Question",
                   placeholder="Ask a question about the text..."),
    ],
    outputs=gr.Textbox(label="Answer")
)
```

### Making Your Demo Stand Out

**1. Add examples.** Pre-loaded examples let visitors try your model instantly without thinking of inputs. Choose examples that showcase your model's strengths.

**2. Handle errors gracefully.** What happens when someone enters empty text or uploads a non-image file? Handle these cases with clear error messages.

**3. Show confidence scores.** Do not just show the prediction — show how confident the model is. This demonstrates awareness of model uncertainty.

**4. Add a description.** Explain what the model does, what data it was trained on, and any limitations. This context makes the demo more meaningful.

**5. Choose a good theme.** Gradio offers several built-in themes. `gr.themes.Soft()` and `gr.themes.Glass()` look professional without extra effort.

**6. Include a link to the code.** Add a link to your GitHub repository in the description so interested visitors can explore the implementation.


![Visual representation of a portfolio project with real-time predictions](/images/blogs/pool-portfolio/7.jpg)

### Beyond Simple Demos

For more complex projects, consider:

**Tabbed interfaces**: Show multiple capabilities in one demo.

```python
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Model AI Demo")

    with gr.Tab("Sentiment Analysis"):
        text_input = gr.Textbox(label="Text")
        sentiment_output = gr.Label()
        text_input.submit(classify_text, text_input, sentiment_output)

    with gr.Tab("Image Classification"):
        image_input = gr.Image(type="pil")
        image_output = gr.Label()
        image_input.change(classify_image, image_input, image_output)
```

**Comparison views**: Let users compare your model against a baseline or different model versions.

**Interactive visualizations**: Use Gradio's Plot component to show attention maps, feature importance, or other model interpretability tools.

### Performance Considerations

Hugging Face Spaces free tier has limited resources (2 vCPU, 16GB RAM). Keep these tips in mind:

- Use smaller models (DistilBERT instead of BERT, MobileNet instead of ResNet-152).
- Load models once at startup, not on every request.
- Consider using ONNX runtime for faster inference.
- If your model is large, use Hugging Face's GPU Spaces (paid but affordable).

### Final Thoughts

A live demo transforms your AI project from a static repository into an interactive experience. It takes a few hours to set up but pays dividends every time someone visits your portfolio. It is the closest thing to letting someone sit at your desk and try your work themselves.

Build the demo. Share the link. Let your work speak for itself.

In the next series of posts, we shift to AI interview preparation, starting with coding challenges in Python and SQL.
