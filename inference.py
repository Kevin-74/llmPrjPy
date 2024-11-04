def run_inference(model, prompt):
    """Run inference using the trained model."""
    return model.generate_text(prompt)
