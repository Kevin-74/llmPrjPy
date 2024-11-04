from data_preprocessing import load_data, preprocess_data
from model import LLM
from training import train_model
from inference import run_inference

def main():
    # Load and preprocess data
    data = load_data('data.csv')
    processed_data = preprocess_data(data)

    # Initialize the model
    model = LLM('gpt2')

    # Train the model
    train_model(model, processed_data)

    # Run inference
    prompt = "Once upon a time"
    generated_text = run_inference(model, prompt)
    print(generated_text)

if __name__ == "__main__":
    main()
