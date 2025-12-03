"""
Batch inference script for processing JSON data with Mussel model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def load_model(model_name: str):
    """
    Load the model and tokenizer from Hugging Face

    Args:
        model_name: The name of the model on Hugging Face Hub

    Returns:
        model: The loaded language model
        tokenizer: The corresponding tokenizer
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    print("Model loaded successfully!")
    return model, tokenizer


def inference(model, tokenizer, prompt: str, system_prompt: str = "", max_length: int = 2048):
    """
    Perform inference using the loaded model
    """
    # Combine system prompt with user prompt if system prompt is provided
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt

    # Tokenize the input
    inputs = tokenizer(full_prompt, return_tensors="pt")

    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_length = inputs['input_ids'].shape[1]

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return generated_text


def load_json_data(input_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from file

    Args:
        input_path: Path to the input JSON file

    Returns:
        data: List of dictionaries containing the JSON data
    """
    print(f"Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data


def save_json_data(output_path: str, data: List[Dict[str, Any]]):
    """
    Save processed data to JSON file

    Args:
        output_path: Path to save the output JSON file
        data: List of dictionaries to save
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} records")


def process_batch(model, tokenizer, data: List[Dict[str, Any]],
                  system_prompt: str, batch_size: int = None) -> List[Dict[str, Any]]:
    """
    Process a batch of data through the model

    Args:
        model: The loaded language model
        tokenizer: The tokenizer for the model
        data: List of data records to process
        system_prompt: System prompt to use for all inferences
        batch_size: Number of records to process (None means process all)

    Returns:
        results: List of dictionaries with original data plus model outputs
    """
    # Determine how many records to process
    num_to_process = min(batch_size, len(data)) if batch_size else len(data)

    print(f"\nProcessing {num_to_process} records...")
    results = []

    # Process each record
    for i, record in enumerate(tqdm(data[:num_to_process], desc="Processing")):
        # Get the input field
        input_text = record.get("input", "")

        # Perform inference
        try:
            generated_output = inference(model, tokenizer, input_text, system_prompt)

            # Create result record with model output and renamed original output
            result_record = record.copy()
            # Rename original 'output' key to 'label'
            if "output" in result_record:
                result_record["label"] = result_record.pop("output")
            result_record["model_output"] = generated_output
            result_record["processing_status"] = "success"

        except Exception as e:
            print(f"\nError processing record {i}: {str(e)}")
            result_record = record.copy()
            result_record["model_output"] = ""
            result_record["processing_status"] = f"error: {str(e)}"

        results.append(result_record)

    return results


def main():
    """
    Main function to run the batch inference pipeline
    """
    # ==================== CONFIGURATION ====================
    # Model configuration
    MODEL_NAME = "../models/mussel_deepseek"

    # Input/Output paths
    INPUT_JSON_PATH = "../data/mussel_test.json"
    OUTPUT_JSON_PATH = "../output/mussel_test_results.json"

    # Processing configuration
    BATCH_SIZE = 2000  # Set to None to process all records, or specify a number (e.g., 10, 50, 100)

    SYSTEM_PROMPT = "### Instruction: You are a helpful assistant. Your task is to analyze the provided code snippet, identify the section marked as a bug between the <vul-start> and <vul-end> tags, and generate a corrected version of the code. When you provide the fixed code snippet, prepend it with the <vul-start> tag.Ensure that the fix you propose is syntactically correct and resolves the issue marked as a bug, adhering to the best practices of code development."

    # Load the model and tokenizer
    model, tokenizer = load_model(MODEL_NAME)

    # Load input data
    input_data = load_json_data(INPUT_JSON_PATH)

    # Process the data
    results = process_batch(
        model=model,
        tokenizer=tokenizer,
        data=input_data,
        system_prompt=SYSTEM_PROMPT,
        batch_size=BATCH_SIZE
    )

    # Save results
    save_json_data(OUTPUT_JSON_PATH, results)

    print("\n" + "=" * 50)
    print("Processing completed successfully!")
    print(f"Total records processed: {len(results)}")
    print(f"Results saved to: {OUTPUT_JSON_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
