import os
import sys

from transformers import AutoTokenizer


# --- Configuration ---
# The identifier for the Llama 3.1 8B model.
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" 

INPUT_FILE = "/projectnb/cs599m1/projects/color-representations/data/colors_single_word.txt"
OUTPUT_FILE = "/projectnb/cs599m1/projects/color-representations/data/colors_single_token.txt"

# --- Tokenizer Initialization and Logic ---
def get_tokenizer():
    """Initializes and returns the Llama 3.1 8B tokenizer."""
    
    print(f"Loading tokenizer for model: {MODEL_NAME}...")
    try:
        # Load the tokenizer from the Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please ensure you have access/permissions for the Llama 3.1 model.")
        return None

def is_single_token(tokenizer, word: str) -> bool:
    """
    Checks if a given word tokenizes to exactly one token ID.
    
    The ' ' (space) prefix is CRUCIAL for Llama/BPE tokenizers, as it simulates
    the word appearing after a space (i.e., at the start of a sentence or after a previous word).
    """
    if not tokenizer:
        return False
        
    # Encode the word with a leading space and without adding special tokens
    tokens = tokenizer.encode(' ' + word, add_special_tokens=False)
    
    # Check if the result is exactly one token
    return len(tokens) == 1


# --- Main Script Logic ---
def filter_colors_by_token(input_path: str, output_path: str):
    """
    Reads a list of words, filters for single-token words using the Llama 3.1 tokenizer,
    and writes the result to a new file.
    """

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found. Please create it first.")
        return

    tokenizer = get_tokenizer()
    if not tokenizer:
        return # Exit if tokenizer loading failed

    single_token_colors = []

    try:
        with open(input_path, 'r') as infile:
            words = [line.strip().lower() for line in infile if line.strip()]

        print(f"Read {len(words)} words from '{input_path}'. Starting Llama 3.1 token check.")

        for word in words:
            if is_single_token(tokenizer, word):
                single_token_colors.append(word)

        # Write the filtered list to the output file
        with open(output_path, 'w') as outfile:
            outfile.write('\n'.join(single_token_colors) + '\n')

        print("-" * 60)
        print(f"Filtering complete.")
        print(f"Total single-token words found: {len(single_token_colors)}")
        print(f"Results saved to '{output_path}'.")
        print("-" * 60)

    except IOError as e:
        print(f"An I/O error occurred during file operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

if __name__ == "__main__":
    filter_colors_by_token(INPUT_FILE, OUTPUT_FILE)