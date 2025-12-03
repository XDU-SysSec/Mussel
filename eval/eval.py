import json
import os
import re
import string
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def normalize_answer(s):
    """
    Normalize the answer by removing articles, punctuation, extra whitespace, and converting to lowercase.

    Args:
        s: Input string to normalize

    Returns:
        Normalized string
    """

    def remove_articles(text):
        # Remove common articles
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # Remove extra whitespace
        return ' '.join(text.split())

    def remove_punc(text):
        # Remove all punctuation
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        # Convert to lowercase
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    """
    Calculate exact match score between prediction and ground truth after normalization.

    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer string

    Returns:
        Boolean indicating exact match
    """
    prediction_norm = normalize_answer(prediction)
    ground_norm = normalize_answer(ground_truth)
    return prediction_norm == ground_norm


def ems(prediction, ground_truths):
    """
    Calculate exact match score with multiple ground truths.

    Args:
        prediction: Predicted answer string
        ground_truths: List of ground truth answer strings

    Returns:
        Maximum exact match score across all ground truths
    """
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def calculate_bleu_score(prediction, ground_truth):
    """
    Calculate BLEU score between prediction and ground truth.
    Uses original text without normalization.
    Uses character-level tokenization for better accuracy.

    Args:
        prediction: Predicted answer string (original, not normalized)
        ground_truth: Ground truth answer string (original, not normalized)

    Returns:
        Dictionary containing BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    # Use character-level tokenization (same as reference code)
    pred_tokens = list(prediction)
    ref_tokens = list(ground_truth)

    # Handle empty cases
    if not pred_tokens or not ref_tokens:
        return {
            'bleu-1': 0.0,
            'bleu-2': 0.0,
            'bleu-3': 0.0,
            'bleu-4': 0.0
        }

    # Use smoothing function method3 (same as reference code)
    smoothing = SmoothingFunction().method3

    # Calculate BLEU scores with different n-gram weights
    bleu_scores = {}

    try:
        # BLEU-1 (unigram)
        bleu_scores['bleu-1'] = sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothing
        )

        # BLEU-2 (bigram)
        bleu_scores['bleu-2'] = sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing
        )

        # BLEU-3 (trigram)
        bleu_scores['bleu-3'] = sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(0.33, 0.33, 0.33, 0),
            smoothing_function=smoothing
        )

        # BLEU-4 (4-gram) - standard weights
        bleu_scores['bleu-4'] = sentence_bleu(
            [ref_tokens], pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing
        )
    except Exception as e:
        print(f"Warning: Error calculating BLEU score: {e}")
        bleu_scores = {
            'bleu-1': 0.0,
            'bleu-2': 0.0,
            'bleu-3': 0.0,
            'bleu-4': 0.0
        }

    return bleu_scores


def calculate_metrics(input_file, output_dir):
    """
    Read JSON file, extract labels and model outputs, normalize them,
    and calculate EM and BLEU metrics.

    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save output results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the JSON file
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract labels and model outputs
    golds = []
    preds = []

    for item in data:
        # Extract label and model_output fields
        label = item.get('label', '')
        model_output = item.get('model_output', '')

        golds.append(label)
        preds.append(model_output)

    # Validate that we have matching number of predictions and labels
    assert len(golds) == len(preds), "Number of labels and predictions must match"

    # Calculate exact matches and BLEU scores
    count = []
    mismatch_after_norm = 0
    results = []

    # Aggregate BLEU scores
    total_bleu_scores = {
        'bleu-1': 0.0,
        'bleu-2': 0.0,
        'bleu-3': 0.0,
        'bleu-4': 0.0
    }

    for i in range(len(golds)):
        # Check if prediction matches gold standard (uses normalized text for EM)
        is_match = ems(preds[i], [golds[i]])
        count.append(1 if is_match else 0)

        # Check for cases where normalized versions don't match but EM score is True
        if is_match and (' '.join(golds[i].split()).strip().lower() !=
                         ' '.join(preds[i].split()).strip().lower()):
            mismatch_after_norm += 1

        # Calculate BLEU scores using ORIGINAL text with character-level tokenization
        bleu_scores = calculate_bleu_score(preds[i], golds[i])

        # Accumulate BLEU scores
        for key in total_bleu_scores:
            total_bleu_scores[key] += bleu_scores[key]

        # Store detailed results
        results.append({
            'index': i,
            'label': golds[i],
            'model_output': preds[i],
            'label_normalized': normalize_answer(golds[i]),
            'output_normalized': normalize_answer(preds[i]),
            'exact_match': bool(is_match),
            'bleu_scores': {
                'bleu-1': round(bleu_scores['bleu-1'], 4),
                'bleu-2': round(bleu_scores['bleu-2'], 4),
                'bleu-3': round(bleu_scores['bleu-3'], 4),
                'bleu-4': round(bleu_scores['bleu-4'], 4)
            }
        })

    # Calculate metrics
    total_count = len(count)
    match_count = sum(count)
    em_percentage = (match_count / total_count) * 100 if total_count > 0 else 0

    # Calculate average BLEU scores
    avg_bleu_scores = {
        key: round((value / total_count) * 100, 2) if total_count > 0 else 0.0
        for key, value in total_bleu_scores.items()
    }

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results (EM + BLEU)")
    print(f"{'=' * 60}")
    print(f"Total samples: {total_count}")
    print(f"\nExact Match Metrics:")
    print(f"  Exact matches: {match_count}")
    print(f"  EM percentage: {em_percentage:.2f}%")
    print(f"  Mismatches after normalization: {mismatch_after_norm}")
    print(f"\nBLEU Scores (averaged, character-level tokenization):")
    print(f"  BLEU-1: {avg_bleu_scores['bleu-1']:.2f}%")
    print(f"  BLEU-2: {avg_bleu_scores['bleu-2']:.2f}%")
    print(f"  BLEU-3: {avg_bleu_scores['bleu-3']:.2f}%")
    print(f"  BLEU-4: {avg_bleu_scores['bleu-4']:.2f}%")
    print(f"{'=' * 60}\n")

    # Prepare output data
    output_data = {
        'summary': {
            'total_samples': total_count,
            'exact_match': {
                'matches': match_count,
                'percentage': round(em_percentage, 2),
                'mismatches_after_normalization': mismatch_after_norm
            },
            'bleu_scores': avg_bleu_scores
        }
    }

    # Save results to output directory
    output_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

    # Also save a summary text file
    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Summary (EM + BLEU)\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Total samples: {total_count}\n\n")
        f.write(f"Exact Match Metrics:\n")
        f.write(f"  Exact matches: {match_count}\n")
        f.write(f"  EM percentage: {em_percentage:.2f}%\n")
        f.write(f"  Mismatches after normalization: {mismatch_after_norm}\n\n")
        f.write(f"BLEU Scores (averaged, character-level tokenization):\n")
        f.write(f"  BLEU-1: {avg_bleu_scores['bleu-1']:.2f}%\n")
        f.write(f"  BLEU-2: {avg_bleu_scores['bleu-2']:.2f}%\n")
        f.write(f"  BLEU-3: {avg_bleu_scores['bleu-3']:.2f}%\n")
        f.write(f"  BLEU-4: {avg_bleu_scores['bleu-4']:.2f}%\n")

    print(f"Summary saved to: {summary_file}")

    return em_percentage, avg_bleu_scores


if __name__ == "__main__":
    # Configuration
    input_file = "../output/mussel_test_results.json"
    output_dir = "output"

    # Run evaluation
    em_score, bleu_scores = calculate_metrics(input_file, output_dir)

    print(f"\nFinal Scores:")
    print(f"  EM Score: {em_score:.2f}%")
    print(f"  BLEU-1: {bleu_scores['bleu-1']:.2f}%")
    print(f"  BLEU-2: {bleu_scores['bleu-2']:.2f}%")
    print(f"  BLEU-3: {bleu_scores['bleu-3']:.2f}%")
    print(f"  BLEU-4: {bleu_scores['bleu-4']:.2f}%")