"""
Simple inference script for Mussel model using Hugging Face Transformers
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_name: str):
    """
    Load the Qwen model and tokenizer from Hugging Face

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
        device_map="auto",  # Automatically distribute model across available devices
        trust_remote_code=True,
        torch_dtype=torch.float16  # Use half precision to save memory
    )

    print("Model loaded successfully!")
    return model, tokenizer


def inference(model, tokenizer, prompt: str, max_length: int = 2048, system_prompt: str = None):
    """
    Perform inference using the loaded model
    """
    if system_prompt is None:
        system_prompt = "### Instruction: You are a helpful assistant..."

    full_prompt = f"{system_prompt}\n{prompt}\n"
    print(f"\nInput prompt: {full_prompt}")
    print("-" * 50)

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    if "<|EOT|>" in generated_text:
        generated_text = generated_text.split("<|EOT|>")[0]

    return generated_text


def main():
    """
    Main function to run the inference pipeline
    """
    # Model configuration
    MODEL_NAME = "models/mussel_deepseek"  # You can change to other Qwen models
    # System prompt (optional)
    SYSTEM_PROMPT = "### Instruction: You are a helpful assistant. Your task is to analyze the provided code snippet, identify the section marked as a bug between the <vul-start> and <vul-end> tags, and generate a corrected version of the code. When you provide the fixed code snippet, prepend it with the <vul-start> tag.Ensure that the fix you propose is syntactically correct and resolves the issue marked as a bug, adhering to the best practices of code development."

    # Hardcoded input prompts for testing
    prompts = [
        "###Input: CWE-125 Code Input Vulnerable Code Is: CWE-125 int obj2ast_keyword ( PyObject * obj , keyword_ty * out , PyArena * arena ) { PyObject * tmp = NULL ; identifier arg ; expr_ty value ; <vul-start> if ( exists_not_none ( obj , & PyId_arg ) ) { <vul-end> int res ; <vul-start> tmp = _PyObject_GetAttrId ( obj , & PyId_arg ) ; <vul-end> if ( tmp == NULL ) goto failed ; <vul-start> res = obj2ast_identifier ( tmp , & arg , arena ) ; <vul-end> if ( res != 0 ) goto failed ; Py_CLEAR ( tmp ) ; <vul-start> } else { <vul-end> arg = NULL ; } if ( _PyObject_HasAttrId ( obj , & PyId_value ) ) { int res ; tmp = _PyObject_GetAttrId ( obj , & PyId_value ) ; if ( tmp == NULL ) goto failed ; res = obj2ast_expr ( tmp , & value , arena ) ; if ( res != 0 ) goto failed ; Py_CLEAR ( tmp ) ; } else { PyErr_SetString ( PyExc_TypeError , \"required field \\\\\"value\\\\\" missing from keyword\" ) ; return 1 ; } * out = keyword ( arg , value , arena ) ; return 0 ; failed : Py_XDECREF ( tmp ) ; return 1 ; } \n ###Response:",
        "###Input: CWE-20 Code Input Vulnerable Code Is: CWE-20 <vul-start> static int read_fragment_table ( long long * directory_table_end ) <vul-end> { int res , i ; <vul-start> int bytes = SQUASHFS_FRAGMENT_BYTES ( sBlk . s . fragments ) ; <vul-end> <vul-start> int indexes = SQUASHFS_FRAGMENT_INDEXES ( sBlk . s . fragments ) ; <vul-end> long long fragment_table_index [ indexes ] ; TRACE ( \"read_fragment_table: %d fragments, reading %d fragment indexes \" \"from 0x%llx\\\\n\" , sBlk . s . fragments , indexes , sBlk . s . fragment_table_start ) ; <vul-start> if ( sBlk . s . fragments == 0 ) { <vul-end> * directory_table_end = sBlk . s . fragment_table_start ; return TRUE ; } fragment_table = malloc ( bytes ) ; if ( fragment_table == NULL ) EXIT_UNSQUASH ( \"read_fragment_table: failed to allocate \" \"fragment table\\\\n\" ) ; res = read_fs_bytes ( fd , sBlk . s . fragment_table_start , <vul-start> SQUASHFS_FRAGMENT_INDEX_BYTES ( sBlk . s . fragments ) , <vul-end> fragment_table_index ) ; if ( res == FALSE ) { ERROR ( \"read_fragment_table: failed to read fragment table \" \"index\\\\n\" ) ; return FALSE ; } SQUASHFS_INSWAP_FRAGMENT_INDEXES ( fragment_table_index , indexes ) ; for ( i = 0 ; i < indexes ; i ++ ) { int expected = ( i + 1 ) != indexes ? SQUASHFS_METADATA_SIZE : bytes & ( SQUASHFS_METADATA_SIZE - 1 ) ; int length = read_block ( fd , fragment_table_index [ i ] , NULL , expected , ( ( char * ) fragment_table ) + ( i * SQUASHFS_METADATA_SIZE ) ) ; TRACE ( \"Read fragment table block %d, from 0x%llx, length %d\\\\n\" , i \n ###Response:",
        "###Input: CWE-000 Code Input Vulnerable Code Is: CWE-000 static int ras_getcmap ( jas_stream_t * in , ras_hdr_t * hdr , ras_cmap_t * cmap ) { int i ; int j ; int x ; int c ; int numcolors ; int actualnumcolors ; switch ( hdr -> maptype ) { case RAS_MT_NONE : break ; case RAS_MT_EQUALRGB : { jas_eprintf ( \"warning: palettized images not fully supported\\\\n\" ) ; numcolors = 1 << hdr -> depth ; <vul-start> assert ( numcolors <= RAS_CMAP_MAXSIZ ) ; <vul-end> actualnumcolors = hdr -> maplength / 3 ; for ( i = 0 ; i < numcolors ; i ++ ) { cmap -> data [ i ] = 0 ; } if ( ( hdr -> maplength % 3 ) || hdr -> maplength < 0 || hdr -> maplength > 3 * numcolors ) { return - 1 ; } for ( i = 0 ; i < 3 ; i ++ ) { for ( j = 0 ; j < actualnumcolors ; j ++ ) { if ( ( c = jas_stream_getc ( in ) ) == EOF ) { return - 1 ; } x = 0 ; switch ( i ) { case 0 : x = RAS_RED ( c ) ; break ; case 1 : x = RAS_GREEN ( c ) ; break ; case 2 : x = RAS_BLUE ( c ) ; break ; } cmap -> data [ j ] |= x ; } } } break ; default : return - 1 ; break ; } return 0 ; } \n ###Response:",
    ]

    # Load the model and tokenizer
    model, tokenizer = load_model(MODEL_NAME)

    # Perform inference for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 50}")
        print(f"Example {i}:")
        print(f"{'=' * 50}")

        result = inference(model, tokenizer, prompt, system_prompt=SYSTEM_PROMPT)

        print(f"\nGenerated response:\n{result}")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
