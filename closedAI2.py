import os
import fitz  # PyMuPDF for PDF parsing
import numpy as np
import faiss  # FAISS for similarity search
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import requests
# from pdf2image import convert_from_path
# import pytesseract

def extract_text_from_pdfs(pdf_dir):
    """
    Extract text from all PDF files in a directory.
    Simple tables will be included as raw text.
    Returns a list of text chunks (strings).
    """
    text_chunks = []
    try:
        files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    except Exception as e:
        print(f"Error accessing directory {pdf_dir}: {e}")
        return []

    for filename in files:
        filepath = os.path.join(pdf_dir, filename)
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                if text:
                    # Each page's text is treated as one chunk (including any tables)
                    text_chunks.append(text)
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
    return text_chunks

# def extract_text_from_pdfs_with_ocr(pdf_dir):
#     text_chunks = []

#     try:
#         files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
#     except Exception as e:
#         print(f"Error accessing directory {pdf_dir}: {e}")
#         return []

#     for filename in files:
#         filepath = os.path.join(pdf_dir, filename)
#         try:
#             doc = fitz.open(filepath)
#             for page_num in range(len(doc)):
#                 page = doc.load_page(page_num)
#                 text = page.get_text().strip()
#                 if text:
#                     text_chunks.append(text)
#                 else:
#                     # No text detected — fallback to OCR
#                     images = convert_from_path(filepath, first_page=page_num + 1, last_page=page_num + 1)
#                     ocr_text = pytesseract.image_to_string(images[0])
#                     if ocr_text.strip():
#                         text_chunks.append(ocr_text.strip())
#         except Exception as e:
#             print(f"Failed to process {filepath}: {e}")
#     return text_chunks
def create_faiss_index(text_chunks, embed_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Embed text chunks and create an in-memory FAISS index.
    Returns the FAISS index and the list of text_chunks.
    """
    # Load the embedding model (downloads automatically if needed)
    embedder = SentenceTransformer(embed_model_name)
    # Compute embeddings for all text chunks (shape: [n_chunks, 384])
    embeddings = embedder.encode(text_chunks, convert_to_numpy=True)
    # Create a FAISS index for L2 similarity search
    dim = embeddings.shape[1]  # Should be 384 for all-MiniLM-L6-v2&#8203;:contentReference[oaicite:3]{index=3}
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)  # Add vectors to the index
    return index, text_chunks

def retrieve_chunks(query, index, text_chunks, embed_model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=5):
    """
    Retrieve the top_k chunks relevant to the query using FAISS similarity search.
    """
    # Embed the query using the same model
    embedder = SentenceTransformer(embed_model_name)
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    # Search the FAISS index for nearest neighbors
    distances, indices = index.search(query_embedding, top_k)
    indices = indices.flatten()
    # Collect the corresponding text chunks (if index is valid)
    results = []
    for idx in indices:
        if idx < len(text_chunks):
            results.append(text_chunks[idx])
    return results

def classify_requirement(requirement, evidence_chunks):
    """
    Use Flan-T5 (text2text-generation) to classify requirement satisfaction.
    Returns a string with [Satisfied/Not Satisfied/Missing] and an explanation.
    """
    # Load the Flan-T5 small model in a text2text-generation pipeline
    # generator = pipeline("text2text-generation", model="google/flan-t5-small")
    # Build a prompt that includes the requirement and bullet-pointed evidence
    # prompt = f"Requirement: {requirement}\nEvidence:\n"
    # big_chunk = ""
    # for chunk in evidence_chunks:
    #     big_chunk += chunk
    # prompt = ""
    # # print("---------------------------------------------------------------------------")
    # # print(big_chunk)
    # # print("----------------------------------------------------------------------------")
    # prompt += ("This is the requirement \n " + requirement + "\n this is the evidence \n" + big_chunk +
    #            "Answer with 'Satisfied', 'Not Satisfied', or 'Missing' and provide a brief explanation.")
    # # Generate the answer (deterministic output with do_sample=False)
    # print("------------------------------------------------------------------------------")
    # print(prompt)
    # print("-------------------------------------------------------------------------------")
    # result = generator(prompt)
    # print(result)
    prompt = (
    "You are a compliance assistant.\n"
    "Requirement: \n"
    f"{requirement}\n\n"
    "Evidence0:\n"
    f"{evidence_chunks[0]}\n\n"
    "Evidence1:\n"
    f"{evidence_chunks[1]}\n\n"
    "Evidence2:\n"
    f"{evidence_chunks[2]}\n\n"
    "1) First write the requirement at the beggining of the answer"
    "2) given the evidence, find if the requirement is: 'Satisfied', 'Not Satisfied', or 'Missing' and explain in short why.\n"
    "3) Which evidence among Evidence0, Evidence1, Evidence2 satisfies the requirement ?"
    )
    response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",       # Use the same name you use in the CLI
        "prompt": prompt,
        "stream": False          # Set to True if you want to stream output
    }
)

    # Extract and print the generated response
    print(response.json()["response"])
    

    return response.json()["response"]

import os
import fitz  # PyMuPDF

def find_pdf_and_page_for_chunk(text_chunk, pdf_dir, match_type='exact'):
    """
    Searches all PDFs in pdf_dir to find which PDF and page(s) contain the given text_chunk.

    Parameters:
        text_chunk (str): The chunk of text to search for.
        pdf_dir (str): Path to directory containing PDF files.
        match_type (str): 'exact' for exact match, 'partial' for substring match.

    Returns:
        List of tuples: [(pdf_filename, page_num), ...]
    """
    matches = []
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    for filename in files:
        filepath = os.path.join(pdf_dir, filename)
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if (
                    (match_type == 'exact' and text_chunk.strip() == text.strip()) or
                    (match_type == 'partial' and text_chunk.strip() in text)
                ):
                    matches.append((filename, page_num + 1))  # 1-based index
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return matches


pdf_directory = "./pdf_files"  
requirement_list = ["The organization must have important discussions in public",
                    "The organization must have 60 football grounds for sports.",
                    "The organization shall determine external and internal issues that are relevant to its purpose and that affect its ability to achieve the intended outcome(s) of its information security management system.",
                    ]


# Step 1: Extract text chunks from PDFs
chunks = extract_text_from_pdfs(pdf_directory)
    

# Step 2: Create embeddings and FAISS index
index, chunks = create_faiss_index(chunks)

for requirement_input in requirement_list:
    top_k = 3
    relevant_chunks = retrieve_chunks(requirement_input, index, chunks, top_k=top_k)
    # for i, chunk in enumerate(relevant_chunks, 1):
    #     print(f"{i}. {chunk[:200]}...")  # Print a snippet of each chunk
    #     print("++++++++++++++++++++++++++++++++++++++++++")

    classification = classify_requirement(requirement_input, relevant_chunks)

    chunk0_posi = find_pdf_and_page_for_chunk(relevant_chunks[0], pdf_directory, match_type='exact')
    print("Evidence0 is in", chunk0_posi)
    chunk1_posi = find_pdf_and_page_for_chunk(relevant_chunks[1], pdf_directory, match_type='exact')
    print("Evidence1 is in", chunk1_posi)
    chunk2_posi = find_pdf_and_page_for_chunk(relevant_chunks[2], pdf_directory, match_type='exact')
    print("Evidence2 is in", chunk2_posi)

    print("---------------------------------------------------------------------------------------")
