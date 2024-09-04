import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
from helper import embedding_text

import re

def chunk_sentences(sentences: list[str], chunk_size: int = 10) -> list[list[str]]:
    """
    Chunks a list of sentences into groups of a specified size.

    Parameters:
        sentences (list[str]): A list of sentences to be chunked.
        chunk_size (int): The number of sentences to include in each chunk. Default is 5.

    Returns:
        list[list[str]]: A list of lists, each containing a chunk of sentences.
    """
    return [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

def open_and_read_pdf(pdf_path: str, chunk_size: int = 10, lang=English()) -> list[dir]:
    """
    Opens a PDF file, reads its text content page by page, chunk sentences, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.
        lang (spacy.lang.Language): The language object from spacy. Default is English.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, the extracted text, the sentences, and the chunked groups of sentences
        for each page.
    """
    pdf_document = fitz.open(pdf_path)
    pdf_content = []
    # add sentencizer to the language object
    lang.add_pipe("sentencizer")
    # loop through the pdf page using enumerate
    for page_num in tqdm(enumerate(pdf_document), total=len(pdf_document), desc="Reading PDF"):
        page = pdf_document[page_num]
        text = page.get_text()
        clean_text = text.replace("\n", " ").strip()
        # get the sentences from the text using sentencizer
        sentences = list(str(lang(clean_text).sents))
        sentence_count = len(sentences)
        # chunk the sentences into groups of 10
        chunked_sentences = chunk_sentences(sentences, chunk_size)
        pdf_content.append(
            {
                "page": page_num + 1,
                "char_count": len(clean_text),
                "word_count": len(clean_text.split(" ")),
                "sentence_spacy_count": sentence_count,
                "chunk_count": chunk_size,
                "token_count": len(clean_text.split())/4, # rough estimate of tokens
                "text": clean_text,
                "sentences": sentences,
                "sentence_chunks": chunked_sentences,
            }
        )

    pdf_document.close()

    return pdf_content

def embed_chunks(pdf_content: list[dict]) -> list[dict]:

    # Split each chunk into its own item
    pages_and_chunks = []
    for item in tqdm(pdf_content):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            
            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo 
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
            
            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks
