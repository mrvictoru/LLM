import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
from helper import embedding_text

def open_and_read_pdf(pdf_path: str) -> list[dir]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    pdf_document = fitz.open(pdf_path)
    pdf_content = []
    # loop through the pdf page using enumerate
    for page_num in tqdm(enumerate(pdf_document), total=len(pdf_document), desc="Reading PDF"):
        page = pdf_document[page_num]
        text = page.get_text()
        clean_text = text.replace("\n", " ").strip()
        sentence_count = clean_text.count(".") + clean_text.count("!") + clean_text.count("?")

        pdf_content.append(
            {
                "page": page_num + 1,
                "char_count": len(clean_text),
                "word_count": len(clean_text.split(" ")),
                "sentence_count_raw": sentence_count,
                "token_count": len(clean_text.split())/4, # rough estimate of tokens
                "text": clean_text
            }
        )

    pdf_document.close()

    return pdf_content

