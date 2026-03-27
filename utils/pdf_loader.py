import logging
import re
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

def fix_doubled_text(text: str) -> str:
    """
    Fix doubled characters in PDF text extraction.
    Some PDFs produce text like 'RRAAGG' instead of 'RAG'.
    This detects and fixes that pattern.
    """
    if not text:
        return text
    
    # Check if text has the doubled-character pattern
    # Sample a portion and check if consecutive chars repeat
    sample = text[:500].replace(' ', '').replace('\n', '')
    if len(sample) < 10:
        return text
    
    # Count how many consecutive char pairs are duplicated
    double_count = 0
    total_pairs = 0
    for i in range(0, len(sample) - 1, 2):
        total_pairs += 1
        if sample[i] == sample[i + 1]:
            double_count += 1
    
    # If more than 60% of pairs are doubled, fix it
    if total_pairs > 0 and (double_count / total_pairs) > 0.6:
        logger.info("🔧 Detected doubled characters in PDF text, fixing...")
        # Remove every other character (the duplicates)
        fixed = ""
        i = 0
        while i < len(text):
            fixed += text[i]
            # If next char is the same (and not whitespace/newline), skip it
            if i + 1 < len(text) and text[i] == text[i + 1] and text[i] not in ' \n\r\t':
                i += 2  # Skip the duplicate
            else:
                i += 1
        logger.info(f"📝 Fixed text: {len(text)} -> {len(fixed)} chars")
        return fixed
    
    return text

def load_pdf(file_path):
    logger.info(f"📄 Loading PDF: {file_path}")
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        logger.info(f"📝 Extracted {len(text)} characters from {len(reader.pages)} pages")
        
        # Fix doubled characters if detected
        text = fix_doubled_text(text)
        
    except Exception as e:
        logger.error(f"❌ Error reading PDF {file_path}: {e}")
        return "" # Return empty string on failure

    return text
