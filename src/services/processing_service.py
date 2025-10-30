"""
Processing Service

Wraps document processing functionality with better interface and error handling.
Provides document ingestion, CSV processing, and file operations.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import csv
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from src.utils.directory import directory_enumerator
from src.utils.hash_utils import hash_text


class ProcessingService:
    """
    Service for managing document processing operations.

    This service wraps document ingestion functionality and provides
    a cleaner interface with better error handling and logging.
    """

    def __init__(self, log_function=None):
        """
        Initialize processing service.

        Args:
            log_function: Optional logging function
        """
        self.log = log_function or print

        # Configuration for memory-efficient processing
        self.max_text_length = 15000  # Maximum characters per document (even more conservative)
        self.batch_size = 3  # Process files in smaller batches to avoid memory issues
    
    def process_source(self, path: str, limit: Optional[int] = None, text_column: Optional[str] = None, url_column: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process data from various sources (directory, CSV, URL, single file).
        
        Args:
            path: Source path (directory, CSV file, URL, or single file)
            limit: Maximum number of items to process
            text_column: Column name for text content in CSV
            url_column: Column name for URLs in CSV
            
        Returns:
            Tuple of (texts, payloads)
        """
        try:
            self.log(f"Processing source: {path}")
            
            # Determine source type and process accordingly
            if os.path.isdir(path):
                return self._process_directory(path, limit)
            elif os.path.isfile(path) and path.endswith(".csv"):
                return self._process_csv(path, limit, text_column, url_column)
            elif path.startswith(("http://", "https://")):
                return self._process_url(path, limit)
            elif os.path.isfile(path):
                return self._process_single_file(path)
            else:
                self.log(f"Unsupported source path: {path}", True)
                return [], []
                
        except Exception as e:
            self.log(f"Failed to process source '{path}': {e}", True)
            return [], []
    
    def _process_directory(self, directory_path: str, limit: Optional[int] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process files in a directory with memory-efficient batching."""
        try:
            texts = []
            payloads = []

            # Get all files from directory with document extensions
            extensions = ['.txt', '.md', '.docx', '.pdf', '.html']
            all_files = directory_enumerator(directory_path, extensions, limit)

            # Apply limit if specified
            if limit:
                all_files = all_files[:limit]

            total_files = len(all_files)
            self.log(f"Found {total_files} files in directory")

            # Process files in batches to avoid memory issues
            for batch_start in range(0, total_files, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_files)
                batch_files = all_files[batch_start:batch_end]

                self.log(f"Processing batch {batch_start//self.batch_size + 1}/{(total_files + self.batch_size - 1)//self.batch_size} ({len(batch_files)} files)")

                for file_path in batch_files:
                    try:
                        # Read file content with proper encoding handling
                        content = self.extract_text_from_file(file_path)

                        if content:
                            # Truncate very long content to prevent memory issues and API timeouts
                            original_length = len(content)
                            if original_length > self.max_text_length:
                                # Use tiktoken for better token estimation if available
                                try:
                                    from tiktoken import encoding_for_model
                                    enc = encoding_for_model("text-embedding-3-small")
                                    tokens = enc.encode(content)

                                    # If token count is very high, truncate more aggressively
                                    if len(tokens) > 6000:  # Lower threshold for safety
                                        # Find character position for ~4000 tokens (very conservative)
                                        target_tokens = 4000
                                        truncated_content = ""

                                        for i in range(0, len(content), 500):  # Check every 500 chars
                                            test_content = content[:i+500]
                                            test_tokens = len(enc.encode(test_content))
                                            if test_tokens >= target_tokens:
                                                truncated_content = test_content
                                                break
                                        else:
                                            truncated_content = content[:self.max_text_length]

                                        content = truncated_content
                                        self.log(f"Token-aware truncated file {file_path} from {original_length} chars ({len(tokens)} tokens) to {len(content)} chars (~{target_tokens} tokens)")
                                    else:
                                        content = content[:self.max_text_length]
                                        self.log(f"Truncated file {file_path} from {original_length} to {len(content)} characters")

                                except ImportError:
                                    # Fallback to simple character truncation
                                    content = content[:self.max_text_length]
                                    self.log(f"Truncated file {file_path} from {original_length} to {len(content)} characters")

                            texts.append(content)
                            payloads.append({
                                "source": file_path,
                                "hash": hash_text(content),
                                "type": "file",
                                "text_content": content[:800],  # Store first 800 chars for better cluster naming
                                "original_length": original_length if original_length > len(content) else None
                            })

                    except Exception as e:
                        self.log(f"Failed to read file {file_path}: {e}", True)
                        continue

            self.log(f"Successfully processed {len(texts)} files from directory")
            return texts, payloads

        except Exception as e:
            self.log(f"Failed to process directory '{directory_path}': {e}", True)
            return [], []
    
    def _process_csv(self, csv_path: str, limit: Optional[int] = None, text_column: Optional[str] = None, url_column: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process CSV file with enhanced functionality."""
        try:
            texts = []
            payloads = []
            
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return texts, payloads
            
            # Apply limit if specified
            if limit:
                rows = rows[:limit]
            
            # Check if we have URL column specified
            if url_column and url_column in rows[0]:
                # Extract content from URLs (HTML or PDF)
                for idx, row in enumerate(rows):
                    url = row[url_column].strip()
                    if not url:
                        continue
                    
                    # Detect content type with HEAD (best-effort)
                    content_type = None
                    try:
                        head_resp = requests.head(url, timeout=10, allow_redirects=True)
                        content_type = head_resp.headers.get('Content-Type', '')
                    except Exception:
                        # Fallback to extension check
                        content_type = ''
                    
                    text_content = ""
                    is_pdf = ('.pdf' in url.lower()) or ('application/pdf' in (content_type or '').lower())
                    if is_pdf:
                        text_content = self._extract_pdf_content(url)
                        doc_type = "url_pdf"
                    else:
                        text_content = self._extract_html_content(url)
                        doc_type = "url_html"
                    
                    if not text_content:
                        continue
                    
                    # Create payload with all CSV fields
                    payload = {
                        "source": url,
                        "hash": hash_text(text_content),
                        "type": doc_type,
                        "url": url
                    }
                    
                    # Add other CSV fields to payload
                    for key, value in row.items():
                        if key != url_column:
                            payload[key] = value
                    
                    texts.append(text_content)
                    payloads.append(payload)
                return texts, payloads
            
            # Determine text column
            if text_column:
                text_col = text_column
            else:
                # Try to find text column automatically
                possible_text_cols = ["text", "Text", "content", "Content", "body", "Body"]
                text_col = None
                for col in possible_text_cols:
                    if col in rows[0]:
                        text_col = col
                        break
                
                if not text_col:
                    self.log("No text column found in CSV. Please specify --text-column", True)
                    return [], []
            
            # Process each row
            for idx, row in enumerate(rows):
                try:
                    text = row.get(text_col, "").strip()
                    if not text:
                        continue
                    
                    texts.append(text)
                    
                    # Create payload with metadata
                    payload = {
                        "source": f"{csv_path}-{idx}",
                        "hash": hash_text(text),
                        "type": "csv_text",
                        "text_content": text[:500]  # Store first 500 chars for cluster naming
                    }
                    
                    # Add other columns as metadata (exclude text column)
                    for col_name, col_value in row.items():
                        if col_name != text_col:
                            payload[col_name] = col_value
                    
                    payloads.append(payload)
                    
                except Exception as e:
                    self.log(f"Failed to process CSV row {idx}: {e}", True)
                    continue
            
            self.log(f"Processed {len(texts)} rows from CSV")
            return texts, payloads
            
        except Exception as e:
            self.log(f"Failed to process CSV '{csv_path}': {e}", True)
            return [], []
    
    def _extract_html_content(self, url: str) -> str:
        """Extract text content from HTML URL using BeautifulSoup."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            self.log(f"Failed to extract content from {url}: {e}", True)
            return ""

    def _extract_pdf_content(self, url: str) -> str:
        """Extract text content from a PDF URL using streaming partial download.

        Best-effort: downloads up to a maximum byte budget and attempts to parse
        the first few pages. If parsing fails, returns empty string.
        """
        try:
            MAX_PDF_BYTES = 8 * 1024 * 1024  # 8 MB cap
            MAX_PAGES = 5  # Extract only first few pages to limit tokens
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/pdf',
                'Range': f'bytes=0-{MAX_PDF_BYTES - 1}'
            }
            resp = requests.get(url, headers=headers, timeout=20, stream=True)
            resp.raise_for_status()
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) >= MAX_PDF_BYTES:
                    break
            if not buf:
                return ""
            # Parse with pypdf
            try:
                from pypdf import PdfReader
                import io
                reader = PdfReader(io.BytesIO(bytes(buf)))
                pages_to_read = min(MAX_PAGES, len(reader.pages))
                texts = []
                for i in range(pages_to_read):
                    try:
                        texts.append(reader.pages[i].extract_text() or "")
                    except Exception:
                        continue
                content = "\n".join([t for t in texts if t])
                return content
            except Exception as parse_err:
                self.log(f"PDF parse failed for {url}: {parse_err}", True)
                return ""
        except Exception as e:
            self.log(f"Failed to fetch PDF from {url}: {e}", True)
            return ""
    
    def _process_url(self, url: str, limit: Optional[int] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process URL content with improved HTML/PDF extraction."""
        try:
            texts = []
            payloads = []
            
            # Detect content type (best-effort)
            content_type = None
            try:
                head_resp = requests.head(url, timeout=10, allow_redirects=True)
                content_type = head_resp.headers.get('Content-Type', '')
            except Exception:
                content_type = ''
            is_pdf = ('.pdf' in url.lower()) or ('application/pdf' in (content_type or '').lower())

            # Extract content
            if is_pdf:
                text_content = self._extract_pdf_content(url)
                doc_type = "url_pdf"
            else:
                text_content = self._extract_html_content(url)
                doc_type = "url"
            
            if text_content:
                texts.append(text_content)
                payloads.append({
                    "source": url,
                    "hash": hash_text(text_content),
                    "type": doc_type,
                    "url": url
                })
            
            self.log(f"Processed URL: {url}")
            return texts, payloads
            
        except Exception as e:
            self.log(f"Failed to process URL '{url}': {e}", True)
            return [], []
    
    def _process_single_file(self, file_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a single file."""
        try:
            texts = []
            payloads = []
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if content:
                texts.append(content)
                payloads.append({
                    "source": file_path,
                    "hash": hash_text(content),
                    "type": "file"
                })
            
            self.log(f"Processed single file: {file_path}")
            return texts, payloads
            
        except Exception as e:
            self.log(f"Failed to process file '{file_path}': {e}", True)
            return [], []
    
    def validate_source(self, path: str) -> Dict[str, Any]:
        """
        Validate a source path and return information about it.
        
        Args:
            path: Source path to validate
            
        Returns:
            Dictionary with validation results and source information
        """
        try:
            result = {
                "path": path,
                "valid": False,
                "type": "unknown",
                "exists": False,
                "readable": False,
                "info": {}
            }
            
            if path.startswith(("http://", "https://")):
                result["type"] = "url"
                result["valid"] = True
                result["info"] = {"protocol": "http" if path.startswith("http://") else "https"}
                
            elif os.path.exists(path):
                result["exists"] = True
                
                if os.path.isfile(path):
                    result["type"] = "file"
                    result["readable"] = os.access(path, os.R_OK)
                    
                    if path.endswith(".csv"):
                        result["type"] = "csv"
                        result["info"] = {
                            "extension": "csv",
                            "size_bytes": os.path.getsize(path)
                        }
                    else:
                        result["info"] = {
                            "extension": Path(path).suffix,
                            "size_bytes": os.path.getsize(path)
                        }
                    
                    result["valid"] = result["readable"]
                    
                elif os.path.isdir(path):
                    result["type"] = "directory"
                    result["readable"] = os.access(path, os.R_OK)
                    
                    # Count files in directory
                    try:
                        files = directory_enumerator(path)
                        result["info"] = {
                            "file_count": len(files),
                            "files": files[:10]  # First 10 files as sample
                        }
                        result["valid"] = result["readable"] and len(files) > 0
                    except:
                        result["valid"] = False
                        
            return result
            
        except Exception as e:
            return {
                "path": path,
                "valid": False,
                "type": "unknown",
                "exists": False,
                "readable": False,
                "error": str(e)
            }
    
    def get_source_stats(self, path: str) -> Dict[str, Any]:
        """
        Get statistics about a source.
        
        Args:
            path: Source path
            
        Returns:
            Dictionary with source statistics
        """
        try:
            validation = self.validate_source(path)
            
            if not validation["valid"]:
                return {"error": "Invalid source", "validation": validation}
            
            # Get basic stats
            stats = {
                "path": path,
                "type": validation["type"],
                "exists": validation["exists"],
                "readable": validation["readable"]
            }
            
            if validation["type"] == "file":
                stats.update({
                    "size_bytes": os.path.getsize(path),
                    "extension": Path(path).suffix
                })
                
                if path.endswith(".csv"):
                    # Get CSV-specific stats
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        
                        stats.update({
                            "csv_rows": len(rows),
                            "csv_columns": list(rows[0].keys()) if rows else [],
                            "csv_sample": rows[:3] if rows else []
                        })
                    except Exception as e:
                        stats["csv_error"] = str(e)
                        
            elif validation["type"] == "directory":
                try:
                    files = directory_enumerator(path)
                    stats.update({
                        "file_count": len(files),
                        "files": files[:20]  # First 20 files
                    })
                except Exception as e:
                    stats["directory_error"] = str(e)
                    
            elif validation["type"] == "url":
                stats.update({
                    "protocol": validation["info"].get("protocol", "unknown")
                })
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text content from a file with robust encoding handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        try:
            # Try UTF-8 first
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 for problematic files
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read().strip()
            except Exception as e:
                self.log(f"Failed to extract text from '{file_path}' with fallback encoding: {e}", True)
                return ""
        except Exception as e:
            self.log(f"Failed to extract text from '{file_path}': {e}", True)
            return ""
    
    def save_processed_data(self, texts: List[str], payloads: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Save processed data to a file.
        
        Args:
            texts: List of texts
            payloads: List of payloads
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "texts": texts,
                "payloads": payloads,
                "count": len(texts),
                "timestamp": str(Path().cwd())
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.log(f"Saved processed data to: {output_path}")
            return True
            
        except Exception as e:
            self.log(f"Failed to save processed data: {e}", True)
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the processing service.

        Returns:
            Dictionary with service information
        """
        return {
            "service_name": "ProcessingService",
            "supported_types": ["directory", "csv", "url", "file"],
            "features": [
                "directory_enumeration",
                "csv_processing",
                "url_content_extraction",
                "file_processing",
                "source_validation",
                "statistics_generation",
                "memory_efficient_batching",
                "content_truncation"
            ],
            "configuration": {
                "max_text_length": self.max_text_length,
                "batch_size": self.batch_size
            }
        }

    def set_max_text_length(self, max_length: int):
        """Set the maximum text length for truncation."""
        self.max_text_length = max_length
        self.log(f"Set max text length to {max_length} characters")

    def set_batch_size(self, batch_size: int):
        """Set the batch size for processing files."""
        self.batch_size = max(1, batch_size)
        self.log(f"Set batch size to {self.batch_size} files")
    
    def close(self):
        """Close the processing service."""
        try:
            self.log("Processing service closed")
        except Exception as e:
            self.log(f"Error closing processing service: {e}", True)
