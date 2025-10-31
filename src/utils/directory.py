import os
import sys

def directory_enumerator(dir_path: str, extensions: list[str], limit:int=None) -> list[str]:
    """
    Recursively enumerate files in directory matching given extensions.
    
    Args:
        dir_path: Root directory to search
        extensions: List of file extensions to match (e.g., ['.txt', '.md'])
        limit: Maximum number of files to return (None for unlimited)
    
    Returns:
        List of full paths to matching files
    """
    matching_files = []
    total = limit if limit is not None else "unlimited"

    # Use os.walk() to recursively traverse directory tree
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Skip temporary files (Office temp files, system files, etc.)
            if file.startswith('~$') or file.startswith('.~') or file.startswith('.'):
                continue
            
            if any(file.endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                matching_files.append(full_path)

                count = len(matching_files)
                try:
                    sys.stdout.write(f"\rFound: {count}/{total} -> {full_path}")
                    sys.stdout.flush()
                except UnicodeEncodeError:
                    # Handle Unicode characters in file paths
                    safe_path = full_path.encode('ascii', 'replace').decode('ascii')
                    sys.stdout.write(f"\rFound: {count}/{total} -> {safe_path}")
                    sys.stdout.flush()

                if limit is not None and count >= limit:
                    break
        
        # Early exit if limit reached
        if limit is not None and len(matching_files) >= limit:
            break

    print()
    return matching_files
