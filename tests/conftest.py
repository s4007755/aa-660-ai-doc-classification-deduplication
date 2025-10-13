"""
Pytest configuration for automatic test cleanup.
"""

import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """
    Clean up test artifacts at the end of the test session.
    This runs automatically after all tests complete.
    """
    yield  # Let all tests run first
    
    # Cleanup after all tests are done
    files_to_remove = [
        "test_coll_clusters.json",
        "test_collection_clusters.json"
    ]
    
    removed_files = []
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_files.append(file_path)
        except OSError:
            pass
    
    if removed_files:
        print(f"\nSession cleanup: Removed {len(removed_files)} test artifacts")
