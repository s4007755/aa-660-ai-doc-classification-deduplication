import sqlite3
import struct
from typing import List, Tuple, Optional, Any


class VecDB:
    """
    Vector Database class for document classification pipeline.
    Uses SQLite with sqlite-vec extension for vector similarity search.
    """

    def __init__(self, table_name: str, create: bool = False, db_path: str = ':memory:'):
        """
        Initialize VecDB instance tied to a specific table.

        Args:
            table_name: Name of the table to perform CRUD operations on
            create: If True, create the table if it doesn't exist
            db_path: Path to the database file (default: ':memory:' for in-memory DB)
        """
        self.table_name = table_name
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._vec_extension_loaded = False

        # Load sqlite-vec extension
        self._load_vec_extension()

        if create:
            success = self._create_table()
            if not success:
                raise RuntimeError(f"Failed to create table {table_name}")

    def _load_vec_extension(self) -> bool:
        """
        Load the sqlite-vec extension for vector similarity operations.

        Returns:
            bool: True if extension loaded successfully, False otherwise
        """
        try:
            self.conn.enable_load_extension(True)
            # Try common paths for the vec0 extension
            extension_paths = ['vec0', './vec0', 'vec0.so', './vec0.so']

            for path in extension_paths:
                try:
                    self.conn.load_extension(path)
                    self._vec_extension_loaded = True
                    return True
                except sqlite3.OperationalError:
                    continue

            print("Vec extension not found")
            return False
        except Exception as e:
            print(f"Extension load error: {e}")
            return False

    def _create_table(self) -> bool:
        """
        Create table with schema: docID, document_text, token_size, vector, 
        distance_to_centroid, class

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create main table
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    docID TEXT PRIMARY KEY,
                    document_text TEXT,
                    token_size INTEGER,
                    vector BLOB,
                    distance_to_centroid REAL,
                    class TEXT
                )
            ''')

            # Create index for better query performance
            self.cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_class 
                ON {self.table_name}(class)
            ''')

            self.cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_distance 
                ON {self.table_name}(distance_to_centroid)
            ''')

            # Create virtual table for vector similarity search if extension is loaded
            if self._vec_extension_loaded:
                self.cursor.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}_vec 
                    USING vec0(
                        docID TEXT PRIMARY KEY,
                        embedding FLOAT[768]
                    )
                ''')

            self.conn.commit()
            return True
        except Exception as e:
            print(f"Table creation failed: {e}")
            self.conn.rollback()
            return False

    def _serialize_vector(self, vector: List[float]) -> bytes:
        """
        Convert list of floats to binary format for storage.

        Args:
            vector: List of float values

        Returns:
            bytes: Serialized vector
        """
        if not vector:
            return b''
        return struct.pack(f'{len(vector)}f', *vector)

    def _deserialize_vector(self, blob: bytes) -> List[float]:
        """
        Convert binary format back to list of floats.

        Args:
            blob: Binary data

        Returns:
            List[float]: Deserialized vector
        """
        if not blob:
            return []
        num_floats = len(blob) // 4
        return list(struct.unpack(f'{num_floats}f', blob))

    def insert_documents(self, documents: List[Tuple]) -> bool:
        """
        Insert documents into the database.

        Args:
            documents: List of tuples in one of two formats:
                - 3 elements: (docID, document_text, token_size)
                - 4 elements: (docID, document_text, token_size, vector)

        Returns:
            bool: True if all inserts successful, False otherwise
        """
        if not documents:
            return True

        try:
            for doc in documents:
                if len(doc) == 3:
                    # Insert without vector
                    docID, document_text, token_size = doc
                    self.cursor.execute(f'''
                        INSERT OR REPLACE INTO {self.table_name} 
                        (docID, document_text, token_size, vector, distance_to_centroid, class)
                        VALUES (?, ?, ?, NULL, NULL, NULL)
                    ''', (docID, document_text, token_size))

                elif len(doc) == 4:
                    # Insert with vector
                    docID, document_text, token_size, vector = doc

                    # Validate vector dimensions
                    if vector and len(vector) != 768:
                        print(f"Invalid vector size: {len(vector)}")

                    vector_blob = self._serialize_vector(
                        vector) if vector else None

                    self.cursor.execute(f'''
                        INSERT OR REPLACE INTO {self.table_name} 
                        (docID, document_text, token_size, vector, distance_to_centroid, class)
                        VALUES (?, ?, ?, ?, NULL, NULL)
                    ''', (docID, document_text, token_size, vector_blob))

                    # Also insert into vector search table if available
                    if self._vec_extension_loaded and vector:
                        self.cursor.execute(f'''
                            INSERT OR REPLACE INTO {self.table_name}_vec 
                            (docID, embedding)
                            VALUES (?, ?)
                        ''', (docID, vector_blob))
                else:
                    print(f"Invalid tuple size: {len(doc)}")
                    return False

            self.conn.commit()
            return True
        except Exception as e:
            print(f"Insert failed: {e}")
            self.conn.rollback()
            return False

    def set_column_values(self, column_name: str, updates: List[Tuple[str, Any]]) -> bool:
        """
        Set values for a specific column using executemany for efficiency.

        Args:
            column_name: Name of the column to update
            updates: List of tuples with (docID, value)

        Returns:
            bool: True if successful, False otherwise
        """
        if not updates:
            return True

        # Validate column name
        valid_columns = {'document_text', 'token_size', 'vector',
                         'distance_to_centroid', 'class'}
        if column_name not in valid_columns:
            print(f"Invalid column: {column_name}")
            return False

        try:
            # Handle vector column specially
            if column_name == 'vector':
                processed_updates = []
                for docID, value in updates:
                    if isinstance(value, list):
                        vector_blob = self._serialize_vector(value)
                        processed_updates.append((vector_blob, docID))

                        # Update vector search table if available
                        if self._vec_extension_loaded:
                            self.cursor.execute(f'''
                                INSERT OR REPLACE INTO {self.table_name}_vec 
                                (docID, embedding)
                                VALUES (?, ?)
                            ''', (docID, vector_blob))
                    else:
                        processed_updates.append((value, docID))
            else:
                processed_updates = [(value, docID)
                                     for docID, value in updates]

            self.cursor.executemany(f'''
                UPDATE {self.table_name}
                SET {column_name} = ?
                WHERE docID = ?
            ''', processed_updates)

            self.conn.commit()
            return True
        except Exception as e:
            print(f"Update failed: {e}")
            self.conn.rollback()
            return False

    def execute_sql(self, sql: str, params: Optional[Tuple] = None) -> bool:
        """
        Execute arbitrary SQL statements for testing or future additions.

        Args:
            sql: SQL statement to execute
            params: Optional parameters for the SQL statement

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)

            # Only commit for non-SELECT statements
            if not sql.strip().upper().startswith('SELECT'):
                self.conn.commit()

            return True
        except Exception as e:
            print(f"SQL execution failed: {e}")
            self.conn.rollback()
            return False

    def query_top_n_closest(self, centroid: List[float], n: int) -> List[Tuple]:
        """
        Query database for top n documents closest to centroid, ranked ascending.

        Args:
            centroid: Vector representing the centroid
            n: Number of results to return

        Returns:
            List of tuples with document information, ranked by distance ascending
        """
        try:
            if self._vec_extension_loaded and centroid:
                # Use vector similarity search
                centroid_blob = self._serialize_vector(centroid)
                self.cursor.execute(f'''
                    SELECT m.docID, m.document_text, m.token_size, m.vector, 
                           m.distance_to_centroid, m.class,
                           vec_distance_L2(v.embedding, ?) as distance
                    FROM {self.table_name} m
                    JOIN {self.table_name}_vec v ON m.docID = v.docID
                    WHERE m.vector IS NOT NULL
                    ORDER BY distance ASC
                    LIMIT ?
                ''', (centroid_blob, n))
            else:
                # Fallback: Use pre-computed distance_to_centroid if available
                self.cursor.execute(f'''
                    SELECT docID, document_text, token_size, vector, 
                           distance_to_centroid, class
                    FROM {self.table_name}
                    WHERE distance_to_centroid IS NOT NULL
                    ORDER BY distance_to_centroid ASC
                    LIMIT ?
                ''', (n,))

            results = self.cursor.fetchall()

            # Deserialize vectors in results
            processed_results = []
            for row in results:
                row_list = list(row)
                vector_idx = 3  # vector is at index 3
                if len(row_list) > vector_idx and row_list[vector_idx]:
                    row_list[vector_idx] = self._deserialize_vector(
                        row_list[vector_idx])
                processed_results.append(tuple(row_list))

            return processed_results
        except Exception as e:
            print(f"Query failed: {e}")
            return []

    def get_all_columns(self, columns: Optional[List[str]] = None) -> List[Tuple]:
        """
        Get all specified columns from db, return list of tuples.

        Args:
            columns: List of column names to retrieve. If None, retrieves all columns.

        Returns:
            List of tuples containing the requested data
        """
        try:
            if columns:
                # Validate column names
                valid_columns = {'docID', 'document_text', 'token_size',
                                 'vector', 'distance_to_centroid', 'class'}
                invalid = set(columns) - valid_columns
                if invalid:
                    print(f"Invalid columns: {invalid}")
                    columns = [c for c in columns if c in valid_columns]

                if not columns:
                    return []

                columns_str = ', '.join(columns)
            else:
                columns_str = 'docID, document_text, token_size, vector, distance_to_centroid, class'

            self.cursor.execute(f'SELECT {columns_str} FROM {self.table_name}')
            results = self.cursor.fetchall()

            # Deserialize vectors if vector column is included
            if not columns or 'vector' in columns:
                vector_idx = columns.index('vector') if columns else 3
                processed_results = []
                for row in results:
                    row_list = list(row)
                    if len(row_list) > vector_idx and row_list[vector_idx]:
                        row_list[vector_idx] = self._deserialize_vector(
                            row_list[vector_idx])
                    processed_results.append(tuple(row_list))
                return processed_results

            return results
        except Exception as e:
            print(f"Retrieval failed: {e}")
            return []

    def create_labels_table(self, labels_table_name: str) -> bool:
        """
        Create a separate table for label vectors.

        Args:
            labels_table_name: Name for the labels table

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create main labels table
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {labels_table_name} (
                    label TEXT PRIMARY KEY,
                    vector BLOB NOT NULL
                )
            ''')

            # Create virtual table for label vector similarity search if extension loaded
            if self._vec_extension_loaded:
                self.cursor.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS {labels_table_name}_vec 
                    USING vec0(
                        label TEXT PRIMARY KEY,
                        embedding FLOAT[768]
                    )
                ''')

            self.conn.commit()
            return True
        except Exception as e:
            print(f"Labels table creation failed: {e}")
            self.conn.rollback()
            return False

    def insert_labels(self, labels_table_name: str, labels: List[Tuple[str, List[float]]]) -> bool:
        """
        Insert label vectors into the labels table for processing directly in SQLite.

        Args:
            labels_table_name: Name of the labels table
            labels: List of tuples with (label, vector)

        Returns:
            bool: True if successful, False otherwise
        """
        if not labels:
            return True

        try:
            for label, vector in labels:
                if not vector:
                    print(f"Empty vector: {label}")
                    continue

                vector_blob = self._serialize_vector(vector)

                self.cursor.execute(f'''
                    INSERT OR REPLACE INTO {labels_table_name} 
                    (label, vector)
                    VALUES (?, ?)
                ''', (label, vector_blob))

                # Also insert into vector search table if available
                if self._vec_extension_loaded:
                    self.cursor.execute(f'''
                        INSERT OR REPLACE INTO {labels_table_name}_vec 
                        (label, embedding)
                        VALUES (?, ?)
                    ''', (label, vector_blob))

            self.conn.commit()
            return True
        except Exception as e:
            print(f"Label insert failed: {e}")
            self.conn.rollback()
            return False

    def get_labels(self, labels_table_name: str) -> List[Tuple[str, List[float]]]:
        """
        Retrieve all labels and their vectors for processing.

        Args:
            labels_table_name: Name of the labels table

        Returns:
            List of tuples with (label, vector)
        """
        try:
            self.cursor.execute(f'''
                SELECT label, vector
                FROM {labels_table_name}
                ORDER BY label
            ''')

            results = self.cursor.fetchall()
            return [(label, self._deserialize_vector(vector))
                    for label, vector in results if vector]
        except Exception as e:
            print(f"Label retrieval failed: {e}")
            return []

    def close(self) -> bool:
        """
        Close the database connection.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.conn.close()
            return True
        except Exception as e:
            print(f"Close failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Create database instance
    db = VecDB("documents", create=True)

    print("Testing VecDB implementation")

    # Test 1: Insert documents without vectors (3-element tuples)
    docs_3 = [
        ("doc1", "This is a sample document about machine learning", 8),
        ("doc2", "Another document discussing deep learning", 5),
    ]
    success = db.insert_documents(docs_3)
    print(f"Inserted 3-element tuples: {success}")

    # Test 2: Insert documents with vectors (4-element tuples)
    import random
    vector1 = [random.random() for _ in range(768)]
    vector2 = [random.random() for _ in range(768)]
    docs_4 = [
        ("doc3", "Document with vector about NLP", 5, vector1),
        ("doc4", "Another with vector about computer vision", 6, vector2),
    ]
    success = db.insert_documents(docs_4)
    print(f"Inserted 4-element tuples: {success}")

    # Test 3: Set column values
    success = db.set_column_values("class", [
        ("doc1", "machine_learning"),
        ("doc2", "deep_learning"),
        ("doc3", "nlp"),
        ("doc4", "computer_vision")
    ])
    print(f"Set class values: {success}")

    # Set distance values
    success = db.set_column_values("distance_to_centroid", [
        ("doc1", 0.5),
        ("doc2", 0.3),
        ("doc3", 0.7),
        ("doc4", 0.2)
    ])
    print(f"Set distance values: {success}")

    # Test 4: Query top n closest documents
    centroid = [random.random() for _ in range(768)]
    top_docs = db.query_top_n_closest(centroid, 2)
    print(f"Found {len(top_docs)} closest documents")

    # Test 5: Get specific columns
    results = db.get_all_columns(["docID", "class"])
    print(f"Retrieved {len(results)} documents")

    # Test 6: Get all columns
    all_docs = db.get_all_columns()
    print(f"Total documents: {len(all_docs)}")

    # Test 7: Create and use labels table
    success = db.create_labels_table("labels")
    print(f"Created labels table: {success}")

    label_vectors = [
        ("machine_learning", [random.random() for _ in range(768)]),
        ("deep_learning", [random.random() for _ in range(768)]),
        ("nlp", [random.random() for _ in range(768)]),
    ]
    success = db.insert_labels("labels", label_vectors)
    print(f"Inserted labels: {success}")

    labels = db.get_labels("labels")
    print(f"Retrieved {len(labels)} labels")

    # Test 8: Execute custom SQL
    success = db.execute_sql(
        f"SELECT COUNT(*) FROM {db.table_name} WHERE class IS NOT NULL"
    )
    if success:
        count = db.cursor.fetchone()[0]
        print(f"Documents with class: {count}")

    # Clean up
    success = db.close()
    print(f"Connection closed: {success}")

    print("Tests completed")
