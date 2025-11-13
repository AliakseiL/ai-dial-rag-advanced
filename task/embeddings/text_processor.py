from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self, table_name: str = "vectors"):
        """Truncate the provided table name safely using SQL identifiers.

        This method centralizes table cleanup logic so callers don't need to
        open/close connections repeatedly.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Use Identifier to avoid SQL injection on table name
                cursor.execute(sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(table_name)))
            conn.commit()

    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)
    def process_text_file(self, file_name: str, chunk_size: int, overlap: int, dimensions: int, truncate_table: bool):
        if truncate_table:
            # Use the helper to truncate the vectors table
            self._truncate_table("vectors")

        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = chunk_text(content, chunk_size, overlap)
        embeddings_dict = self.embeddings_client.get_embeddings(chunks, dimensions)

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for index, chunk in enumerate(chunks):
                    embedding_vector = embeddings_dict[index]
                    embedding_str = "[" + ",".join(map(str, embedding_vector)) + "]"
                    cursor.execute(
                        "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector);",
                        (file_name, chunk, embedding_str)
                    )
            conn.commit()

    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`
    def search(self, search_mode: SearchMode, user_request: str, top_k: int, score_threshold: float, dimensions: int) -> list[str]:
        request_embedding_dict = self.embeddings_client.get_embeddings(user_request, dimensions)
        request_embedding_vector = request_embedding_dict[0]
        request_embedding_str = "[" + ",".join(map(str, request_embedding_vector)) + "]"

        distance_operator = "<->" if search_mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"

        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        query = f"""
            SELECT text, embedding {distance_operator} %s::vector AS distance
            FROM vectors
            WHERE embedding {distance_operator} %s::vector < %s
            ORDER BY distance
            LIMIT %s;
        """

        results = []
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (request_embedding_str, request_embedding_str, max_distance, top_k))
                results = cursor.fetchall()

        retrieved_chunks = []
        for row in results:
            if search_mode == SearchMode.COSINE_DISTANCE:
                similarity = 1.0 - row['distance']
            else:
                similarity = 1.0 / (1.0 + row['distance'])

            print(f"---Similarity score: {similarity:.2f}---")
            print(f"Data: {row['text']}\n")
            retrieved_chunks.append(row['text'])

        return retrieved_chunks
