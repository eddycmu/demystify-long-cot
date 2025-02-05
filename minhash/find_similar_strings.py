import os
import re
import time
from tqdm import tqdm
import argparse
import pandas as pd
import yaml
from multiprocessing import Pool, cpu_count
from datasketch import MinHash, MinHashLSHForest
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# --- Helper Functions ---

def sanitize_filename(s):
    """Sanitize a string to be used as a file/directory name."""
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')

def get_shingles(text, shingle_size=3):
    """
    Create a set of shingles (overlapping groups of words) from the input text.
    Lower-case the text and remove punctuation.
    """
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = text.split()
    if len(tokens) < shingle_size:
        return set(tokens)
    return set(' '.join(tokens[i:i+shingle_size]) for i in range(len(tokens) - shingle_size + 1))

def build_query_index(queries, num_perm):
    """
    For each query string, compute its MinHash signature and build an LSH forest.
    Returns a tuple of (query_index, query_signatures) where:
      - query_index: a MinHashLSHForest built from the queries.
      - query_signatures: a dict mapping query string to its MinHash object.
    """
    query_signatures = {}
    query_index = MinHashLSHForest(num_perm=num_perm)
    for query in queries:
        m = MinHash(num_perm=num_perm)
        shingles = get_shingles(query)
        for shingle in shingles:
            m.update(shingle.encode("utf8"))
        query_signatures[query] = m
        query_index.add(query, m)
    query_index.index()
    return query_index, query_signatures

def partition_list(lst, num_partitions):
    """
    Split a list into num_partitions roughly equal parts.
    """
    k, m = divmod(len(lst), num_partitions)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(num_partitions)]

def list_input_files(input_dir, min_file=None, max_file=None):
    """
    List all parquet files in input_dir whose filename (without extension)
    falls between min_file and max_file (lexicographically). Files are sorted
    by filename (without extension).
    """
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")]
    filtered_files = [
        f for f in all_files
        if (min_file is None or os.path.splitext(f)[0] >= min_file) and 
           (max_file is None or os.path.splitext(f)[0] <= max_file)
    ]
    filtered_files = sorted(filtered_files, key=lambda f: os.path.splitext(f)[0])
    file_paths = [os.path.join(input_dir, f) for f in filtered_files]
    return file_paths

def load_queries_from_yaml(yaml_path):
    """
    Load queries from a YAML file.
    Expected YAML format:
      queries:
        - "query string one"
        - "another query string"
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    queries = data.get("queries")
    if not queries:
        raise ValueError("No 'queries' key found in the YAML file or it is empty.")
    return queries

# --- Worker Function ---

def process_file_group(args):
    """
    Each worker process is assigned a list of file paths.
    For each file:
      - Load the parquet file.
      - For each passage, compute its MinHash signature and query the query index.
      - If any candidate query meets the similarity threshold, record the passage.
      - Write out the matched passages to an output file with the same naming convention.
    """
    (file_paths, passage_column, num_perm, query_index, query_signatures,
     similarity_threshold, output_dir) = args

    for file_path in file_paths:
        try:
            print(f"Process {os.getpid()} processing file {file_path}")
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if passage_column not in df.columns:
            print(f"Column {passage_column} not found in {file_path}. Skipping.")
            continue

        matched_rows = []
        # Process each passage in the file.
        # We'll split each paragraph into sentences and check each sentence separately.
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            passage = row[passage_column]
            sentences = sent_tokenize(passage)
            matched_queries = []
            for sentence in sentences:
                m_sentence = MinHash(num_perm=num_perm)
                shingles = get_shingles(sentence)
                for shingle in shingles:
                    m_sentence.update(shingle.encode("utf8"))
                # Query the query index using the sentence's minhash.
                candidates = query_index.query(m_sentence, len(query_signatures))
                for candidate in candidates:
                    sim = m_sentence.jaccard(query_signatures[candidate])
                    if sim >= similarity_threshold:
                        matched_queries.append(candidate)
            # If any sentence in the passage matched, record the passage.
            if matched_queries:
                # Remove duplicates.
                matched_queries = list(set(matched_queries))
                row_copy = row.copy()
                row_copy["matched_queries"] = ", ".join(matched_queries)
                matched_rows.append(row_copy)

        if matched_rows:
            matched_df = pd.DataFrame(matched_rows)
            base = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(output_dir, f"{base}_matched.parquet")
            matched_df.to_parquet(output_file, index=False)
            print(f"Process {os.getpid()} wrote {len(matched_df)} matched passages to {output_file}")
        else:
            print(f"Process {os.getpid()} found no matches in {file_path}")

# --- Main Processing ---

def main():
    parser = argparse.ArgumentParser(
        description="Process parquet files by partitioning files among processes. "
                    "Each process loads its group, checks passages against a query index built from YAML, "
                    "and outputs matches to a file with the same naming convention."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input parquet files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output matched parquet files.")
    parser.add_argument("--min_file", type=str, default=None,
                        help="Minimum filename (without extension, lexicographically) to process.")
    parser.add_argument("--max_file", type=str, default=None,
                        help="Maximum filename (without extension, lexicographically) to process.")
    parser.add_argument("--passage_column", type=str, default="passage",
                        help="Name of the column containing passages in the parquet files.")
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Number of permutations for MinHash.")
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Minimum Jaccard similarity (approximate) to consider a passage matching a query.")
    parser.add_argument("--queries_yaml", type=str, required=True,
                        help="YAML file containing query strings.")
    args = parser.parse_args()

    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load queries from the YAML file.
    print("Loading queries from YAML...")
    queries = load_queries_from_yaml(args.queries_yaml)
    print(f"Loaded {len(queries)} queries.")

    # Build the query index and signatures.
    query_index, query_signatures = build_query_index(queries, args.num_perm)
    print("Query index built.")

    # List and partition the input files.
    file_paths = list_input_files(args.input_dir, args.min_file, args.max_file)
    if not file_paths:
        raise ValueError("No input files found matching the criteria.")
    print(f"Found {len(file_paths)} input files.")
    num_workers = min(cpu_count(), len(file_paths))
    partitions = partition_list(file_paths, num_workers)
    print(f"Partitioned files into {num_workers} groups.")

    # Prepare arguments for each worker.
    worker_args = []
    for part in partitions:
        worker_args.append((
            part,
            args.passage_column,
            args.num_perm,
            query_index,
            query_signatures,
            args.similarity_threshold,
            args.output_dir
        ))

    # Launch the workers.
    start_time = time.time()
    with Pool(num_workers) as pool:
        pool.map(process_file_group, worker_args)
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
