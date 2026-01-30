import pandas as pd
import re
import os
import ast
import numpy as np
from typing import Any
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from apted import APTED
from apted.helpers import Tree
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

# Class for CodeBERT and AST Edit Distance


class CodeSimilarityCalculator:
    def __init__(self):
        # Initialize CodeBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.model.eval()

        # Initialize tree-sitter for C++ parsing
        self.cpp_language = Language(tscpp.language())
        self.parser = Parser(self.cpp_language)

    def clean_code(self, code_str: str) -> str:
        """Clean code string by removing language markers and extra whitespace"""
        # Remove language markers like 'cpp\n' at the beginning
        code_str = re.sub(
            r"^(cpp|python|java|c\+\+)\s*\n", "", code_str, flags=re.IGNORECASE
        )
        # Remove extra whitespace
        return code_str.strip()

    def code_to_ast_tree(self, code: str) -> Tree:
        """Convert code to AST tree format for edit distance calculation"""
        try:
            # Clean the code
            clean_code = self.clean_code(code)
            # Detect language and parse accordingly
            if self._is_cpp_code(clean_code):
                return self._cpp_to_tree(clean_code)
            else:
                try:
                    tree = ast.parse(clean_code)
                    return self._python_ast_to_tree(tree)
                except:
                    # If both fail, use a simple token-based tree
                    return self._tokenize_to_tree(clean_code)
        except Exception as e:
            print(f"Error parsing code: {e}")
            # Fallback: create a simple tree based on tokens
            return self._tokenize_to_tree(code)

    def _is_cpp_code(self, code: str) -> bool:
        """Detect if code is C++"""
        cpp_indicators = [
            "#include",
            "std::",
            "::",
            "cout",
            "cin",
            "endl",
            "class",
            "public:",
            "private:",
            "protected:",
            "int main(",
            "namespace",
            "template<",
            "vector<",
        ]
        return any(indicator in code for indicator in cpp_indicators)

    def _cpp_to_tree(self, code: str) -> Tree:
        """Convert C++ code to tree using tree-sitter"""
        try:
            # Parse with tree-sitter
            tree = self.parser.parse(bytes(code, "utf8"))
            return self._simple_treesitter_to_apted(tree.root_node, depth=0)
        except Exception as e:
            print(f"Tree-sitter parsing failed: {e}")
            return self._tokenize_to_tree(code)

    def _simple_treesitter_to_apted(self, node, depth=0, max_depth=3) -> Tree:
        """Convert tree-sitter node to apted Tree with depth limiting"""
        try:
            # Limit recursion depth to prevent issues
            if depth > max_depth:
                return Tree.from_text("{leaf}")
            # Get node type safely
            node_type = str(getattr(node, "type", "unknown"))
            # Clean node type to avoid parsing issues
            node_type = re.sub(r"[^a-zA-Z0-9_]", "_", node_type)
            # For leaf nodes or when we're at max depth
            if not hasattr(node, "children") or not node.children or depth >= max_depth:
                return Tree.from_text(f"{{{node_type}}}")
            # Build tree string manually for APTED
            tree_parts = [node_type]
            child_count = 0
            for child in node.children:
                if child_count >= 5:  # Limit children
                    break
                try:
                    child_tree = self._simple_treesitter_to_apted(
                        child, depth + 1, max_depth
                    )
                    if child_tree:
                        # Extract the tree representation
                        child_repr = str(child_tree).strip("{}")
                        tree_parts.append(child_repr)
                        child_count += 1
                except:
                    continue

            # Create tree string in APTED format
            if len(tree_parts) == 1:
                tree_str = f"{{{tree_parts[0]}}}"
            else:
                tree_str = "{" + "{".join(tree_parts) + "}" * len(tree_parts)

            return Tree.from_text(tree_str)

        except Exception as e:
            return Tree.from_text("{error}")

    def _treesitter_to_apted(self, node) -> Tree:
        """Convert tree-sitter node to apted Tree"""
        try:
            # Get node type - handle both string and node types
            if hasattr(node, "type"):
                node_type = str(node.type)
            else:
                node_type = str(type(node).__name__)

            # Get children
            children = []
            if hasattr(node, "children") and node.children:
                # Limit the number of children to prevent excessive tree size
                for child in node.children[:10]:  # Limit to first 10 children
                    try:
                        child_tree = self._treesitter_to_apted(child)
                        if child_tree:
                            children.append(child_tree)
                    except Exception as e:
                        # Skip problematic children
                        continue

            # Create tree - APTED expects specific format
            # Use Tree.from_text() for simple node creation
            if children:
                # For nodes with children, create a proper tree structure
                tree_str = f"{{{node_type}"
                for i, child in enumerate(children):
                    tree_str += (
                        f"{{{child.name if hasattr(child, 'name') else str(child)}}}"
                    )
                tree_str += "}"
                return Tree.from_text(tree_str)
            else:
                # For leaf nodes, try to include text content if available and short
                try:
                    if hasattr(node, "text") and node.text:
                        text = node.text
                        if isinstance(text, bytes):
                            text = text.decode("utf-8", errors="ignore")
                        if len(str(text)) < 30:  # Limit text length
                            # Remove braces that confuse parser
                            clean_text = re.sub(r"[{}]", "", str(text))
                            return Tree.from_text(f"{{{node_type}:{clean_text}}}")
                except:
                    pass
                return Tree.from_text(f"{{{node_type}}}")

        except Exception as e:
            # Fallback to simple node representation
            return Tree.from_text(f"{{error_node_{hash(str(node)) % 1000}}}")

    def _tokenize_to_tree(self, code: str) -> Tree:
        """Create a simple token-based tree as fallback"""
        try:
            import re

            # Simple tokenization
            clean_code = self.clean_code(code)
            tokens = re.findall(r"\w+|[{}();,=+\-*/]", clean_code)

            # Limit tokens to prevent excessive computation
            tokens = tokens[:15]

            if not tokens:
                return Tree.from_text("{empty}")

            # Create a simple flat tree structure
            if len(tokens) == 1:
                return Tree.from_text(f"{{program{{{tokens[0]}}}}}")
            else:
                # Build nested structure manually
                tree_str = "{program"
                for token in tokens:
                    clean_token = re.sub(r"[^a-zA-Z0-9_]", "_", str(token))
                    tree_str += f"{{{clean_token}}}"
                tree_str += "}"
                return Tree.from_text(tree_str)

        except Exception as e:
            return Tree.from_text("{fallback}")

    def _python_ast_to_tree(self, node: Any) -> Tree:
        """Recursively convert Python AST node to Tree using from_text"""
        try:
            if isinstance(node, ast.AST):
                # Get node type
                node_type = type(node).__name__

                # Get child nodes
                children = []
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, ast.AST):
                                children.append(self._python_ast_to_tree(item))
                    elif isinstance(value, ast.AST):
                        children.append(self._python_ast_to_tree(value))

                # Create tree string for APTED
                if children and len(children) <= 5:  # Limit children
                    tree_str = f"{{{node_type}"
                    for child in children:
                        child_str = str(child).strip("{}")
                        tree_str += f"{{{child_str}}}"
                    tree_str += "}"
                    return Tree.from_text(tree_str)
                else:
                    return Tree.from_text(f"{{{node_type}}}")
            else:
                # For non-AST nodes, convert to string
                clean_str = re.sub(r"[^a-zA-Z0-9_]", "_", str(node)[:20])
                return Tree.from_text(f"{{{clean_str}}}")
        except:
            return Tree.from_text("{error}")

    def calculate_ast_edit_distance(self, code1: str, code2: str) -> float:
        """Calculate AST edit distance between two code snippets"""
        try:
            # First try to create trees
            tree1 = self.code_to_ast_tree(code1)
            tree2 = self.code_to_ast_tree(code2)

            # Validate trees
            if not tree1 or not tree2:
                print("Failed to create valid trees, using fallback")
                return self._token_based_distance(code1, code2)

            # Calculate edit distance with error handling
            try:
                apted = APTED(tree1, tree2)
                distance = apted.compute_edit_distance()
            except Exception as apted_error:
                print(f"APTED calculation failed: {apted_error}")
                return self._token_based_distance(code1, code2)

            # Normalize the distance
            # Simple normalization based on average tree complexity
            code1_tokens = len(re.findall(r"\w+", self.clean_code(code1)))
            code2_tokens = len(re.findall(r"\w+", self.clean_code(code2)))
            avg_complexity = (code1_tokens + code2_tokens) / 2

            if avg_complexity == 0:
                return 0.0

            # Normalize distance (0 = identical, 1 = completely different)
            normalized_distance = min(1.0, distance / max(avg_complexity, 1))
            return float(normalized_distance)

        except Exception as e:
            print(f"Error calculating AST edit distance: {e}")
            # Fallback: use token-based comparison
            return self._token_based_distance(code1, code2)

    def _get_tree_size(self, tree) -> int:
        """Get approximate size of tree"""
        try:
            if not hasattr(tree, "children") or not tree.children:
                return 1

            total_size = 1  # Count current node
            for child in tree.children:
                total_size += self._get_tree_size(child)
                if total_size > 100:  # Prevent excessive computation
                    return 100
            return total_size
        except:
            return 1

    def _token_based_distance(self, code1: str, code2: str) -> float:
        """Fallback token-based distance calculation"""
        try:
            # Clean and tokenize both codes
            clean1 = self.clean_code(code1)
            clean2 = self.clean_code(code2)

            # Simple tokenization
            tokens1 = re.findall(r"\w+|[{}();,=+\-*/]", clean1)
            tokens2 = re.findall(r"\w+|[{}();,=+\-*/]", clean2)

            if not tokens1 and not tokens2:
                return 0.0
            if not tokens1 or not tokens2:
                return 1.0

            # Calculate Jaccard distance
            set1 = set(tokens1)
            set2 = set(tokens2)

            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))

            if union == 0:
                return 0.0

            jaccard_similarity = intersection / union
            return 1.0 - jaccard_similarity

        except Exception as e:
            print(f"Fallback distance calculation failed: {e}")
            return 1.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance as fallback"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_code_embedding(self, code: str) -> np.ndarray:
        """Get CodeBERT embedding for code snippet"""
        try:
            # Clean the code
            clean_code = self.clean_code(code)

            # Tokenize
            tokens = self.tokenizer(
                clean_code,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**tokens)
                # Use mean pooling of last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.numpy().flatten()
        except Exception as e:
            print(f"Error getting CodeBERT embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(768)  # CodeBERT embedding dimension

    def calculate_codebert_similarity(self, code1: str, code2: str) -> float:
        """Calculate CodeBERT cosine similarity between two code snippets"""
        try:
            emb1 = self.get_code_embedding(code1)
            emb2 = self.get_code_embedding(code2)

            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating CodeBERT similarity: {e}")
            return 0.0

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe to add similarity metrics"""
        # Create a copy of the dataframe
        result_df = df.copy()

        # Initialize lists to store results
        ast_distances = []
        codebert_similarities = []

        for idx, row in df.iterrows():
            llm_response = str(row["LLM_response"])
            student_response = str(row["student_response"])

            # Calculate AST edit distance
            ast_dist = self.calculate_ast_edit_distance(llm_response, student_response)
            ast_distances.append(ast_dist)

            # Calculate CodeBERT similarity
            codebert_sim = self.calculate_codebert_similarity(
                llm_response, student_response
            )
            codebert_similarities.append(codebert_sim)

        # Add new columns to dataframe
        result_df["ast_edit_distance"] = ast_distances
        result_df["codebert_cosine_similarity"] = codebert_similarities

        return result_df
