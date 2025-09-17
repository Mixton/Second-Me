from concurrent.futures import ThreadPoolExecutor
import json
import os
import logging
import random
import re
import traceback
from typing import List
import hashlib
import pickle
from pathlib import Path
import time

import openai
import pandas as pd
from tqdm import tqdm
from enum import Enum
import tiktoken
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.L2.data_pipeline.data_prep.diversity.utils import dedup_by_similarity
import lpm_kernel.L2.data_pipeline.data_prep.diversity.template_diversity as template_diversity

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class DataSynthesisMode(Enum):
    LOW = {"large_aug_para":1, "tiny_aug_para":1, "mini_aug_para":1}
    MEDIUM = {"large_aug_para":2, "tiny_aug_para":2, "mini_aug_para":2}
    HIGH = {"large_aug_para":4, "tiny_aug_para":3, "mini_aug_para":2}


class TqdmLoggingHandler:
    def __init__(self):
        pass
    
    def write(self, msg):
        logger.info(msg.strip())
    
    def flush(self):
        pass
    
tqdm_handler = TqdmLoggingHandler()


class DiversityDataGenerator:
    """Generates diversity data for training language models.
    
    This class is responsible for creating diverse training data based on user notes,
    entities, and configurations. It leverages LLMs to generate questions and answers.
    """
    
    def __init__(self, preference_language: str, is_cot: bool = True, cache_dir: str = None, enable_cache: bool = True):
        """Initialize the diversity data generator.
        
        Args:
            preference_language: The language to use for generating data.
            is_cot: Whether to use chain of thought pattern.
            cache_dir: Directory to store cache files. If None, uses default cache directory.
            enable_cache: Whether to enable caching functionality.
        """
        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        if user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.model_name = user_llm_config.chat_model_name
    
            self.client = openai.OpenAI(
                api_key=user_llm_config.chat_api_key,
                base_url=user_llm_config.chat_endpoint,
            )
        self.preference_language = preference_language
        self.max_workers = os.environ.get("concurrency_threads", 2)
        self.data_synthesis_mode = os.environ.get("DATA_SYNTHESIS_MODE", "low")
        self.is_cot = is_cot
        if self.is_cot:
            logger.info("generate diversity data in longcot pattern!!!")
            self.model_name = user_llm_config.thinking_model_name
            self.api_key = user_llm_config.thinking_api_key
            self.base_url = user_llm_config.thinking_endpoint
            if self.model_name.startswith("deepseek"):
                self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                logger.error(f"Error model_name, longcot data generating model_name: deepseek series")
                raise
        
        # Initialize tokenizer for token management
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Could not initialize tiktoken, falling back to character counting")
            self.tokenizer = None
            
        # Initialize cache system
        self.enable_cache = enable_cache
        if cache_dir is None:
            self.cache_dir = Path("cache/diversity_data_generator")
        else:
            self.cache_dir = Path(cache_dir)
        
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.preprocess_cache_dir = self.cache_dir / "preprocess_cache"
            self.dedup_cache_dir = self.preprocess_cache_dir / "dedup_cache"
            self.pipeline_cache_dir = self.cache_dir / "pipeline_cache"
            self.llm_cache_dir = self.cache_dir / "llm_call_cache"
            self.question_cache_dir = self.llm_cache_dir / "questions"
            self.answer_cache_dir = self.llm_cache_dir / "answers"
            
            # Create all cache directories
            for cache_path in [self.preprocess_cache_dir, self.dedup_cache_dir, 
                              self.pipeline_cache_dir, self.question_cache_dir, self.answer_cache_dir]:
                cache_path.mkdir(parents=True, exist_ok=True)
                
            logger.info(f"Cache system enabled. Cache directory: {self.cache_dir}")
        else:
            logger.info("Cache system disabled")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to character count estimation (roughly 4 chars per token)
            return len(text) // 4

    def _generate_hash(self, *args) -> str:
        """Generate a hash from multiple arguments."""
        combined = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _save_cache(self, cache_path: Path, data):
        """Save data to cache file."""
        if not self.enable_cache:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def _load_cache(self, cache_path: Path):
        """Load data from cache file."""
        if not self.enable_cache or not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded cache: {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def _get_input_hash(self, entities_path: str, note_list: list, config_path: str, graph_path: str, user_name: str) -> str:
        """Generate hash for preprocess input parameters."""
        # Create hash from file modification times and input parameters
        try:
            entities_mtime = os.path.getmtime(entities_path) if os.path.exists(entities_path) else 0
            config_mtime = os.path.getmtime(config_path) if os.path.exists(config_path) else 0
            graph_mtime = os.path.getmtime(graph_path) if os.path.exists(graph_path) else 0
            
            # For note_list, create a hash of the content
            note_content_hash = self._generate_hash([item.to_json() if hasattr(item, 'to_json') else str(item) for item in note_list])
            
            return self._generate_hash(entities_path, entities_mtime, config_path, config_mtime, 
                                     graph_path, graph_mtime, user_name, note_content_hash)
        except Exception as e:
            logger.warning(f"Failed to generate input hash: {e}")
            return self._generate_hash(entities_path, config_path, graph_path, user_name, str(note_list))
    
    def _clean_cache(self, job_id: str = None):
        """Clean cache files. If job_id is provided, only clean that specific job's pipeline cache."""
        if not self.enable_cache:
            return
        try:
            if job_id:
                job_cache_dir = self.pipeline_cache_dir / job_id
                if job_cache_dir.exists():
                    import shutil
                    shutil.rmtree(job_cache_dir)
                    logger.info(f"Cleaned pipeline cache for job: {job_id}")
            else:
                # Clean all caches
                import shutil
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    logger.info(f"Cleaned all caches in: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean cache: {e}")
    
    def clean_all_cache(self):
        """Public method to clean all cache files."""
        self._clean_cache()
    
    def clean_preprocess_cache(self):
        """Clean only the preprocess cache."""
        if not self.enable_cache:
            return
        try:
            import shutil
            if self.preprocess_cache_dir.exists():
                shutil.rmtree(self.preprocess_cache_dir)
                self.preprocess_cache_dir.mkdir(parents=True, exist_ok=True)
                self.dedup_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleaned preprocess cache")
        except Exception as e:
            logger.warning(f"Failed to clean preprocess cache: {e}")
    
    def clean_llm_cache(self):
        """Clean only the LLM call cache."""
        if not self.enable_cache:
            return
        try:
            import shutil
            if self.llm_cache_dir.exists():
                shutil.rmtree(self.llm_cache_dir)
                self.question_cache_dir.mkdir(parents=True, exist_ok=True)
                self.answer_cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleaned LLM call cache")
        except Exception as e:
            logger.warning(f"Failed to clean LLM cache: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about cache usage."""
        if not self.enable_cache:
            return {"cache_enabled": False}
        
        def count_files(path):
            if not path.exists():
                return 0
            return len([f for f in path.iterdir() if f.is_file()])
        
        def get_size(path):
            if not path.exists():
                return 0
            total = 0
            for f in path.rglob('*'):
                if f.is_file():
                    total += f.stat().st_size
            return total
        
        stats = {
            "cache_enabled": True,
            "cache_dir": str(self.cache_dir),
            "preprocess_cache_files": count_files(self.preprocess_cache_dir),
            "dedup_cache_files": count_files(self.dedup_cache_dir),
            "question_cache_files": count_files(self.question_cache_dir),
            "answer_cache_files": count_files(self.answer_cache_dir),
            "pipeline_jobs": len([d for d in self.pipeline_cache_dir.iterdir() if d.is_dir()]) if self.pipeline_cache_dir.exists() else 0,
            "total_cache_size_mb": round(get_size(self.cache_dir) / (1024 * 1024), 2)
        }
        
        return stats

    def _split_large_note(self, note_dict: dict, max_tokens: int) -> List[dict]:
        """Split a large note into multiple smaller notes to preserve all content.
        
        Args:
            note_dict: The note dictionary to split
            max_tokens: Maximum tokens allowed per note chunk
            
        Returns:
            List of note chunks, each within token limits
        """
        # Extract note content based on available fields
        if "processed" in note_dict:
            content = note_dict["processed"]
            content_field = "processed"
        else:
            title = note_dict.get("title", "")
            content_body = note_dict.get("content", "")
            insight = note_dict.get("insight", "")
            content = f"Title: {title}\nContent: {content_body}\nAI Insight: {insight}"
            content_field = None
        
        total_tokens = self._count_tokens(content)
        if total_tokens <= max_tokens:
            # Return original if within limits
            return [note_dict.copy()]
        
        # Split content into chunks
        lines = content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        
        # Reserve tokens for metadata (title, chunk info, etc.)
        available_tokens = max_tokens - 200
        
        # Always include title in first chunk if present
        title_lines = []
        if lines and lines[0].startswith("Title:"):
            title_lines.append(lines[0])
            title_tokens = self._count_tokens(lines[0])
            if title_tokens < available_tokens:
                current_chunk_lines.append(lines[0])
                current_tokens = title_tokens
                lines = lines[1:]
        
        chunk_num = 1
        for line in lines:
            line_tokens = self._count_tokens(line + '\n')
            
            # Check if adding this line would exceed chunk limit
            if current_chunk_lines and (current_tokens + line_tokens > available_tokens):
                # Finalize current chunk
                chunk_content = '\n'.join(current_chunk_lines)
                if chunk_num > 1:
                    # Add chunk indicator and title reference for context
                    chunk_content = f"[NOTE CHUNK {chunk_num}/{chunk_num}+] {title_lines[0] if title_lines else ''}\n{chunk_content}"
                
                chunk = note_dict.copy()
                if content_field:
                    chunk[content_field] = chunk_content
                else:
                    # Parse back into structured format
                    chunk["title"] = f"{chunk.get('title', '')} (Part {chunk_num})"
                    chunk["content"] = chunk_content
                    chunk["insight"] = f"Chunk {chunk_num} of large note"
                
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_lines = []
                current_tokens = 0
                chunk_num += 1
                
                # Add title reference to new chunk for context
                if title_lines:
                    current_chunk_lines.extend(title_lines)
                    current_tokens = self._count_tokens('\n'.join(title_lines))
            
            # Add line to current chunk
            current_chunk_lines.append(line)
            current_tokens += line_tokens
        
        # Add final chunk if it has content
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if chunk_num > 1:
                # Update all previous chunks to show correct total
                for prev_chunk in chunks:
                    if content_field:
                        prev_content = prev_chunk[content_field]
                        prev_content = prev_content.replace(f"/{chunk_num}+]", f"/{chunk_num}]")
                        prev_chunk[content_field] = prev_content
                    else:
                        prev_chunk["insight"] = f"Chunk {chunks.index(prev_chunk)+1} of {chunk_num}"
                
                chunk_content = f"[NOTE CHUNK {chunk_num}/{chunk_num}] {title_lines[0] if title_lines else ''}\n{chunk_content}"
            
            chunk = note_dict.copy()
            if content_field:
                chunk[content_field] = chunk_content
            else:
                chunk["title"] = f"{chunk.get('title', '')} (Part {chunk_num})" if chunk_num > 1 else chunk.get('title', '')
                chunk["content"] = chunk_content
                chunk["insight"] = f"Chunk {chunk_num} of {chunk_num}" if chunk_num > 1 else chunk.get("insight", "")
            
            chunks.append(chunk)
        
        logger.info(f"Split large note ({total_tokens} tokens) into {len(chunks)} chunks")
        return chunks

    def _truncate_note_content(self, note_dict: dict, max_tokens: int) -> dict:
        """Intelligently truncate a note's content to fit within token limits.
        
        Args:
            note_dict: The note dictionary to truncate
            max_tokens: Maximum tokens allowed for this note
            
        Returns:
            Truncated note dictionary
        """
        # Extract note content based on available fields
        if "processed" in note_dict:
            content = note_dict["processed"]
        else:
            title = note_dict.get("title", "")
            content_body = note_dict.get("content", "")
            insight = note_dict.get("insight", "")
            content = f"Title: {title}\nContent: {content_body}\nAI Insight: {insight}"
        
        if self._count_tokens(content) <= max_tokens:
            # Return copy of original if within limits
            return note_dict.copy()
            
        # Truncate content intelligently
        lines = content.split('\n')
        result_lines = []
        current_tokens = 0
        
        # Always include title line if present
        if lines and lines[0].startswith("Title:"):
            title_line = lines[0]
            title_tokens = self._count_tokens(title_line)
            if title_tokens <= max_tokens - 100:  # Leave room for content
                result_lines.append(title_line)
                current_tokens += title_tokens
                lines = lines[1:]
        
        # Add remaining lines until we hit the token limit
        for line in lines:
            line_tokens = self._count_tokens(line + '\n')
            if current_tokens + line_tokens > max_tokens - 50:  # Leave buffer
                break
            result_lines.append(line)
            current_tokens += line_tokens
        
        # Add truncation indicator
        if len(result_lines) < len(content.split('\n')):
            result_lines.append("[CONTENT TRUNCATED FOR TOKEN LIMITS]")
        
        truncated_content = '\n'.join(result_lines)
        
        # Return modified copy of note_dict
        result_dict = note_dict.copy()
        if "processed" in result_dict:
            result_dict["processed"] = truncated_content
        else:
            # Reconstruct the truncated components
            truncated_lines = truncated_content.split('\n')
            new_title = ""
            new_content = ""
            new_insight = ""
            
            current_section = "title"
            for line in truncated_lines:
                if line.startswith("Title: "):
                    new_title = line[7:]  # Remove "Title: " prefix
                    current_section = "content"
                elif line.startswith("Content: "):
                    new_content = line[9:]  # Remove "Content: " prefix
                    current_section = "content"
                elif line.startswith("AI Insight: "):
                    new_insight = line[12:]  # Remove "AI Insight: " prefix
                    current_section = "insight"
                elif current_section == "content":
                    new_content += "\n" + line if new_content else line
                elif current_section == "insight":
                    new_insight += "\n" + line if new_insight else line
            
            result_dict["title"] = new_title or result_dict.get("title", "")
            result_dict["content"] = new_content or result_dict.get("content", "")
            result_dict["insight"] = new_insight or result_dict.get("insight", "")
        
        return result_dict

    def _create_cluster_chunks(self, cluster: dict, max_chunk_tokens: int = 15000, template_reserve: int = 8000) -> List[dict]:
        """Split a large cluster into smaller chunks that fit within token limits.
        
        Args:
            cluster: The cluster containing notes to process
            max_chunk_tokens: Maximum tokens per chunk (including template overhead)
            template_reserve: Tokens to reserve for system prompts and templates
            
        Returns:
            List of cluster chunks, each within token limits
        """
        notes = cluster.get("note", [])
        if not notes:
            return [cluster]
        
        # Calculate available tokens for actual content
        available_tokens_per_chunk = max_chunk_tokens - template_reserve
        
        chunks = []
        current_chunk_notes = []
        current_chunk_tokens = 0
        
        logger.info(f"Chunking cluster '{cluster.get('entity_name', 'unknown')}' with {len(notes)} notes")
        
        for note_dict in notes:
            # Calculate tokens for this note
            if "processed" in note_dict:
                content = note_dict["processed"]
            else:
                content = f"Title: {note_dict.get('title', '')}\nContent: {note_dict.get('content', '')}\nAI Insight: {note_dict.get('insight', '')}"
            
            note_tokens = self._count_tokens(content)
            
            # If this single note exceeds chunk limits, split it instead of truncating
            if note_tokens > available_tokens_per_chunk:
                logger.info(f"Note exceeds chunk limit ({note_tokens} > {available_tokens_per_chunk}), splitting into sub-notes")
                note_chunks = self._split_large_note(note_dict, available_tokens_per_chunk)
                
                # Process each note chunk
                for note_chunk in note_chunks:
                    # Recalculate tokens for the chunk
                    if "processed" in note_chunk:
                        chunk_content = note_chunk["processed"]
                    else:
                        chunk_content = f"Title: {note_chunk.get('title', '')}\nContent: {note_chunk.get('content', '')}\nAI Insight: {note_chunk.get('insight', '')}"
                    
                    chunk_tokens = self._count_tokens(chunk_content)
                    
                    # Check if adding this note chunk would exceed chunk limit
                    if current_chunk_notes and (current_chunk_tokens + chunk_tokens > available_tokens_per_chunk):
                        # Finalize current chunk
                        chunk = cluster.copy()
                        chunk["note"] = current_chunk_notes.copy()
                        chunk["chunk_info"] = f"chunk_{len(chunks)+1}_of_multiple"
                        chunks.append(chunk)
                        logger.info(f"Created chunk {len(chunks)} with {len(current_chunk_notes)} notes, {current_chunk_tokens} tokens")
                        
                        # Start new chunk
                        current_chunk_notes = []
                        current_chunk_tokens = 0
                    
                    # Add note chunk to current chunk
                    current_chunk_notes.append(note_chunk)
                    current_chunk_tokens += chunk_tokens
                
                continue  # Skip the original processing for this note
            
            # Check if adding this note would exceed chunk limit
            if current_chunk_notes and (current_chunk_tokens + note_tokens > available_tokens_per_chunk):
                # Finalize current chunk
                chunk = cluster.copy()
                chunk["note"] = current_chunk_notes.copy()
                chunk["chunk_info"] = f"chunk_{len(chunks)+1}_of_multiple"
                chunks.append(chunk)
                logger.info(f"Created chunk {len(chunks)} with {len(current_chunk_notes)} notes, {current_chunk_tokens} tokens")
                
                # Start new chunk
                current_chunk_notes = []
                current_chunk_tokens = 0
            
            # Add note to current chunk
            current_chunk_notes.append(note_dict)
            current_chunk_tokens += note_tokens
        
        # Add final chunk if it has notes
        if current_chunk_notes:
            chunk = cluster.copy()
            chunk["note"] = current_chunk_notes.copy()
            chunk["chunk_info"] = f"chunk_{len(chunks)+1}_of_multiple" if chunks else "single_chunk"
            chunks.append(chunk)
            logger.info(f"Created final chunk {len(chunks)} with {len(current_chunk_notes)} notes, {current_chunk_tokens} tokens")
        
        logger.info(f"Split cluster '{cluster.get('entity_name', 'unknown')}' into {len(chunks)} chunks")
        return chunks

    def _manage_cluster_tokens(self, cluster: dict, max_total_tokens: int = 20000) -> dict:
        """Manage token limits for a cluster of notes.
        
        This method now primarily serves as a fallback for small clusters
        that don't need chunking.
        
        Args:
            cluster: The cluster containing notes to process
            max_total_tokens: Maximum total tokens for all notes in the cluster
            
        Returns:
            Modified cluster with token-managed notes
        """
        notes = cluster.get("note", [])
        if not notes:
            return cluster
        
        # Reserve space for template text, entity info, system prompts and output
        available_tokens = max_total_tokens - 5000
        
        # Calculate tokens per note - be more reasonable since we now have chunking
        max_tokens_per_note = available_tokens // len(notes)
        max_tokens_per_note = max(500, min(max_tokens_per_note, 2500))  # More generous: Min 500, max 2.5k per note
        
        logger.info(f"Managing cluster '{cluster.get('entity_name', 'unknown')}' with {len(notes)} notes, {max_tokens_per_note} tokens each (max_total: {max_total_tokens})")
        
        # Process each note - use splitting for preservation, truncation only as fallback
        managed_notes = []
        total_tokens = 0
        
        for note_dict in notes:
            # Calculate current note size
            if "processed" in note_dict:
                content = note_dict["processed"]
            else:
                content = f"Title: {note_dict.get('title', '')}\nContent: {note_dict.get('content', '')}\nAI Insight: {note_dict.get('insight', '')}"
            
            note_tokens = self._count_tokens(content)
            
            # If note is too large, try splitting first
            if note_tokens > max_tokens_per_note:
                logger.info(f"Note exceeds token limit ({note_tokens} > {max_tokens_per_note}), attempting to split")
                note_chunks = self._split_large_note(note_dict, max_tokens_per_note)
                
                # Check if splitting would create too many notes for this cluster
                if len(managed_notes) + len(note_chunks) <= len(notes) * 2:  # Allow up to 2x notes through splitting
                    managed_notes.extend(note_chunks)
                    # Count tokens for all chunks
                    for chunk in note_chunks:
                        if "processed" in chunk:
                            chunk_content = chunk["processed"]
                        else:
                            chunk_content = f"Title: {chunk.get('title', '')}\nContent: {chunk.get('content', '')}\nAI Insight: {chunk.get('insight', '')}"
                        total_tokens += self._count_tokens(chunk_content)
                else:
                    # Fallback to truncation if splitting creates too many notes
                    logger.warning(f"Splitting would create too many notes, falling back to truncation")
                    managed_note = self._truncate_note_content(note_dict, max_tokens_per_note)
                    managed_notes.append(managed_note)
                    
                    # Count tokens for truncated note
                    if "processed" in managed_note:
                        note_tokens = self._count_tokens(managed_note["processed"])
                    else:
                        truncated_content = f"Title: {managed_note.get('title', '')}\nContent: {managed_note.get('content', '')}\nAI Insight: {managed_note.get('insight', '')}"
                        note_tokens = self._count_tokens(truncated_content)
                    total_tokens += note_tokens
            else:
                # Note is within limits, use as-is
                managed_notes.append(note_dict.copy())
                total_tokens += note_tokens
        
        # Create modified cluster
        managed_cluster = cluster.copy()
        managed_cluster["note"] = managed_notes
        
        logger.info(f"Cluster '{cluster.get('entity_name', 'unknown')}' final token usage: {total_tokens} tokens for {len(managed_notes)} notes")
        
        return managed_cluster

        

    def _preprocess(self, entities_path: str, note_list: list, config_path: str, graph_path: str, user_name: str):
        """Preprocess the input data for diversity generation.
        
        Args:
            entities_path: Path to entities data file.
            note_list: List of note objects.
            config_path: Path to configuration file.
            graph_path: Path to graph data file.
            user_name: Name of the user.
            
        Returns:
            Tuple containing entity descriptions, entity types, and QA configuration.
        """
        # Check preprocess cache
        if self.enable_cache:
            input_hash = self._get_input_hash(entities_path, note_list, config_path, graph_path, user_name)
            cache_path = self.preprocess_cache_dir / f"{input_hash}.pkl"
            cached_result = self._load_cache(cache_path)
            if cached_result is not None:
                logger.info(f"Loaded preprocess result from cache: {cache_path}")
                return cached_result
        
        logger.info("Running preprocess (not cached)")
        
        entity_df = pd.read_parquet(graph_path)
        entity2type = {
            item["title"]: item["type"] for item in entity_df.to_dict(orient="records")
        }

        # read entity2desc
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                entities = json.load(f)
                entity2desc = {
                    item["entity_name"]: {
                        key: value for key, value in item.items() if key != "entity_name"
                    }
                    for item in entities
                }
        except Exception as e:
            return None, None, None
        
        # read note data
        id2note = {
            item.id: {
                key: value for key, value in item.to_json().items() if key != "id"
            }
            for item in note_list
        }

        for entity, entity_info in entity2desc.copy().items():
            doc_ids = entity_info["doc_id"]
            tmp = []
            for doc_id in doc_ids:
                if isinstance(doc_id, str):
                    continue
                else:
                    note_desc = id2note.get(doc_id, "")
                    if note_desc:
                        tmp.append(note_desc)
            entity2desc[entity]["note"] = tmp

        entity2desc.pop(f"{user_name}", None)
        entity2desc.pop(f"{user_name.upper()}", None)

        # exclude keys with time format
        time_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        filtered_data = {
            k: v for k, v in entity2desc.items() if not re.match(time_pattern, k)
        }
        entity2desc = filtered_data

        # clean note level data with cached deduplication
        for entity, entity_info in entity2desc.copy().items():
            clusters = entity_info["note"]
            
            # Check dedup cache for this entity
            dedup_cache_path = None
            if self.enable_cache:
                entity_hash = self._generate_hash(entity, clusters)
                dedup_cache_path = self.dedup_cache_dir / f"{entity_hash}.pkl"
                cached_dedup = self._load_cache(dedup_cache_path)
                if cached_dedup is not None:
                    logger.debug(f"Loaded dedup result for entity {entity} from cache")
                    entity2desc[entity]["note"] = cached_dedup
                    continue
            
            # Run deduplication if not cached
            unique_dicts, cnt = dedup_by_similarity(clusters, similarity_threshold=0.9)
            entity2desc[entity]["note"] = unique_dicts
            
            # Save dedup result to cache
            if self.enable_cache and dedup_cache_path:
                self._save_cache(dedup_cache_path, unique_dicts)

        # read config file
        with open(config_path, "r", encoding="utf-8") as f:
            QA_config = json.load(f)

        result = (entity2desc, entity2type, QA_config)
        
        # Save preprocess result to cache
        if self.enable_cache:
            self._save_cache(cache_path, result)
            logger.info(f"Saved preprocess result to cache: {cache_path}")

        return result


    def _get_A_input(self, cluster: dict, question: str, user_name: str) -> str:
        """Generate the input for answer generation.
        
        Args:
            cluster: The data cluster containing entity information.
            question: The question to be answered.
            user_name: Name of the user.
            
        Returns:
            A string containing the formatted input for the answer generation model.
        """
        # Use the cluster as-is if it's already been chunked, otherwise apply token management
        if "chunk_info" in cluster:
            managed_cluster = cluster
            logger.info(f"Using pre-chunked cluster: {cluster['chunk_info']}")
        else:
            managed_cluster = self._manage_cluster_tokens(cluster, max_total_tokens=20000)
        
        entity = managed_cluster["entity_name"]
        entity_desc = managed_cluster["entity_description"]
        entity_desc = f"Entity'{entity}',Relevant Info：'{entity_desc}'"

        # Add chunk information to entity description if available
        if "chunk_info" in managed_cluster:
            entity_desc += f" ({managed_cluster['chunk_info']})"

        tmpl = f"""I am {user_name}. Regarding {entity_desc}, here is some information I previously mentioned:\n\n"""

        chunk_tmpl = ""
        for ind, entity_dict in enumerate(managed_cluster["note"]):
            if "processed" in entity_dict:
                content = entity_dict["processed"]
            else:
                content = entity_dict["content"]
                title = entity_dict["title"]
                insight = entity_dict["insight"]
                content = f"Title: {title}\nContent: {content}\nAI Insight: {insight}"

            tmp = f"___________________\n{content}\n"
            chunk_tmpl += tmp

        tmpl = (
            tmpl
            + chunk_tmpl
            + f"Based on the information I have previously recorded, please answer '{question}'. Note that you need to ensure the perspective is consistent, meaning that all instances of {user_name} should be replaced with the second person 'you'."
        )
        
        # Log final token usage for monitoring
        total_tokens = self._count_tokens(tmpl)
        logger.info(f"Generated A_input with {total_tokens} tokens for cluster '{entity}' ({managed_cluster.get('chunk_info', 'no chunk info')})")

        return tmpl


    def _get_Q_input(self, cluster: dict, user_name: str) -> str:
        """Generate the input for question generation.
        
        Args:
            cluster: The data cluster containing entity information.
            user_name: Name of the user.
            
        Returns:
            A string containing the formatted input for the question generation model.
        """
        # Use the cluster as-is if it's already been chunked, otherwise apply token management
        if "chunk_info" in cluster:
            managed_cluster = cluster
            logger.info(f"Using pre-chunked cluster: {cluster['chunk_info']}")
        else:
            managed_cluster = self._manage_cluster_tokens(cluster, max_total_tokens=20000)
        
        entity = managed_cluster["entity_name"]
        entity_desc = managed_cluster["entity_description"]
        entity_desc = f"Entity'{entity}'：{entity_desc}"
        
        # Add chunk information to entity description if available
        if "chunk_info" in managed_cluster:
            entity_desc += f" ({managed_cluster['chunk_info']})"
            
        tmpl = f""""For {entity_desc}, here is the relevant content from my interactions with the AI robot:\n"""
        chunk_tmpl = ""
        for ind, entity_dict in enumerate(managed_cluster["note"]):
            content = entity_dict["content"]
            title = entity_dict["title"]
            insight = entity_dict["insight"]
            content = f"Title: {title}\nContent: {content}\nAI Insight: {insight}"

            tmp = f"# Content {ind+1} #\n{content}\n"
            chunk_tmpl += tmp
        tmpl = (
            tmpl
            + chunk_tmpl
            + f"Please help me generate questions; note that you need to phrase them from my perspective, meaning all expressions of {user_name} should be replaced with the first person 'I'."
        )
        
        # Log final token usage for monitoring
        total_tokens = self._count_tokens(tmpl)
        logger.info(f"Generated Q_input with {total_tokens} tokens for cluster '{entity}' ({managed_cluster.get('chunk_info', 'no chunk info')})")

        return tmpl


    def generate_data(self, entities_path: str, note_list: list, config_path: str, 
                     graph_path: str, user_name: str, global_bio: str, output_path: str, resume_from_cache: bool = True):
        """Generate diversity data based on user notes and entities.
        
        Args:
            entities_path: Path to entities data file.
            note_list: List of note objects.
            config_path: Path to configuration file.
            graph_path: Path to graph data file.
            user_name: Name of the user.
            global_bio: User biography text.
            output_path: Path to save the generated data.
            resume_from_cache: Whether to resume from cached pipeline steps.
        """
        # Generate job ID for this run
        job_id = self._generate_hash(entities_path, config_path, graph_path, user_name, global_bio, 
                                   self.data_synthesis_mode, str(time.time())[:10])  # Include date for uniqueness
        logger.info(f"Starting data generation job: {job_id}")
        
        # Create job-specific cache directory
        if self.enable_cache:
            job_cache_dir = self.pipeline_cache_dir / job_id
            job_cache_dir.mkdir(parents=True, exist_ok=True)
        
        language_desc = f"Keep your response in {self.preference_language}"

        entity2desc, entity2type, QA_config = self._preprocess(
            entities_path, note_list, config_path, graph_path, user_name
        )

        if entity2desc is None:
            return 

        tmp = QA_config["query"]

        q_dict = {item["type"]: {k: item[k] for k in item if k != "type"} for item in tmp}

        tmp = QA_config["answer"]
        a_dict = {item["type"]: {k: item[k] for k in item if k != "type"} for item in tmp}

        templater = template_diversity.templater(
            q_dict, a_dict, user_name, global_bio, self.is_cot
        )

        entity2desc_list = [{**{"entity_name": k}, **v} for k, v in entity2desc.items()]

        # global questions, only process clusters with more than 8 notes, and split very large clusters
        large_clusters = [item for item in entity2desc_list if len(item["note"]) >= 8]
        logger.info(f"Large clusters: {len(large_clusters)}")

        exploded_clusters = []
        # split
        for sub_dict in large_clusters:
            for i in range(0, len(sub_dict["note"]), 4):
                tmp_dict = sub_dict.copy()

                tmp_dict["note"] = sub_dict["note"][i : i + 4]
                tmp_dict["doc_id"] = sub_dict["doc_id"][i : i + 4]
                exploded_clusters.append(tmp_dict)

            # ensure global effect, add some large global data
            notes_and_ids = list(zip(sub_dict["note"], sub_dict["doc_id"]))
            for _ in range(len(sub_dict["note"]) // 10 + 1):
                tmp_dict = sub_dict.copy()
                sampled_notes_and_ids = random.sample(
                    notes_and_ids, min(10, len(notes_and_ids))
                )
                tmp_dict["note"], tmp_dict["doc_id"] = zip(
                    *sampled_notes_and_ids
                )  # Unpack into two lists
                exploded_clusters.append(tmp_dict)

        # process small clusters
        mini_clusters = [
            item
            for item in entity2desc_list
            if len(item["note"]) < 8 and len(item["note"]) > 1
        ]

        logger.info(f"Mini clusters: {len(mini_clusters)}")

        # process other clusters
        tiny_clusters = [item for item in entity2desc_list if len(item["note"]) <= 1]

        logger.info(f"Tiny clusters: {len(tiny_clusters)}")

        filtered_tiny_clusters = [
            d
            for d in tiny_clusters
            if entity2type.get(d["entity_name"], "")
            in ["PERSON", "人", "组织", "ORGANIZATION", "人物"]
        ]

        logger.info(f"Filtered tiny clusters: {len(filtered_tiny_clusters)}")

        # Process large clusters with caching
        data_large = []
        if len(exploded_clusters) > 0:
            large_cache_path = job_cache_dir / "large_clusters_done.pkl" if self.enable_cache else None
            if resume_from_cache and large_cache_path and large_cache_path.exists():
                data_large = self._load_cache(large_cache_path)
                logger.info(f"Loaded large cluster results from cache ({len(data_large)} entries)")
            else:
                logger.info("Execute large cluster generation")
                data_large = self._pipline(exploded_clusters, DataSynthesisMode[self.data_synthesis_mode.upper()].value["large_aug_para"], 
                                           q_dict, templater, language_desc, user_name, job_id)
                if self.enable_cache:
                    self._save_cache(large_cache_path, data_large)
                    logger.info(f"Saved large cluster results to cache")
        else:
            logger.info("Large cluster number is 0")

        # Process mini clusters with caching
        data_mini = []
        if len(mini_clusters) > 0:
            mini_cache_path = job_cache_dir / "mini_clusters_done.pkl" if self.enable_cache else None
            if resume_from_cache and mini_cache_path and mini_cache_path.exists():
                data_mini = self._load_cache(mini_cache_path)
                logger.info(f"Loaded mini cluster results from cache ({len(data_mini)} entries)")
            else:
                logger.info("Execute small cluster generation")
                data_mini = self._pipline(mini_clusters, DataSynthesisMode[self.data_synthesis_mode.upper()].value["mini_aug_para"], 
                                          q_dict, templater, language_desc, user_name, job_id)
                if self.enable_cache:
                    self._save_cache(mini_cache_path, data_mini)
                    logger.info(f"Saved mini cluster results to cache")
        else:
            logger.info("Small cluster number is 0")

        # Process tiny clusters with caching  
        data_tiny = []
        if len(filtered_tiny_clusters) > 0:
            tiny_cache_path = job_cache_dir / "tiny_clusters_done.pkl" if self.enable_cache else None
            if resume_from_cache and tiny_cache_path and tiny_cache_path.exists():
                data_tiny = self._load_cache(tiny_cache_path)
                logger.info(f"Loaded tiny cluster results from cache ({len(data_tiny)} entries)")
            else:
                logger.info("Execute single entity cluster generation")
                q_dict_copy = q_dict.copy()  # Don't modify the original
                q_dict_copy.pop("unanswerable", None)
                q_dict_copy.pop("global", None)
                data_tiny = self._pipline(filtered_tiny_clusters, DataSynthesisMode[self.data_synthesis_mode.upper()].value["tiny_aug_para"], 
                                          q_dict_copy, templater, language_desc, user_name, job_id)
                if self.enable_cache:
                    self._save_cache(tiny_cache_path, data_tiny)
                    logger.info(f"Saved tiny cluster results to cache")
        else:
            logger.info("Single entity cluster number is 0")

        combined_list = data_large + data_mini + data_tiny
        # calculate total entries
        total_entries = len(combined_list)
        logger.info(f"Total entries: {total_entries}")
        # store data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_list, f, ensure_ascii=False, indent=4)

        logger.info(f"Data has been stored to {output_path}")
        
        # Clean up job-specific cache after successful completion
        if self.enable_cache:
            self._clean_cache(job_id)
            logger.info(f"Cleaned up pipeline cache for completed job: {job_id}")


    def _pipline(self, clusters: list, aug_para: int, q_dict: dict, 
                templater, language_desc: str, user_name: str, job_id: str = None) -> list:
        """Execute the pipeline for data generation.
        
        Args:
            clusters: List of data clusters.
            aug_para: Data augmentation coefficient.
            q_dict: Dictionary of question types.
            templater: Template handler object.
            language_desc: Language description string.
            user_name: Name of the user.
            
        Returns:
            List of generated QA data.
        """
        # Step 1: Apply chunking to large clusters to prevent token overflow
        processed_clusters = []
        for cluster in clusters:
            notes = cluster.get("note", [])
            if len(notes) > 0:
                # Estimate total tokens for this cluster
                total_tokens = 0
                for note in notes:
                    if "processed" in note:
                        content = note["processed"]
                    else:
                        content = f"Title: {note.get('title', '')}\nContent: {note.get('content', '')}\nAI Insight: {note.get('insight', '')}"
                    total_tokens += self._count_tokens(content)
                
                # If cluster is too large, chunk it
                if total_tokens > 12000:  # Conservative threshold before template overhead
                    chunks = self._create_cluster_chunks(cluster, max_chunk_tokens=15000, template_reserve=8000)
                    processed_clusters.extend(chunks)
                    logger.info(f"Chunked cluster '{cluster.get('entity_name', 'unknown')}' from {total_tokens} tokens into {len(chunks)} chunks")
                else:
                    processed_clusters.append(cluster)
            else:
                processed_clusters.append(cluster)
        
        # Step 2: Explode clusters for augmentation
        explode_clusters = []
        explode_questions_types = []
        for item in processed_clusters:
            # add elements multiple times based on aug_para
            explode_clusters.extend([item] * aug_para)
            # randomly select different types based on weights
            weights = [v["weight"] for v in q_dict.values()]
            random_types = random.choices(list(q_dict.keys()), weights, k=aug_para)
            explode_questions_types.extend(random_types)

        logger.info("Start generating data")
        logger.info(f"Original clusters: {len(clusters)}")
        logger.info(f"After chunking: {len(processed_clusters)} chunks")
        logger.info(f"Exploded clusters: {len(explode_clusters)}")
        logger.info(f"Explode questions types: {len(explode_questions_types)}")

        questions, answers, answer_types, flat_question_types, flat_clusters = self._generate(
            explode_clusters, explode_questions_types, templater, q_dict, language_desc, user_name, job_id
        )

        # store data
        data = []
        for cluster, question, answer, question_type, answer_type in zip(
            flat_clusters, questions, answers, flat_question_types, answer_types
        ):
            if len(question) == 0 or len(answer) == 0:
                continue
            
            # Include chunk information in the data if available
            entity_name = cluster["entity_name"]
            if "chunk_info" in cluster:
                entity_name = f"{entity_name}_{cluster['chunk_info']}"
            
            data.append(
                {
                    "user": question,
                    "assistant": answer,
                    "entity_name": entity_name,
                    "question_type": question_type,
                    "answer_type": answer_type,
                    "doc_id": cluster["doc_id"],
                }
            )
        return data


    def _generate(self, explode_clusters: list, explode_questions_types: list, 
                 templater, q_dict: dict, language_desc: str, user_name: str, job_id: str = None) -> tuple:
        """Generate questions and answers using ThreadPoolExecutor.
        
        Args:
            explode_clusters: List of expanded data clusters.
            explode_questions_types: List of question types to generate.
            templater: Template handler object.
            q_dict: Dictionary of question types.
            language_desc: Language description string.
            user_name: Name of the user.
            
        Returns:
            Tuple of (questions, answers, answer_types, flat_question_types, flat_clusters).
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._Q_generate, cluster, question_type, templater, q_dict, language_desc, user_name, job_id)
                for cluster, question_type in zip(
                    explode_clusters, explode_questions_types
                )
            ]
            questions = []
            flat_clusters = []
            flat_question_types = []
            cnt = 0
            for future, cluster, question_type in zip(
                tqdm(futures, total=len(futures), desc="Q_generate", file=tqdm_handler),
                explode_clusters,
                explode_questions_types,
            ):
                try:
                    result = future.result()
                    cnt += 1 if result else 0
                    # Assuming result is a list of questions
                    questions.extend(result)
                    # Extend clusters and question types to match the number of questions
                    flat_clusters.extend([cluster] * len(result))
                    flat_question_types.extend([question_type] * len(result))
                except Exception as e:
                    logger.error(traceback.format_exc())

        # safety check
        logger.info(f"Count: {cnt}, len(explode_clusters): {len(explode_clusters)}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._A_generate, cluster, question, question_type, templater, language_desc, user_name, job_id)
                for cluster, question, question_type in zip(
                    flat_clusters, questions, flat_question_types
                )
            ]

            answers = []
            answer_types = []

            for future in tqdm(futures, total=len(futures), desc="A_generate", file=tqdm_handler):
                try:
                    result, answer_type = future.result()
                    answers.append(result)
                    answer_types.append(answer_type)
                except Exception as e:
                    logger.error(traceback.format_exc())

        return questions, answers, answer_types, flat_question_types, flat_clusters


    def _Q_generate(self, cluster: dict, question_type: str, templater, 
                   q_dict: dict, language_desc: str, user_name: str, job_id: str = None) -> list:
        """Generate questions based on the given cluster and type.
        
        Args:
            cluster: The data cluster containing entity information.
            question_type: Type of questions to generate.
            templater: Template handler object.
            q_dict: Dictionary of question types.
            language_desc: Language description string.
            user_name: Name of the user.
            job_id: Job ID for this generation run.
            
        Returns:
            List of generated questions.
        """
        user_input = self._get_Q_input(cluster, user_name)

        system_prompt = templater.get_Q_template(
            question_type_prompt=q_dict[question_type]["prompt"]
        )
        
        # Check LLM call cache
        if self.enable_cache:
            # Create cache key from all input parameters
            cache_key = self._generate_hash(
                cluster.get('entity_name', ''), 
                cluster.get('chunk_info', ''),
                str(cluster.get('note', [])),  # Convert notes to string for hashing
                question_type,
                system_prompt,
                user_input,
                language_desc
            )
            cache_path = self.question_cache_dir / f"{cache_key}.pkl"
            cached_questions = self._load_cache(cache_path)
            if cached_questions is not None:
                logger.debug(f"Loaded Q_generate result from cache for '{cluster.get('entity_name', 'unknown')}'")
                return cached_questions
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input + language_desc},
        ]
        
        # Log token usage for monitoring (chunking should prevent most overflows)
        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(user_input + language_desc)
        total_tokens = system_tokens + user_tokens
        
        logger.info(f"Q_generate tokens for '{cluster.get('entity_name', 'unknown')}': "
                   f"system={system_tokens}, user={user_tokens}, total={total_tokens}")
        
        # Emergency safety check - this should rarely trigger with chunking
        if total_tokens > 120000:  # Much higher threshold since chunking prevents most issues
            logger.error(f"EMERGENCY: Token count {total_tokens} exceeds model limit! "
                        f"Chunking failed for cluster '{cluster.get('entity_name', 'unknown')}'")
            return []
        
        res = ""  # Initialize res to handle API failures gracefully
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            if self.is_cot:
                response_message = response.choices[0].message
                res = "<think>" + response_message.reasoning_content + "</think>" + response_message.content
            else:
                res = response.choices[0].message.content
        except Exception as e:
            logger.error(f"API call failed for cluster '{cluster.get('entity_name', 'unknown')}': {str(e)}")
            logger.error(traceback.format_exc())
            return []  # Return empty list on API failure
        
        # post-processing
        try:
            pattern = r"Question\s*\d+\s*:\s*(.*?)\|\|"
            questions = re.findall(pattern, res + "||")
        except Exception as e:
            logger.error(f"Failed to parse questions from response: {str(e)}")
            logger.error(traceback.format_exc())
            questions = []
            return questions

        # safety check
        if questions:
            if "|" in questions[0] and len(questions) == 0:
                questions = questions[0].split("|")

        # Save to cache
        if self.enable_cache:
            self._save_cache(cache_path, questions)
            logger.debug(f"Saved Q_generate result to cache for '{cluster.get('entity_name', 'unknown')}'")

        return questions


    def _A_generate(self, cluster: dict, question: str, question_type: str, 
                   templater, language_desc: str, user_name: str, job_id: str = None) -> tuple:
        """Generate answers based on questions and clusters.
        
        Args:
            cluster: The data cluster containing entity information.
            question: The question to answer.
            question_type: Type of question.
            templater: Template handler object.
            language_desc: Language description string.
            user_name: Name of the user.
            job_id: Job ID for this generation run.
            
        Returns:
            Tuple of (answer_text, answer_type).
        """
        user_input = self._get_A_input(cluster, question, user_name)
        system_prompt, answer_type = templater.get_A_template(question_type)
        
        # Check LLM call cache
        if self.enable_cache:
            # Create cache key from all input parameters
            cache_key = self._generate_hash(
                cluster.get('entity_name', ''), 
                cluster.get('chunk_info', ''),
                str(cluster.get('note', [])),  # Convert notes to string for hashing
                question,
                question_type,
                system_prompt,
                user_input,
                language_desc
            )
            cache_path = self.answer_cache_dir / f"{cache_key}.pkl"
            cached_answer = self._load_cache(cache_path)
            if cached_answer is not None:
                logger.debug(f"Loaded A_generate result from cache")
                return cached_answer
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input + language_desc},
        ]
        
        # Log token usage for monitoring (chunking should prevent most overflows)
        system_tokens = self._count_tokens(system_prompt)
        user_tokens = self._count_tokens(user_input + language_desc)
        total_tokens = system_tokens + user_tokens
        
        logger.info(f"A_generate tokens for answer generation: "
                   f"system={system_tokens}, user={user_tokens}, total={total_tokens}")
        
        # Emergency safety check - this should rarely trigger with chunking
        if total_tokens > 120000:  # Much higher threshold since chunking prevents most issues
            logger.error(f"EMERGENCY: Answer generation token count {total_tokens} exceeds model limit!")
            return "[ERROR: Content too large despite chunking]", answer_type
        
        res = ""  # Initialize res to handle API failures gracefully
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            if self.is_cot:
                response_message = response.choices[0].message
                res = "<think>" + response_message.reasoning_content + "</think>" + response_message.content
            else:
                res = response.choices[0].message.content
        except Exception as e:
            logger.error(f"API call failed for answer generation: {str(e)}")
            logging.error(traceback.format_exc())
            res = "[ERROR: Could not generate answer due to API failure]"
        
        result = (res, answer_type)
        
        # Save to cache
        if self.enable_cache:
            self._save_cache(cache_path, result)
            logger.debug(f"Saved A_generate result to cache")
            
        return result