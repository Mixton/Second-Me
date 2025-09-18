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
import sys
import gc
import shutil

try:
    import psutil
except ImportError:
    psutil = None

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
    
    # Pre-compile regex patterns to avoid recompilation and reduce memory usage
    CHAT_PATTERNS = [
        re.compile(r'^(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4},\s*\d{1,2}:\d{2})\s*-\s*([^:]+?):\s*(.*)$', re.IGNORECASE),
        re.compile(r'^\[(\d{1,2}:\d{2})\]\s*([^:]+?):\s*(.*)$', re.IGNORECASE),
        re.compile(r'^([^-]+)\s*-\s*(Today at \d{1,2}:\d{2}\s*(AM|PM))\s*\n(.+)$', re.IGNORECASE),
        re.compile(r'^([^:]+?):\s*(.*)$', re.IGNORECASE),
    ]
    
    def __init__(self, preference_language: str, is_cot: bool = True, cache_dir: str = None, enable_cache: bool = True,
                 enable_relevance_filtering: bool = True, relevance_threshold: float = 0.15):
        """Initialize the diversity data generator.
        
        Args:
            preference_language: The language to use for generating data.
            is_cot: Whether to use chain of thought pattern.
            cache_dir: Directory to store cache files. If None, uses default cache directory.
            enable_cache: Whether to enable caching functionality.
            enable_relevance_filtering: Whether to extract relevant chat segments from notes.
            relevance_threshold: Minimum relevance score for non-chat content filtering.
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

        # Chat segment extraction configuration
        self.enable_relevance_filtering = enable_relevance_filtering
        self.relevance_threshold = relevance_threshold

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
    
    def _generate_clusters_hash(self, entity_name: str, clusters: list) -> str:
        """Generate a memory-efficient hash for clusters data.
        
        This avoids serializing the entire clusters content which can cause OOM issues
        for entities with large amounts of data.
        
        Args:
            entity_name: Name of the entity
            clusters: List of cluster dictionaries
            
        Returns:
            Hash string representing the clusters signature
        """
        # Create a lightweight signature instead of serializing all data
        signature_parts = [entity_name, len(clusters)]
        
        # Add signatures from first few clusters and a sampling of others
        sample_size = min(10, len(clusters))  # Sample max 10 clusters
        indices_to_sample = []
        
        if len(clusters) <= 10:
            indices_to_sample = list(range(len(clusters)))
        else:
            # Sample first 5, last 5, and some middle ones
            indices_to_sample.extend(range(5))  # First 5
            indices_to_sample.extend(range(len(clusters)-5, len(clusters)))  # Last 5
            
        for i in indices_to_sample:
            if i < len(clusters):
                cluster = clusters[i]
                # Create a lightweight signature for this cluster
                cluster_sig = [
                    len(str(cluster.get('content', ''))),  # Content length
                    cluster.get('title', '')[:50],  # First 50 chars of title
                    len(cluster.get('insight', '')),  # Insight length
                    str(cluster.get('timestamp', ''))[:20]  # Timestamp prefix
                ]
                signature_parts.extend(cluster_sig)
        
        # Convert to string and hash
        signature_str = '|'.join(str(part) for part in signature_parts)
        return hashlib.md5(signature_str.encode('utf-8')).hexdigest()
    
    def _generate_deterministic_seed(self, *args) -> int:
        """Generate a deterministic seed from multiple arguments.
        
        Args:
            *args: Arguments to include in seed generation
            
        Returns:
            Deterministic integer seed within valid range
        """
        # Create a consistent hash from all arguments
        hash_str = self._generate_hash(*args)
        # Convert to integer and ensure it's within valid seed range
        return int(hash_str[:8], 16) % (2**32)
    
    def _save_cache(self, cache_path: Path, data):
        """Save data to cache file."""
        if not self.enable_cache:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def _load_cache(self, cache_path: Path):
        """Load data from cache file."""
        if not self.enable_cache or not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded cache: {cache_path} (size: {self._get_memory_size(data):.2f} MB)")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None
    
    def _get_memory_size(self, obj) -> float:
        """Get approximate memory size of an object in MB."""
        try:
            import sys
            return sys.getsizeof(obj) / (1024 * 1024)
        except:
            return 0.0
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        try:
            import gc
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
        except:
            pass
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except ImportError:
            logger.warning("psutil not available, cannot monitor memory usage")
            return 0.0
        except:
            return 0.0
    
    def _log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        memory_mb = self._get_current_memory_usage()
        if memory_mb > 0:
            logger.info(f"Memory usage {context}: {memory_mb:.1f} MB")
            
            # Warn if memory usage is very high
            if memory_mb > 4000:  # 4GB
                logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
                
            # Force garbage collection if memory is critically high
            if memory_mb > 6000:  # 6GB
                logger.warning(f"Critical memory usage: {memory_mb:.1f} MB - forcing garbage collection")
                self._force_garbage_collection()
    
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
        
        preprocess_size = get_size(self.preprocess_cache_dir)
        llm_size = get_size(self.llm_cache_dir)
        pipeline_size = get_size(self.pipeline_cache_dir)
        total_size = get_size(self.cache_dir)
        
        stats = {
            "cache_enabled": True,
            "cache_dir": str(self.cache_dir),
            "current_memory_usage_mb": self._get_current_memory_usage(),
            "preprocess_cache_files": count_files(self.preprocess_cache_dir),
            "preprocess_cache_size_mb": round(preprocess_size / (1024 * 1024), 2),
            "dedup_cache_files": count_files(self.dedup_cache_dir),
            "question_cache_files": count_files(self.question_cache_dir),
            "answer_cache_files": count_files(self.answer_cache_dir),
            "llm_cache_size_mb": round(llm_size / (1024 * 1024), 2),
            "pipeline_jobs": len([d for d in self.pipeline_cache_dir.iterdir() if d.is_dir()]) if self.pipeline_cache_dir.exists() else 0,
            "pipeline_cache_size_mb": round(pipeline_size / (1024 * 1024), 2),
            "total_cache_size_mb": round(total_size / (1024 * 1024), 2)
        }
        
        return stats
    
    def clean_large_cache_files(self, max_size_mb: float = 100.0):
        """Remove cache files larger than the specified size to free memory.
        
        Args:
            max_size_mb: Maximum size in MB for cache files to keep
        """
        if not self.enable_cache:
            return
        
        removed_files = 0
        freed_mb = 0.0
        
        try:
            for cache_file in self.cache_dir.rglob('*.pkl'):
                if cache_file.is_file():
                    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                    if file_size_mb > max_size_mb:
                        freed_mb += file_size_mb
                        cache_file.unlink()
                        removed_files += 1
                        logger.info(f"Removed large cache file: {cache_file} ({file_size_mb:.2f} MB)")
            
            if removed_files > 0:
                logger.info(f"Cleaned {removed_files} large cache files, freed {freed_mb:.2f} MB")
                self._force_garbage_collection()
            else:
                logger.info("No large cache files found to clean")
                
        except Exception as e:
            logger.warning(f"Failed to clean large cache files: {e}")
    
    def emergency_memory_cleanup(self):
        """Emergency memory cleanup for critical memory situations."""
        logger.warning("Performing emergency memory cleanup")
        
        try:
            
            # if self.enable_cache and self.cache_dir.exists():
            #     corrupted_files = 0
            #     for cache_file in self.cache_dir.rglob('*.pkl'):
            #         try:
            #             # Try to open file briefly to check if it's corrupted
            #             with open(cache_file, 'rb') as f:
            #                 pickle.load(f)
            #         except Exception:
            #             # File is corrupted, remove it
            #             try:
            #                 cache_file.unlink()
            #                 corrupted_files += 1
            #                 logger.info(f"Removed corrupted cache file: {cache_file}")
            #             except:
            #                 pass
                
            #     if corrupted_files > 0:
            #         logger.info(f"Removed {corrupted_files} corrupted cache files")
            
            # Force multiple garbage collections
            for i in range(3):
                collected = gc.collect()
                logger.info(f"Garbage collection round {i+1}: freed {collected} objects")
            
            # Log memory usage after cleanup
            final_memory = self._get_current_memory_usage()
            logger.info(f"Memory usage after emergency cleanup: {final_memory:.1f} MB")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")

    def _extract_relevant_chat_segments(self, chat_content: str, entity_name: str, entity_description: str = "", 
                                       context_window: int = 3) -> list:
        """Extract relevant segments from chat conversations while preserving context.
        
        Args:
            chat_content: The full chat conversation content
            entity_name: Name of the entity to find relevant segments for
            entity_description: Description of the entity
            context_window: Number of messages before/after relevant message to include for context
            
        Returns:
            List of relevant chat segments with context
        """
        try:
            # Split chat into individual messages (handles various chat formats)
            messages = self._parse_chat_messages(chat_content)
            
            if not messages:
                return []
            
            entity_lower = entity_name.lower()
            desc_words = set(entity_description.lower().split()) if entity_description else set()
            
            relevant_segments = []
            relevant_indices = set()
            
            # Find messages that mention the entity or related topics
            for i, message in enumerate(messages):
                message_lower = message.get('content', '').lower()
                
                # Direct entity mentions
                if entity_lower in message_lower:
                    relevant_indices.add(i)
                    continue
                
                # Description keyword matches (need multiple matches for relevance)
                if desc_words:
                    message_words = set(message_lower.split())
                    overlap_count = len(desc_words.intersection(message_words))
                    if overlap_count >= 2:  # At least 2 keywords match
                        relevant_indices.add(i)
                        continue
                
                # Topic coherence - look for related discussion
                if self._is_topic_related_message(message_lower, entity_lower, desc_words):
                    relevant_indices.add(i)
            
            # Expand relevant indices to include context
            expanded_indices = set()
            for idx in relevant_indices:
                start = max(0, idx - context_window)
                end = min(len(messages), idx + context_window + 1)
                expanded_indices.update(range(start, end))
            
            # Create segments from consecutive expanded indices
            if expanded_indices:
                sorted_indices = sorted(expanded_indices)
                segments = []
                current_segment = []
                
                for i in range(len(sorted_indices)):
                    current_idx = sorted_indices[i]
                    current_segment.append(messages[current_idx])
                    
                    # Check if next index is consecutive
                    if (i == len(sorted_indices) - 1 or 
                        sorted_indices[i + 1] != current_idx + 1):
                        # End of consecutive segment
                        if current_segment:
                            segment_text = self._format_chat_segment(current_segment, entity_name)
                            if segment_text:
                                segments.append(segment_text)
                        current_segment = []
                
                return segments
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to extract chat segments: {e}")
            return []

    def _parse_chat_messages(self, chat_content: str) -> list:
        """Parse chat content into individual messages with timestamps and speakers.
        
        Args:
            chat_content: Raw chat conversation text
            
        Returns:
            List of message dictionaries with 'timestamp', 'speaker', 'content'
        """
        # Pre-filter to remove WhatsApp system messages using simple string matching
        # Use simple string operations instead of complex regex for better memory efficiency
        
        # Define system message keywords to look for (much faster than regex)
        system_keywords = [
            "messages et les appels sont chiffrés de bout en bout",
            "messages and calls are end-to-end encrypted", 
            "chiffrés de bout en bout",
            "encrypted end-to-end",
            "partager. en savoir plus",
            "tap to learn more",
            "seules les personnes prenant part",
            "en savoir plus",
            "learn more",
            "code de sécurité",
            "security code"
        ]
        
        # Filter lines using simple string matching (memory efficient)
        filtered_lines = []
        for line in chat_content.split('\n'):
            logger.debug(f"Processing line (filtered): {line}")
            line_lower = line.lower().strip()
            
            # Skip very short lines that might be fragments
            if len(line_lower) < 10:
                continue
            
            # Check if line contains any system message keywords
            is_system_line = False
            for keyword in system_keywords:
                if keyword in line_lower:
                    is_system_line = True
                    break
            
            if not is_system_line:
                filtered_lines.append(line)
        
        messages = []
        lines = filtered_lines
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            matched = False
            for pattern in self.CHAT_PATTERNS:
                logger.debug(f"Trying pattern {pattern.pattern} on line: {line}")
                match = pattern.match(line)
                if match:
                    if len(match.groups()) == 3:  # timestamp, speaker, content
                        timestamp, speaker, content = match.groups()
                    elif len(match.groups()) == 2:  # speaker, content (no timestamp)
                        speaker, content = match.groups()
                        timestamp = None
                    elif len(match.groups()) == 4:  # Discord format
                        speaker, timestamp, _, content = match.groups()
                    else:
                        i += 1
                        continue
                    
                    # Look ahead for multi-line messages
                    full_content = content
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                        # Check if next line is another message
                        is_next_message = any(p.match(next_line) for p in self.CHAT_PATTERNS)
                        if is_next_message:
                            break
                        full_content += " " + next_line
                        j += 1
                    
                    # Check if this is a media-only message using simple string comparison (faster)
                    full_content_lower = full_content.lower().strip()
                    
                    # Use simple string matching instead of regex for better performance
                    # Check for media indicators (case insensitive)
                    media_indicators = ["médias omis", "media omitted", "omitted"]
                    is_media_only = any(indicator in full_content_lower for indicator in media_indicators)
                    
                    # Skip media-only messages entirely
                    if is_media_only:
                        i = j
                        matched = True
                        break
                    
                    # Clean WhatsApp media omitted text and any whitespace (case insensitive)
                    # Remove all variations of media omitted messages
                    media_patterns_to_remove = [
                        "<Médias omis>", "<médias omis>", "<MÉDIAS OMIS>",
                        "<Media omitted>", "<media omitted>", "<MEDIA OMITTED>",
                        "<Omitted>", "<omitted>", "<OMITTED>"
                    ]
                    for pattern in media_patterns_to_remove:
                        full_content = full_content.replace(pattern, "")
                    full_content = full_content.strip()
                    
                    # Filter out WhatsApp system messages using simple string matching
                    system_message_keywords = [
                        "code de sécurité",
                        "security code", 
                        "código de segurança",
                        "messages et les appels sont chiffrés",
                        "messages and calls are end-to-end encrypted",
                        "chiffrés de bout en bout",
                        "encrypted end-to-end",
                        "seules les personnes prenant part",
                        "only people taking part",
                        "en savoir plus",
                        "learn more",
                        "partager. en savoir plus",
                        "share. learn more"
                    ]
                    
                    is_system_message = False
                    full_content_lower = full_content.lower()
                    for keyword in system_message_keywords:
                        if keyword in full_content_lower:
                            is_system_message = True
                            break
                    
                    # Only add message if content is not empty after cleaning and not a system message
                    if full_content and not is_system_message:
                        messages.append({
                            'timestamp': timestamp,
                            'speaker': speaker.strip(),
                            'content': full_content
                        })
                    
                    i = j
                    matched = True
                    break
            
            if not matched:
                i += 1
        
        # Final filtering pass to remove any remaining unwanted messages
        filtered_messages = []
        for message in messages:
            content = message.get('content', '').strip()
            content_lower = content.lower()
            
            # Skip empty messages
            if not content:
                continue
                
            # Skip messages that are only media indicators
            if content_lower in ['médias omis', 'media omitted', 'omitted']:
                continue
                
            # Skip messages that are primarily system messages
            system_indicators = [
                'les messages et les appels sont chiffrés',
                'messages and calls are end-to-end encrypted',
                'seules les personnes prenant part',
                'only people taking part',
                'en savoir plus',
                'learn more'
            ]
            
            is_system = False
            for indicator in system_indicators:
                if indicator in content_lower:
                    is_system = True
                    break
            
            if not is_system:
                filtered_messages.append(message)
        
        return filtered_messages

    def _is_topic_related_message(self, message_lower: str, entity_lower: str, desc_words: set) -> bool:
        """Check if a message is related to the topic through contextual clues.
        
        Args:
            message_lower: Lowercase message content
            entity_lower: Lowercase entity name
            desc_words: Set of description keywords
            
        Returns:
            True if message appears related to the topic
        """
        # Skip very short messages or pure reactions
        if len(message_lower.split()) < 3:
            return False
        
        # Skip common chat noise
        noise_patterns = [
            r'^\s*(ok|okay|yes|no|lol|haha|👍|😂)\s*$',
            r'^\s*\w{1,3}\s*$',  # Very short responses
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, message_lower):
                return False
        
        # Look for topic-related context indicators
        if desc_words:
            message_words = set(message_lower.split())
            # Even one keyword match might be relevant in conversation context
            if len(desc_words.intersection(message_words)) >= 1:
                return True
        
        return False

    def _clean_raw_content(self, content: str) -> str:
        """Clean raw content to remove WhatsApp system messages and media indicators.
        
        Args:
            content: Raw content string
            
        Returns:
            Cleaned content string
        """
        if not content:
            return content
            
        # Remove WhatsApp system messages using simple string operations
        system_messages_to_remove = [
            "Les messages et les appels sont chiffrés de bout en bout. Seules les personnes prenant part à cette discussion peuvent les lire, les écouter ou les partager. En savoir plus.",
            "Messages and calls are end-to-end encrypted. Only people taking part in this conversation can read, listen to or share them. Learn more.",
            "Les messages et les appels sont chiffrés de bout en bout",
            "Messages and calls are end-to-end encrypted",
            "Seules les personnes prenant part à cette discussion peuvent les lire",
            "Only people taking part in this conversation can read"
        ]
        
        cleaned_content = content
        for system_msg in system_messages_to_remove:
            cleaned_content = cleaned_content.replace(system_msg, "")
        
        # Remove media indicators
        media_patterns = [
            "<Médias omis>", "<médias omis>", "<MÉDIAS OMIS>",
            "<Media omitted>", "<media omitted>", "<MEDIA OMITTED>",
            "<Omitted>", "<omitted>", "<OMITTED>"
        ]
        
        for pattern in media_patterns:
            cleaned_content = cleaned_content.replace(pattern, "")
        
        # Remove extra whitespace and empty lines
        lines = [line.strip() for line in cleaned_content.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        return '\n'.join(lines)

    def _format_chat_segment(self, messages: list, entity_name: str) -> str:
        """Format a segment of chat messages into readable text.
        
        Args:
            messages: List of message dictionaries
            entity_name: Entity name for context
            
        Returns:
            Formatted chat segment text
        """
        if not messages:
            return ""
        
        formatted_lines = []
        formatted_lines.append(f"[Chat segment related to {entity_name}]")
        
        for msg in messages:
            timestamp = msg.get('timestamp', '')
            speaker = msg.get('speaker', 'Unknown')
            content = msg.get('content', '')
            
            if timestamp:
                line = f"{timestamp} - {speaker}: {content}"
            else:
                line = f"{speaker}: {content}"
            
            formatted_lines.append(line)
        
        return "\n".join(formatted_lines)

    def _filter_relevant_notes(self, notes: list, entity_name: str, entity_description: str = "", 
                              relevance_threshold: float = 0.3, max_notes: int = None) -> tuple:
        """Extract relevant segments from chat notes while preserving conversational context.
        
        Args:
            notes: List of note dictionaries (chat extracts)
            entity_name: Name of the entity
            entity_description: Description of the entity
            relevance_threshold: Minimum relevance threshold (unused for segment extraction)
            max_notes: Maximum number of processed notes to keep
            
        Returns:
            Tuple of (processed_notes, stats_dict)
        """
        if not notes:
            return notes, {"original_count": 0, "filtered_count": 0, "segments_extracted": 0}
        
        processed_notes = []
        total_segments_extracted = 0
        
        for note in notes:
            # Extract note content
            if "processed" in note:
                content = note["processed"]
            else:
                title = note.get("title", "")
                content_body = note.get("content", "")
                insight = note.get("insight", "")
                
                # Apply basic filtering to raw content before processing
                if content_body:
                    content_body = self._clean_raw_content(content_body)
                
                content = f"Title: {title}\nContent: {content_body}\nAI Insight: {insight}".strip()
            
            # Check if this looks like chat data
            if self._is_chat_content(content):
                # Extract relevant chat segments
                segments = self._extract_relevant_chat_segments(
                    content, entity_name, entity_description, context_window=2
                )
                
                if segments:
                    # Create processed note with relevant segments
                    processed_note = note.copy()
                    
                    # Combine segments into processed content
                    segments_text = "\n\n".join(segments)
                    
                    if "processed" in processed_note:
                        processed_note["processed"] = segments_text
                    else:
                        processed_note["title"] = f"{title} (Relevant segments)"
                        processed_note["content"] = segments_text
                        processed_note["insight"] = f"Extracted {len(segments)} relevant chat segments related to {entity_name}"
                    
                    processed_notes.append(processed_note)
                    total_segments_extracted += len(segments)
                    
                    logger.info(f"Extracted {len(segments)} relevant segments from chat note for entity '{entity_name}'")
            else:
                # For non-chat content, apply basic relevance filtering
                relevance_score = self._calculate_basic_relevance(content, entity_name, entity_description)
                if relevance_score >= relevance_threshold:
                    processed_notes.append(note.copy())
                elif len(processed_notes) == 0 and relevance_score > 0:
                    # If no notes pass the threshold but this note has some relevance, keep it
                    processed_notes.append(note.copy())
                    logger.info(f"Kept low-relevance note for entity '{entity_name}' (score: {relevance_score:.2f})")
        
        # Limit number of notes if specified
        if max_notes and len(processed_notes) > max_notes:
            processed_notes = processed_notes[:max_notes]
        
        # Safety mechanism: if no notes were kept and we had original notes, keep the first few
        if not processed_notes and notes:
            logger.warning(f"No notes passed relevance filter for entity '{entity_name}', keeping first 2 notes as fallback")
            processed_notes = notes[:2]  # Keep first 2 notes as fallback
        
        # Generate stats
        stats = {
            "original_count": len(notes),
            "filtered_count": len(processed_notes),
            "removed_count": len(notes) - len(processed_notes),
            "segments_extracted": total_segments_extracted,
        }
        
        if total_segments_extracted > 0:
            logger.debug(f"Entity '{entity_name}': Extracted {total_segments_extracted} relevant chat segments "
                        f"from {stats['filtered_count']}/{stats['original_count']} notes")
        elif stats['removed_count'] > 0:
            logger.debug(f"Entity '{entity_name}': Filtered {stats['removed_count']} notes, kept {stats['filtered_count']}")
        
        return processed_notes, stats

    def _is_chat_content(self, content: str) -> bool:
        """Determine if content appears to be chat/conversation data.
        
        Args:
            content: Text content to analyze
            
        Returns:
            True if content appears to be chat conversation
        """
        # Use simple string operations for better memory efficiency
        content_lower = content.lower()
        
        # Check for common chat indicators using string containment (much faster)
        chat_keywords = ["<médias omis>", "<media omitted>", "<omitted>"]
        media_indicator_count = sum(1 for keyword in chat_keywords if keyword in content_lower)
        
        # Count lines that look like timestamps and messages using simple parsing
        lines = content.split('\n')
        whatsapp_message_lines = 0
        speaker_lines = 0
        timestamp_lines = 0
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Simple timestamp detection (faster than regex)
            if ('/' in line_stripped or '-' in line_stripped) and ':' in line_stripped:
                # Check if it looks like a WhatsApp timestamp format
                if len([c for c in line_stripped[:20] if c.isdigit()]) >= 6:  # At least 6 digits in first 20 chars
                    timestamp_lines += 1
                    if ' - ' in line_stripped and ':' in line_stripped[line_stripped.find(' - '):]:
                        whatsapp_message_lines += 1
            
            # Simple speaker detection
            if ':' in line_stripped and not line_stripped.startswith('http'):
                colon_pos = line_stripped.find(':')
                if colon_pos > 0 and colon_pos < 50:  # Reasonable speaker name length
                    speaker_lines += 1
        
        # Consider it chat if we have:
        # 1. Multiple media indicators, OR
        # 2. Multiple WhatsApp-style message lines, OR  
        # 3. Multiple speaker lines, OR
        # 4. Multiple timestamp lines
        return (media_indicator_count >= 2 or 
                whatsapp_message_lines >= 3 or 
                speaker_lines >= 5 or
                timestamp_lines >= 3)

    def _calculate_basic_relevance(self, content: str, entity_name: str, entity_description: str) -> float:
        """Calculate basic relevance for non-chat content.
        
        Args:
            content: Content to evaluate
            entity_name: Entity name
            entity_description: Entity description
            
        Returns:
            Basic relevance score (0-1)
        """
        content_lower = content.lower()
        entity_lower = entity_name.lower()
        
        # Direct entity mentions (full name)
        entity_mentions = content_lower.count(entity_lower)
        direct_score = min(entity_mentions * 0.3, 0.6)
        
        # Partial entity name matches (words from entity name)
        entity_words = entity_lower.split()
        partial_score = 0.0
        if len(entity_words) > 1:  # Multi-word entity names
            content_words = set(content_lower.split())
            entity_word_set = set(entity_words)
            overlap = len(entity_word_set.intersection(content_words))
            if overlap > 0:
                partial_score = min((overlap / len(entity_words)) * 0.4, 0.4)
        
        # Description keywords
        desc_score = 0.0
        if entity_description:
            desc_words = set(entity_description.lower().split())
            content_words = set(content_lower.split())
            if desc_words:
                overlap = len(desc_words.intersection(content_words))
                desc_score = min((overlap / len(desc_words)) * 0.3, 0.3)
        
        # Base score for any content (minimal relevance)
        base_score = 0.1 if len(content.strip()) > 20 else 0.0
        
        total_score = min(direct_score + partial_score + desc_score + base_score, 1.0)
        
        # Debug logging for very low scores
        if total_score < 0.2:
            logger.debug(f"Low relevance for entity '{entity_name}': {total_score:.3f} (direct:{direct_score:.2f}, partial:{partial_score:.2f}, desc:{desc_score:.2f})")
        
        return total_score

    def _enhance_cluster_relevance(self, cluster: dict, relevance_threshold: float = 0.3) -> dict:
        """Enhance cluster by filtering irrelevant notes and improving quality.
        
        Args:
            cluster: The cluster containing entity and notes
            relevance_threshold: Minimum relevance score to keep a note
            
        Returns:
            Enhanced cluster with filtered notes
        """
        if not cluster.get("note"):
            return cluster
        
        entity_name = cluster.get("entity_name", "")
        entity_description = cluster.get("entity_description", "")
        original_notes = cluster.get("note", [])
        
        # Filter for relevance
        filtered_notes, filter_stats = self._filter_relevant_notes(
            original_notes, entity_name, entity_description, relevance_threshold
        )
        
        # Create enhanced cluster
        enhanced_cluster = cluster.copy()
        enhanced_cluster["note"] = filtered_notes
        enhanced_cluster["relevance_filter_stats"] = filter_stats
        
        return enhanced_cluster

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
            
            # Apply cleaning to raw content
            if content_body:
                content_body = self._clean_raw_content(content_body)
            
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
            
            # Apply cleaning to raw content
            if content_body:
                content_body = self._clean_raw_content(content_body)
            
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
                title = note_dict.get('title', '')
                content_body = note_dict.get('content', '')
                insight = note_dict.get('insight', '')
                
                # Apply cleaning to raw content
                if content_body:
                    content_body = self._clean_raw_content(content_body)
                
                content = f"Title: {title}\nContent: {content_body}\nAI Insight: {insight}"
            
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
                        title = note_chunk.get('title', '')
                        content_body = note_chunk.get('content', '')
                        insight = note_chunk.get('insight', '')
                        
                        # Apply cleaning to raw content 
                        if content_body:
                            content_body = self._clean_raw_content(content_body)
                        
                        chunk_content = f"Title: {title}\nContent: {content_body}\nAI Insight: {insight}"
                    
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
        # CRITICAL: Check memory at start and clean if necessary
        initial_memory = self._get_current_memory_usage()
        if initial_memory > 15000:  # 15GB - emergency cleanup
            logger.warning(f"CRITICAL: Initial memory usage {initial_memory:.1f} MB - performing emergency cleanup")           
            self._force_garbage_collection()
        
        # Check preprocess cache
        if self.enable_cache:
            input_hash = self._get_input_hash(entities_path, note_list, config_path, graph_path, user_name)
            cache_path = self.preprocess_cache_dir / f"{input_hash}.pkl"
            
            # Check if cache file exists and isn't corrupted
            if cache_path.exists():
                try:
                    # Check file size first - if too large, skip                                     
                    cached_result = self._load_cache(cache_path)
                    if cached_result is not None:
                        logger.info(f"Loaded preprocess result from cache: {cache_path}")
                        # Make a copy to return and clear the original from memory
                        result_copy = cached_result
                        del cached_result
                        self._force_garbage_collection()
                        return result_copy
                    else:
                        logger.warning(f"Cache file corrupted, deleting: {cache_path}")
                        cache_path.unlink()  # Delete corrupted cache
                except Exception as e:
                    logger.warning(f"Error loading cache {cache_path}: {e} - deleting corrupted file")
                    try:
                        cache_path.unlink()
                    except:
                        pass
        
        logger.info("Running preprocess (not cached)")
        self._log_memory_usage("before preprocessing")
        
        # Load data with memory monitoring
        try:
            entity_df = pd.read_parquet(graph_path)
            entity2type = {
                item["title"]: item["type"] for item in entity_df.to_dict(orient="records")
            }
            # Clear dataframe immediately to free memory
            del entity_df
            self._force_garbage_collection()
            self._log_memory_usage("after loading graph data")
        except Exception as e:
            logger.error(f"Failed to load graph data: {e}")
            return None, None, None

        # read entity2desc with memory management
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                entities = json.load(f)
                entity2desc = {
                    item["entity_name"]: {
                        key: value for key, value in item.items() if key != "entity_name"
                    }
                    for item in entities
                }
                # Clear entities list immediately to free memory
                del entities
                self._force_garbage_collection()
                self._log_memory_usage("after loading entity descriptions")
        except Exception as e:
            logger.error(f"Failed to load entities: {e}")
            return None, None, None
        
        # read note data with memory management
        id2note = {}
        processed_notes = 0
        for item in note_list:
            try:
                item_json = item.to_json()
                note_id = item.id
                note_data = {
                    key: value for key, value in item_json.items() if key != "id"
                }
                id2note[note_id] = note_data
                processed_notes += 1
                
                # Debug: Log first few note IDs and their types
                if processed_notes <= 3:
                    logger.debug(f"Note {processed_notes}: ID = '{note_id}' (type: {type(note_id)}), keys = {list(note_data.keys())}")
                
                # Clear item_json immediately
                del item_json
            except Exception as e:
                logger.warning(f"Failed to process note item {getattr(item, 'id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {processed_notes} notes into id2note dictionary (total keys: {len(id2note)})")
        if len(id2note) > 0:
            sample_keys = list(id2note.keys())[:3]
            logger.debug(f"Sample id2note keys: {sample_keys}")
        
        self._log_memory_usage("after loading note data")
        
        # Process in smaller batches to prevent memory buildup
        logger.info(f"Processing {len(entity2desc)} entities for note attachment")
        
        # Debug: Check entity structure
        if len(entity2desc) > 0:
            sample_entity_name = list(entity2desc.keys())[0]
            sample_entity_data = entity2desc[sample_entity_name]
            logger.debug(f"Sample entity '{sample_entity_name}': keys = {list(sample_entity_data.keys())}")
            if "doc_id" in sample_entity_data:
                doc_ids = sample_entity_data["doc_id"]
                logger.debug(f"Sample doc_ids: {doc_ids[:3] if len(doc_ids) > 3 else doc_ids} (type: {type(doc_ids)})")
                if len(doc_ids) > 0:
                    logger.debug(f"First doc_id: '{doc_ids[0]}' (type: {type(doc_ids[0])})")

        # Process entities in batches to manage memory
        entity_items = list(entity2desc.items())
        batch_size = 50  # Process 50 entities at a time
        notes_found = 0
        
        for batch_start in range(0, len(entity_items), batch_size):
            batch_end = min(batch_start + batch_size, len(entity_items))
            logger.info(f"Processing entity batch {batch_start//batch_size + 1}/{(len(entity_items) + batch_size - 1)//batch_size}")
            
            for i in range(batch_start, batch_end):
                entity, entity_info = entity_items[i]
                doc_ids = entity_info["doc_id"]
                tmp = []
                
                # Debug: Check what doc_ids look like and if they exist in id2note
                if batch_start == 0 and i < 5:  # Debug first few entities only
                    logger.info(f"DEBUGGING Entity '{entity}': doc_ids = {doc_ids[:5] if len(doc_ids) > 5 else doc_ids}")
                    sample_id = doc_ids[0] if doc_ids else None
                    if sample_id:
                        exists_direct = sample_id in id2note
                        exists_str = str(sample_id) in id2note
                        logger.info(f"DEBUGGING Sample doc_id '{sample_id}' (type: {type(sample_id)}) exists: direct={exists_direct}, str_version={exists_str}")
                        if len(id2note) > 0:
                            sample_keys = list(id2note.keys())[:3]
                            logger.info(f"DEBUGGING First 3 id2note keys: {[(k, type(k)) for k in sample_keys]}")
                        
                        # Show actual note data retrieval
                        if sample_id in id2note:
                            note_data = id2note[sample_id]
                            logger.info(f"DEBUGGING Found note data for {sample_id}: keys = {list(note_data.keys()) if note_data else 'None'}")
                        elif str(sample_id) in id2note:
                            note_data = id2note[str(sample_id)]
                            logger.info(f"DEBUGGING Found note data for str({sample_id}): keys = {list(note_data.keys()) if note_data else 'None'}")
                
                for doc_id in doc_ids:
                    note_data = None
                    
                    # Try both the original doc_id and its string version
                    if doc_id in id2note:
                        note_data = id2note[doc_id]
                    elif str(doc_id) in id2note:
                        note_data = id2note[str(doc_id)]
                    elif isinstance(doc_id, str) and doc_id.isdigit():
                        # Try converting string to int if it's numeric
                        int_doc_id = int(doc_id)
                        if int_doc_id in id2note:
                            note_data = id2note[int_doc_id]
                    
                    if note_data is not None:
                        tmp.append(note_data)
                
                entity2desc[entity]["note"] = tmp
                if tmp:  # Count entities with notes
                    notes_found += len(tmp)
                
                # Debug: Log attachment results for first few entities
                if batch_start == 0 and i < 5:
                    logger.info(f"DEBUGGING Entity '{entity}': Attached {len(tmp)} notes from {len(doc_ids)} doc_ids")
            
            # Force garbage collection every batch
            if (batch_start // batch_size + 1) % 5 == 0:  # Every 5 batches
                self._force_garbage_collection()
                self._log_memory_usage(f"after processing batch {batch_start//batch_size + 1}")
        
        logger.info(f"Found {notes_found} notes across {len(entity_items)} entities")
        
        # Debug: Count entities with notes
        entities_with_notes = sum(1 for entity_info in entity2desc.values() if len(entity_info.get("note", [])) > 0)
        logger.info(f"Entities with notes after attachment: {entities_with_notes}/{len(entity2desc)}")
        
        # Clear temporary variables
        del entity_items, id2note

        entity2desc.pop(f"{user_name}", None)
        entity2desc.pop(f"{user_name.upper()}", None)

        # exclude keys with time format
        time_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        filtered_data = {
            k: v for k, v in entity2desc.items() if not re.match(time_pattern, k)
        }
        entity2desc = filtered_data

        # clean note level data with cached deduplication
        total_entities = len(entity2desc)
        processed_entities = 0
        
        for entity, entity_info in entity2desc.copy().items():
            logger.info(f"Processing entity: {entity}")
            clusters = entity_info["note"]
            
            # Memory check before processing large entities
            current_memory = self._get_current_memory_usage()
            cluster_count = len(clusters)
            if cluster_count > 1000 or current_memory > 20000:  # 20GB threshold
                logger.warning(f"Processing large entity '{entity}' with {cluster_count} clusters, current memory: {current_memory:.1f} MB")
                if current_memory > 25000:  # 25GB critical threshold
                    logger.error(f"CRITICAL: Memory usage too high before processing '{entity}': {current_memory:.1f} MB")
                    self.emergency_memory_cleanup()
                    # Check if cleanup helped
                    # post_cleanup_memory = self._get_current_memory_usage()
                    # if post_cleanup_memory > 12000:  # Still too high after cleanup
                    #     logger.error(f"Skipping entity '{entity}' due to memory constraints: {post_cleanup_memory:.1f} MB")
                    #     processed_entities += 1
                    #     continue
            
            # Check dedup cache for this entity
            dedup_cache_path = None
            cached_dedup = None
            if self.enable_cache:
                logger.info(f"Looking for deduplication cache for entity: {entity}")
                entity_hash = self._generate_clusters_hash(entity, clusters)
                logger.info(f"Generated hash for entity '{entity}': {entity_hash}")
                dedup_cache_path = self.dedup_cache_dir / f"{entity_hash}.pkl"
                cached_dedup = self._load_cache(dedup_cache_path)
                if cached_dedup is not None:
                    logger.info(f"Loaded dedup result for entity {entity} from cache")
                    entity2desc[entity]["note"] = cached_dedup
                    # Clear the cached data from memory immediately
                    del cached_dedup
                    processed_entities += 1
                    
                    # Force garbage collection every 10 entities
                    if processed_entities % 10 == 0:
                        self._force_garbage_collection()
                        logger.debug(f"Processed {processed_entities}/{total_entities} entities, freed memory")
                    continue
            
            logger.info(f"{entity}: Starting deduplication of {len(clusters)} notes")
            # Run deduplication if not cached
            original_count = len(clusters)
            
            # Debug: Log if clusters is empty before deduplication
            if original_count == 0:
                logger.debug(f"Entity '{entity}' has no notes before deduplication")
                entity2desc[entity]["note"] = []
                processed_entities += 1
                continue
            
            unique_dicts, cnt = dedup_by_similarity(clusters, similarity_threshold=0.9)
            logger.info(f"Deduplication for entity '{entity}': {original_count} -> {len(unique_dicts)} notes (removed {cnt} duplicates)")
            
            # Clear the original clusters from memory
            del clusters
            
            # Apply chat segment extraction and relevance filtering if enabled
            if self.enable_relevance_filtering:
                entity_description = entity2desc[entity].get("entity_description", "")
                processed_notes, stats = self._filter_relevant_notes(
                    unique_dicts, entity, entity_description, self.relevance_threshold
                )
                
                # Debug: Log relevance filtering results
                if len(processed_notes) == 0 and len(unique_dicts) > 0:
                    logger.warning(f"Relevance filtering removed ALL {len(unique_dicts)} notes for entity '{entity}'")
                    # Let's keep at least one note to prevent total data loss
                    processed_notes = unique_dicts[:1]
                    logger.info(f"Keeping 1 note for entity '{entity}' to prevent data loss")
                
                if len(processed_notes) != len(unique_dicts):
                    logger.info(f"Relevance filtering for entity '{entity}': {len(unique_dicts)} -> {len(processed_notes)} "
                               f"(extracted {stats.get('segments_extracted', 0)} chat segments)")
                
                entity2desc[entity]["note"] = processed_notes
                # Save the processed notes to cache (not empty list)
                cache_data = processed_notes
                # Clear intermediate data
                del unique_dicts
                del processed_notes
            else:
                entity2desc[entity]["note"] = unique_dicts
                cache_data = unique_dicts
                del unique_dicts
            
            # Save dedup result to cache (the actual processed data, not potentially empty filtered results)
            if self.enable_cache and dedup_cache_path and len(cache_data) > 0:
                self._save_cache(dedup_cache_path, cache_data)
                logger.info(f"Cached {len(cache_data)} notes for entity '{entity}'")
            
            processed_entities += 1
            
            # Memory monitoring after each entity
            current_memory = self._get_current_memory_usage()
            logger.info(f"Processed entity {processed_entities}/{total_entities}: '{entity}' ({len(entity2desc[entity]['note'])} final notes) - Memory: {current_memory:.1f} MB")
            
            if current_memory > 12000:  # 12GB warning
                logger.warning(f"High memory usage: {current_memory:.1f} MB - forcing garbage collection")
                self._force_garbage_collection()
                new_memory = self._get_current_memory_usage()
                logger.info(f"Memory after cleanup: {new_memory:.1f} MB")
            
            # Force garbage collection every 10 entities
            if processed_entities % 10 == 0:
                self._force_garbage_collection()
                logger.debug(f"Processed {processed_entities}/{total_entities} entities, freed memory")
        
        # Final garbage collection
        self._force_garbage_collection()
        self._log_memory_usage("after preprocessing")
        logger.info(f"Completed preprocessing {total_entities} entities with memory management")

        # read config file
        with open(config_path, "r", encoding="utf-8") as f:
            QA_config = json.load(f)

        result = (entity2desc, entity2type, QA_config)
        
        # Save preprocess result to cache
        if self.enable_cache:
            self._save_cache(cache_path, result)
            logger.info(f"Saved preprocess result to cache: {cache_path}")

        # Final cleanup and memory management
        self._force_garbage_collection()
        self._log_memory_usage("at end of preprocessing")
        
        # Check final memory usage
        final_memory = self._get_current_memory_usage()
        if final_memory > 25000:  # 25GB
            logger.warning(f"High memory usage at end of preprocessing: {final_memory:.1f} MB")            

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
            if "processed" in entity_dict:
                content = entity_dict["processed"]
            else:
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
        # EMERGENCY: Check memory at start of generate_data
        initial_memory = self._get_current_memory_usage()
        logger.info(f"Starting generate_data with memory usage: {initial_memory:.1f} MB")
        
        if initial_memory > 15000:  # 15GB - emergency intervention
            logger.error(f"CRITICAL: Memory usage too high at start: {initial_memory:.1f} MB")
            self.emergency_memory_cleanup()
            
            # Check memory after cleanup
            post_cleanup_memory = self._get_current_memory_usage()
            logger.info(f"Memory usage after emergency cleanup: {post_cleanup_memory:.1f} MB")
            
            if post_cleanup_memory > 12000:  # Still too high
                raise RuntimeError(f"Cannot proceed: Memory usage still too high after cleanup: {post_cleanup_memory:.1f} MB")
        
        # Generate job ID for this run (consistent across restarts for same inputs)
        job_id = self._generate_hash(entities_path, config_path, graph_path, user_name, global_bio, 
                                   self.data_synthesis_mode)
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
            for iteration in range(len(sub_dict["note"]) // 10 + 1):
                tmp_dict = sub_dict.copy()
                # Generate deterministic seed for reproducible sampling
                seed = self._generate_deterministic_seed(job_id, sub_dict["entity_name"], "global_sampling", iteration)
                random.seed(seed)
                sampled_notes_and_ids = random.sample(
                    notes_and_ids, min(10, len(notes_and_ids))
                )
                logger.debug(f"Global sampling for '{sub_dict['entity_name']}' iteration {iteration}: seed={seed}, sampled {len(sampled_notes_and_ids)} notes")
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
                cached_large = self._load_cache(large_cache_path)
                if cached_large is not None:
                    data_large = cached_large
                    del cached_large  # Clear from memory immediately
                    self._force_garbage_collection()
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
        
        # Clear exploded_clusters from memory as we're done with them
        del exploded_clusters
        self._force_garbage_collection()

        # Process mini clusters with caching
        data_mini = []
        if len(mini_clusters) > 0:
            mini_cache_path = job_cache_dir / "mini_clusters_done.pkl" if self.enable_cache else None
            if resume_from_cache and mini_cache_path and mini_cache_path.exists():
                cached_mini = self._load_cache(mini_cache_path)
                if cached_mini is not None:
                    data_mini = cached_mini
                    del cached_mini  # Clear from memory immediately
                    self._force_garbage_collection()
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
        
        # Clear mini_clusters from memory
        del mini_clusters
        self._force_garbage_collection()

        # Process tiny clusters with caching  
        data_tiny = []
        if len(filtered_tiny_clusters) > 0:
            tiny_cache_path = job_cache_dir / "tiny_clusters_done.pkl" if self.enable_cache else None
            if resume_from_cache and tiny_cache_path and tiny_cache_path.exists():
                cached_tiny = self._load_cache(tiny_cache_path)
                if cached_tiny is not None:
                    data_tiny = cached_tiny
                    del cached_tiny  # Clear from memory immediately
                    self._force_garbage_collection()
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
                del q_dict_copy  # Clean up the copy
        else:
            logger.info("Single entity cluster number is 0")
        
        # Clear filtered_tiny_clusters from memory
        del filtered_tiny_clusters
        self._force_garbage_collection()

        combined_list = data_large + data_mini + data_tiny
        # calculate total entries
        total_entries = len(combined_list)
        logger.info(f"Total entries: {total_entries}")
        # store data
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_list, f, ensure_ascii=False, indent=4)

        logger.info(f"Data has been stored to {output_path}")
        
        # Log deterministic caching summary
        logger.info(f"Diversity data generation completed with deterministic seeding:")
        logger.info(f"  - Job ID: {job_id}")
        logger.info(f"  - Total entries generated: {total_entries}")
        logger.info(f"  - Random sampling and Q&A generation are reproducible across service restarts")
        logger.info(f"  - Cache keys include job_id for consistency")
        
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
        for cluster_idx, item in enumerate(processed_clusters):
            # add elements multiple times based on aug_para
            explode_clusters.extend([item] * aug_para)
            # Generate deterministic seed for reproducible question type selection
            seed = self._generate_deterministic_seed(job_id, item.get("entity_name", ""), item.get("chunk_info", ""), "question_types", cluster_idx)
            random.seed(seed)
            # randomly select different types based on weights
            weights = [v["weight"] for v in q_dict.values()]
            random_types = random.choices(list(q_dict.keys()), weights, k=aug_para)
            explode_questions_types.extend(random_types)
            logger.debug(f"Question type selection for cluster {cluster_idx} '{item.get('entity_name', 'unknown')}': seed={seed}, types={random_types}")

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
            # Create deterministic cache key including job_id for consistency
            # Sort notes by a consistent field to ensure deterministic hashing
            notes = cluster.get('note', [])
            sorted_notes = sorted(notes, key=lambda x: str(x.get('content', '') + x.get('title', '')))
            cache_key = self._generate_hash(
                job_id,  # Include job_id for cache consistency
                cluster.get('entity_name', ''), 
                cluster.get('chunk_info', ''),
                str(sorted_notes),  # Use sorted notes for deterministic hashing
                question_type,
                system_prompt,
                user_input,
                language_desc
            )
            cache_path = self.question_cache_dir / f"{cache_key}.pkl"
            cached_questions = self._load_cache(cache_path)
            if cached_questions is not None:
                logger.info(f"Q_generate cache HIT for '{cluster.get('entity_name', 'unknown')}' (job: {job_id[:8]}...)")
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
            logger.info(f"Q_generate cached {len(questions)} questions for '{cluster.get('entity_name', 'unknown')}' (job: {job_id[:8]}...)")

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
            # Create deterministic cache key including job_id for consistency
            # Sort notes by a consistent field to ensure deterministic hashing
            notes = cluster.get('note', [])
            sorted_notes = sorted(notes, key=lambda x: str(x.get('content', '') + x.get('title', '')))
            cache_key = self._generate_hash(
                job_id,  # Include job_id for cache consistency
                cluster.get('entity_name', ''), 
                cluster.get('chunk_info', ''),
                str(sorted_notes),  # Use sorted notes for deterministic hashing
                question,
                question_type,
                system_prompt,
                user_input,
                language_desc
            )
            cache_path = self.answer_cache_dir / f"{cache_key}.pkl"
            cached_answer = self._load_cache(cache_path)
            if cached_answer is not None:
                logger.info(f"A_generate cache HIT for '{cluster.get('entity_name', 'unknown')}' (job: {job_id[:8]}...)")
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
            logger.info(f"A_generate cached answer for '{cluster.get('entity_name', 'unknown')}' (job: {job_id[:8]}...)")
            
        return result