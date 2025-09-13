from typing import Dict, List, Any, Optional
import json
import re
import traceback

from openai import OpenAI
import numpy as np
import tiktoken

from lpm_kernel.L1.bio import (
    Cluster,
    Note,
    ShadeInfo,
    ShadeTimeline,
    ShadeMergeInfo,
    ShadeMergeResponse,
)
from lpm_kernel.L1.prompt import (
    PREFER_LANGUAGE_SYSTEM_PROMPT,
    SHADE_INITIAL_PROMPT,
    PERSON_PERSPECTIVE_SHIFT_V2_PROMPT,
    SHADE_MERGE_PROMPT,
    SHADE_IMPROVE_PROMPT,
    SHADE_MERGE_DEFAULT_SYSTEM_PROMPT,
)
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config

from lpm_kernel.api.common.script_executor import ScriptExecutor

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

class ShadeGenerator:
    def __init__(self):
        self.preferred_language = "en"
        self.model_params = {
            "temperature": 0,
            "max_tokens": 3000,
            "top_p": 0,
            "frequency_penalty": 0,
            "seed": 42,
            "presence_penalty": 0,
            "timeout": 45,
        }
        # Initialize tokenizer for token management
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Could not initialize tiktoken, falling back to character counting")
            self.tokenizer = None
        
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        if self.user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.client = OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            self.model_name = self.user_llm_config.chat_model_name
        self._top_p_adjusted = False  # Flag to track if top_p has been adjusted

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to character count estimation (roughly 4 chars per token)
            return len(text) // 4

    def _truncate_note_content(self, note: Note, max_tokens: int) -> str:
        """Intelligently truncate a note's content to fit within token limits.
        
        Args:
            note: The note to truncate
            max_tokens: Maximum tokens allowed for this note
            
        Returns:
            Truncated note content as string
        """
        full_content = note.to_str()
        
        if self._count_tokens(full_content) <= max_tokens:
            return full_content
            
        # Try to preserve structure by keeping the beginning and trimming intelligently
        lines = full_content.split('\n')
        result_lines = []
        current_tokens = 0
        
        # Always include the first few lines (metadata/title)
        for i, line in enumerate(lines[:3]):  # Keep first 3 lines
            line_tokens = self._count_tokens(line + '\n')
            if current_tokens + line_tokens > max_tokens:
                break
            result_lines.append(line)
            current_tokens += line_tokens
        
        # Add remaining lines until we hit the token limit
        remaining_budget = max_tokens - current_tokens
        if remaining_budget > 100:  # Only continue if we have reasonable space left
            remaining_lines = lines[len(result_lines):]
            
            for line in remaining_lines:
                line_tokens = self._count_tokens(line + '\n')
                if current_tokens + line_tokens > max_tokens - 50:  # Leave buffer
                    break
                result_lines.append(line)
                current_tokens += line_tokens
                
        # Add truncation indicator if we cut content
        if len(result_lines) < len(lines):
            result_lines.append(f"\n[CONTENT TRUNCATED - Original had {len(lines)} lines, showing {len(result_lines)}]")
            
        return '\n'.join(result_lines)

    def _manage_memory_tokens(self, new_memory_list: List[Note], max_total_tokens: int = 100000) -> str:
        """Manage token limits across multiple notes for prompt generation.
        
        Args:
            new_memory_list: List of notes to process
            max_total_tokens: Maximum total tokens for all notes combined
            
        Returns:
            Combined prompt string with token management applied
        """
        if not new_memory_list:
            return ""
            
        # Reserve space for system prompt and output (estimated)
        available_tokens = max_total_tokens - 5000  # Conservative buffer
        
        # Calculate tokens per note (distributed evenly with minimum)
        min_tokens_per_note = 500  # Minimum meaningful content per note
        max_tokens_per_note = available_tokens // len(new_memory_list)
        
        # Ensure each note gets at least minimum tokens, but cap at reasonable max
        tokens_per_note = max(min_tokens_per_note, min(max_tokens_per_note, 8000))
        
        logger.info(f"Managing {len(new_memory_list)} notes with {tokens_per_note} tokens each (max total: {max_total_tokens})")
        
        processed_notes = []
        actual_total_tokens = 0
        
        for i, note in enumerate(new_memory_list):
            truncated_content = self._truncate_note_content(note, tokens_per_note)
            note_tokens = self._count_tokens(truncated_content)
            actual_total_tokens += note_tokens
            processed_notes.append(truncated_content)
            
            logger.debug(f"Note {i}: {note_tokens} tokens")
            
        combined_prompt = "\n\n".join(processed_notes)
        logger.info(f"Final prompt: {actual_total_tokens} tokens ({len(combined_prompt)} characters)")
        
        return combined_prompt

    def _manage_shade_merge_tokens(self, shade_info_list: List[ShadeInfo], max_total_tokens: int = 80000) -> str:
        """Manage token limits when merging multiple shades.
        
        Args:
            shade_info_list: List of shade information to merge
            max_total_tokens: Maximum total tokens for all shades combined
            
        Returns:
            Combined prompt string with token management applied
        """
        if not shade_info_list:
            return ""
            
        # Reserve space for system prompt and output
        available_tokens = max_total_tokens - 3000
        
        # Calculate tokens per shade
        tokens_per_shade = available_tokens // len(shade_info_list)
        tokens_per_shade = max(1000, min(tokens_per_shade, 10000))  # Min 1k, max 10k per shade
        
        logger.info(f"Managing {len(shade_info_list)} shades with {tokens_per_shade} tokens each")
        
        processed_shades = []
        actual_total_tokens = 0
        
        for i, shade_info in enumerate(shade_info_list):
            full_shade_text = f"User Interest Domain {i} Analysis:\n{shade_info.to_str()}"
            
            # Truncate if necessary
            if self._count_tokens(full_shade_text) > tokens_per_shade:
                # Keep the header and truncate the content
                header = f"User Interest Domain {i} Analysis:\n"
                available_for_content = tokens_per_shade - self._count_tokens(header)
                
                shade_content = shade_info.to_str()
                content_lines = shade_content.split('\n')
                
                truncated_lines = []
                current_tokens = 0
                
                for line in content_lines:
                    line_tokens = self._count_tokens(line + '\n')
                    if current_tokens + line_tokens > available_for_content - 100:  # Leave buffer
                        truncated_lines.append("[CONTENT TRUNCATED]")
                        break
                    truncated_lines.append(line)
                    current_tokens += line_tokens
                
                truncated_content = header + '\n'.join(truncated_lines)
            else:
                truncated_content = full_shade_text
            
            shade_tokens = self._count_tokens(truncated_content)
            actual_total_tokens += shade_tokens
            processed_shades.append(truncated_content)
            
            logger.debug(f"Shade {i}: {shade_tokens} tokens")
        
        combined_prompt = "\n\n".join(processed_shades)
        logger.info(f"Final shade merge prompt: {actual_total_tokens} tokens ({len(combined_prompt)} characters)")
        
        return combined_prompt

    def _fix_top_p_param(self, error_message: str) -> bool:
        """Fixes the top_p parameter if an API error indicates it's invalid.
        
        Some LLM providers don't accept top_p=0 and require values in specific ranges.
        This function checks if the error is related to top_p and adjusts it to 0.001,
        which is close enough to 0 to maintain deterministic behavior while satisfying
        API requirements.
        
        Args:
            error_message: Error message from the API response.
            
        Returns:
            bool: True if top_p was adjusted, False otherwise.
        """
        if not self._top_p_adjusted and "top_p" in error_message.lower():
            logger.warning("Fixing top_p parameter from 0 to 0.001 to comply with model API requirements")
            self.model_params["top_p"] = 0.001
            self._top_p_adjusted = True
            return True
        return False

    def _call_llm_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Calls the LLM API with automatic retry for parameter adjustments.
        
        This function handles making API calls to the language model while
        implementing automatic parameter fixes when errors occur. If the API
        rejects the call due to invalid top_p parameter, it will adjust the
        parameter value and retry the call once.
        
        Args:
            messages: List of messages for the API call.
            **kwargs: Additional parameters to pass to the API call.
            
        Returns:
            API response object from the language model.
            
        Raises:
            Exception: If the API call fails after all retries or for unrelated errors.
        """
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.model_params,
                **kwargs
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API Error: {error_msg}")
            
            # Try to fix top_p parameter if needed
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400:
                if self._fix_top_p_param(error_msg):
                    logger.info("Retrying LLM API call with adjusted top_p parameter")
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **self.model_params,
                        **kwargs
                    )
            
            # Re-raise the exception
            raise

    def _build_message(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Builds the message structure for the LLM API.
        
        Args:
            system_prompt: The system prompt to guide the LLM behavior.
            user_prompt: The user prompt containing the actual query.
            
        Returns:
            A list of message dictionaries formatted for the LLM API.
        """
        raw_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.preferred_language:
            raw_message.append(
                {
                    "role": "system",
                    "content": PREFER_LANGUAGE_SYSTEM_PROMPT.format(
                        language=self.preferred_language
                    ),
                }
            )
        return raw_message


    def __add_second_view_info(self, shade_info: ShadeInfo) -> ShadeInfo:
        """Adds second-person perspective information to the shade info.
        
        Args:
            shade_info: The ShadeInfo object with third-person perspective.
            
        Returns:
            Updated ShadeInfo object with second-person perspective.
        """
        user_prompt = f"""Domain Name: {shade_info.name}
Domain Description: {shade_info.desc_third_view}
Domain Content: {shade_info.content_third_view}
Domain Timelines: 
{
    "-".join([f"{timeline.create_time}, {timeline.desc_third_view}, {timeline.ref_memory_id}" for timeline in shade_info.timelines if timeline.is_new])
}
"""
        shift_perspective_message = self._build_message(
            PERSON_PERSPECTIVE_SHIFT_V2_PROMPT, user_prompt
        )
        response = self._call_llm_with_retry(shift_perspective_message)
        content = response.choices[0].message.content
        shift_pattern = r"\{.*\}"
        shift_perspective_result = self.__parse_json_response(content, shift_pattern)
        
        # Check if result is None and provide default values to avoid TypeError
        if shift_perspective_result is None:
            logger.warning(f"Failed to parse perspective shift result, using default values: {content}")
            # Create a default mapping with expected parameters
            shift_perspective_result = {
                "domainDesc": f"You have knowledge and experience related to {shade_info.name}.",
                "domainContent": shade_info.content_third_view,
                "domainTimeline": []
            }
            
        # Now it's safe to pass shift_perspective_result as kwargs
        shade_info.add_second_view(**shift_perspective_result)
        return shade_info


    def __parse_json_response(
        self, content: str, pattern: str, default_res: dict = None
    ) -> Dict[str, Any]:
        """Parses JSON response from LLM output.
        
        Args:
            content: The raw text response from the LLM.
            pattern: Regex pattern to extract the JSON string.
            default_res: Default result to return if parsing fails.
            
        Returns:
            Parsed JSON dictionary or default_res if parsing fails.
        """
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            logger.error(f"No Json Found: {content}")
            return default_res
        try:
            json_res = json.loads(matches[0])
        except Exception as e:
            logger.error(f"Json Parse Error: {traceback.format_exc()}-{content}")
            return default_res
        return json_res


    def __shade_initial_postprocess(self, content: str) -> Optional[ShadeInfo]:
        """Processes the initial shade generation response.
        
        Args:
            content: Raw LLM response text.
            
        Returns:
            ShadeInfo object or empty dictionary if processing fails.
        """
        shade_generate_pattern = r"\{.*\}"
        shade_raw_info = self.__parse_json_response(content, shade_generate_pattern)

        if not shade_raw_info:
            logger.error(f"Failed to parse the shade generate result: {content}")
            return {}  # Return an empty dictionary

        logger.info(f"Shade Generate Result: {shade_raw_info}")

        raw_shade_info = ShadeInfo(
            name=shade_raw_info.get("domainName", ""),
            aspect=shade_raw_info.get("aspect", ""),
            icon=shade_raw_info.get("icon", ""),
            descThirdView=shade_raw_info.get("domainDesc", ""),
            contentThirdView=shade_raw_info.get("domainContent", ""),
        )
        raw_shade_info.timelines = [
            ShadeTimeline.from_raw_format(timeline)
            for timeline in shade_raw_info.get("domainTimelines", [])
        ]
        raw_shade_info = self.__add_second_view_info(raw_shade_info)
        return raw_shade_info


    def _initial_shade_process(self, new_memory_list: List[Note]) -> Optional[ShadeInfo]:
        """Processes the initial shade generation from new memories.
        
        Args:
            new_memory_list: List of new memories to generate shade from.
            
        Returns:
            A new ShadeInfo object generated from the memories.
        """
        # Use token-managed prompt generation instead of simple concatenation
        user_prompt = self._manage_memory_tokens(new_memory_list, max_total_tokens=100000)
        
        # Log token usage for monitoring
        total_tokens = self._count_tokens(user_prompt)
        logger.info(f"Generated prompt with {total_tokens} tokens for {len(new_memory_list)} notes")

        shade_generate_message = self._build_message(SHADE_INITIAL_PROMPT, user_prompt)

        response = self._call_llm_with_retry(shade_generate_message)
        content = response.choices[0].message.content

        logger.info(f"Shade Generate Result: {content}")
        return self.__shade_initial_postprocess(content)


    def _merge_shades_info(
        self, old_memory_list: List[Note], shade_info_list: List[ShadeInfo]
    ) -> ShadeInfo:
        """Merges multiple shades into a single shade.
        
        Args:
            old_memory_list: List of existing memories.
            shade_info_list: List of shade information to be merged.
            
        Returns:
            A new ShadeInfo object representing the merged shade.
        """
        # Use token-managed prompt generation for shade merging
        user_prompt = self._manage_shade_merge_tokens(shade_info_list, max_total_tokens=80000)
        
        # Log token usage for monitoring
        total_tokens = self._count_tokens(user_prompt)
        logger.info(f"Generated shade merge prompt with {total_tokens} tokens for {len(shade_info_list)} shades")

        merge_shades_message = self._build_message(SHADE_MERGE_PROMPT, user_prompt)
        response = self._call_llm_with_retry(merge_shades_message)
        content = response.choices[0].message.content
        logger.info(f"Shade Generate Result: {content}")
        return self.__shade_merge_postprocess(content)


    def __shade_merge_postprocess(self, content: str) -> ShadeInfo:
        """Processes the shade merging response.
        
        Args:
            content: Raw LLM response text.
            
        Returns:
            A new ShadeInfo object representing the merged shade.
            
        Raises:
            Exception: If parsing the shade generation result fails.
        """
        shade_merge_pattern = r"\{.*\}"
        shade_merge_info = self.__parse_json_response(content, shade_merge_pattern)
        if not shade_merge_info:
            raise Exception(f"Failed to parse the shade generate result: {content}")

        logger.info(f"Shade Merge Result: {shade_merge_info}")
        merged_shade_info = ShadeInfo(
            name=shade_merge_info.get("newInterestName", ""),
            aspect=shade_merge_info.get("newInterestAspect", ""),
            icon=shade_merge_info.get("newInterestIcon", ""),
            descThirdView=shade_merge_info.get("newInterestDesc", ""),
            contentThirdView=shade_merge_info.get("newInterestContent", ""),
        )

        merged_shade_info.timelines = [
            ShadeTimeline.from_raw_format(timeline)
            for timeline in shade_merge_info.get("newInterestTimelines", [])
        ]
        merged_shade_info = self.__add_second_view_info(merged_shade_info)
        return merged_shade_info


    def __shade_improve_postprocess(self, old_shade: ShadeInfo, content: str) -> ShadeInfo:
        """Processes the shade improvement response.
        
        Args:
            old_shade: The original ShadeInfo object to improve.
            content: Raw LLM response text.
            
        Returns:
            Updated ShadeInfo object.
            
        Raises:
            Exception: If parsing the shade generation result fails.
        """
        shade_improve_pattern = r"\{.*\}"
        shade_improve_info = self.__parse_json_response(content, shade_improve_pattern)
        if not shade_improve_info:
            raise Exception(f"Failed to parse the shade generate result: {content}")

        logger.info(f"Shade Improve Result: {shade_improve_info}")
        old_shade.imporve_shade_info(**shade_improve_info)
        shade_info = self.__add_second_view_info(old_shade)
        return shade_info


    def _improve_shade_info(
        self, new_memory_list: List[Note], old_shade_info: ShadeInfo
    ) -> ShadeInfo:
        """Improves existing shade information with new memories.
        
        Args:
            new_memory_list: List of new memories to incorporate.
            old_shade_info: Existing ShadeInfo object to improve.
            
        Returns:
            Updated ShadeInfo object.
        """
        recent_memories_str = "\n\n".join(
            [memory.to_str() for memory in new_memory_list]
        )

        user_prompt = f""" Original Shade Info:
{old_shade_info.to_str()}

Recent Memories:
{recent_memories_str}
"""
        shade_improve_message = self._build_message(SHADE_IMPROVE_PROMPT, user_prompt)
        response = self._call_llm_with_retry(shade_improve_message)
        content = response.choices[0].message.content
        logger.info(f"Shade Generate Result: {content}")
        return self.__shade_improve_postprocess(old_shade_info, content)


    def generate_shade(
        self,
        old_memory_list: List[Note],
        new_memory_list: List[Note],
        shade_info_list: List[ShadeInfo],
    ) -> Optional[ShadeInfo]:
        """Generates or updates a shade based on memories.
        
        Each time, a batch of memories within a cluster is passed in,
        so it appears that only one shade is generated here.
        
        Args:
            old_memory_list: List of existing memories.
            new_memory_list: List of new memories to incorporate.
            shade_info_list: List of existing ShadeInfo objects.
            
        Returns:
            A new or updated ShadeInfo object, or None if generation fails.
            
        Raises:
            Exception: If input parameters are abnormal.
        """
        logger.warning(f"shade_info_list: {shade_info_list}")
        logger.warning(f"old_memory_list: {old_memory_list}")
        logger.warning(f"new_memory_list: {new_memory_list}")
        
        if not (shade_info_list or old_memory_list):
            logger.info(
                f"Shades initial Process! Current shade have {len(new_memory_list)} memories!"
            )
            new_shade = self._initial_shade_process(new_memory_list)
        elif shade_info_list and old_memory_list:
            if len(shade_info_list) > 1:
                logger.info(
                    f"Merge shades Process! {len(shade_info_list)} shades need to be merged!"
                )
                raw_shade = self._merge_shades_info(old_memory_list, shade_info_list)
            else:
                raw_shade = shade_info_list[0]
            logger.info(
                f"Update shade Process! Current shade should improve {len(new_memory_list)} memories!"
            )
            new_shade = self._improve_shade_info(new_memory_list, raw_shade)
        else:
            # Means either shade_info_list or old_memory_list is empty, indicating an abnormal backend input parameter.
            logger.error(traceback.format_exc())
            raise Exception(
                "The shade_info_list or old_memory_list is empty! Please check the input!"
            )

        # Check if new_shade is an empty dictionary(focus on initial stage)
        if not new_shade:
            return None

        return new_shade


class ShadeMerger:
    def __init__(self):
        self.user_llm_config_service = UserLLMConfigService()
        self.user_llm_config = self.user_llm_config_service.get_available_llm()
        if self.user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.client = OpenAI(
                api_key=self.user_llm_config.chat_api_key,
                base_url=self.user_llm_config.chat_endpoint,
            )
            self.model_name = self.user_llm_config.chat_model_name
        
        self.model_params = {
            "temperature": 0,
            "max_tokens": 3000,
            "top_p": 0,
            "frequency_penalty": 0,
            "seed": 42,
            "presence_penalty": 0,
            "timeout": 45,
        }
        self.preferred_language = "en"
        self._top_p_adjusted = False  # Flag to track if top_p has been adjusted


    def _fix_top_p_param(self, error_message: str) -> bool:
        """Fixes the top_p parameter if an API error indicates it's invalid.
        
        Some LLM providers don't accept top_p=0 and require values in specific ranges.
        This function checks if the error is related to top_p and adjusts it to 0.001,
        which is close enough to 0 to maintain deterministic behavior while satisfying
        API requirements.
        
        Args:
            error_message: Error message from the API response.
            
        Returns:
            bool: True if top_p was adjusted, False otherwise.
        """
        if not self._top_p_adjusted and "top_p" in error_message.lower():
            logger.warning("Fixing top_p parameter from 0 to 0.001 to comply with model API requirements")
            self.model_params["top_p"] = 0.001
            self._top_p_adjusted = True
            return True
        return False

    def _call_llm_with_retry(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Calls the LLM API with automatic retry for parameter adjustments.
        
        This function handles making API calls to the language model while
        implementing automatic parameter fixes when errors occur. If the API
        rejects the call due to invalid top_p parameter, it will adjust the
        parameter value and retry the call once.
        
        Args:
            messages: List of messages for the API call.
            **kwargs: Additional parameters to pass to the API call.
            
        Returns:
            API response object from the language model.
            
        Raises:
            Exception: If the API call fails after all retries or for unrelated errors.
        """
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.model_params,
                **kwargs
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"API Error: {error_msg}")
            
            # Try to fix top_p parameter if needed
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 400:
                if self._fix_top_p_param(error_msg):
                    logger.info("Retrying LLM API call with adjusted top_p parameter")
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **self.model_params,
                        **kwargs
                    )
            
            # Re-raise the exception
            raise

    def _build_user_prompt(self, shade_info_list: List[ShadeMergeInfo]) -> str:
        """Builds a user prompt from shade information list.
        
        Args:
            shade_info_list: List of shade merge information.
            
        Returns:
            Formatted string containing shade information.
        """
        shades_str = "\n\n".join(
            [
                f"Shade ID: {shade.id}\n"
                f"Name: {shade.name}\n"
                f"Aspect: {shade.aspect}\n"
                f"Description Third View: {shade.desc_third_view}\n"
                f"Content Third View: {shade.content_third_view}\n"
                for shade in shade_info_list
            ]
        )

        return f"""Shades List:
{shades_str}
"""


    def _calculate_merged_shades_center_embed(
        self, shades: List[ShadeMergeInfo]
    ) -> List[float]:
        """Calculates the center embedding for merged shades.
        
        Args:
            shades: List of shades to merge.
            
        Returns:
            A list of floats representing the new center embedding.
            
        Raises:
            ValueError: If no valid shades found or total cluster size is zero.
        """
        if not shades:
            raise ValueError("No valid shades found for the given merge list.")

        total_embedding = np.zeros(
            len(shades[0].cluster_info["centerEmbedding"])
        )  # Assuming center_embedding is a fixed-length vector
        total_cluster_size = 0

        for shade in shades:
            cluster_size = shade.cluster_info["clusterSize"]
            center_embedding = np.array(shade.cluster_info["centerEmbedding"])
            total_embedding += cluster_size * center_embedding
            total_cluster_size += cluster_size

        if total_cluster_size == 0:
            raise ValueError(
                "Total cluster size is zero, cannot compute the new center embedding."
            )

        new_center_embedding = total_embedding / total_cluster_size
        return new_center_embedding.tolist()


    def _build_message(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Builds the message structure for the LLM API.
        
        Args:
            system_prompt: The system prompt to guide the LLM behavior.
            user_prompt: The user prompt containing the actual query.
            
        Returns:
            A list of message dictionaries formatted for the LLM API.
        """
        raw_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.preferred_language:
            raw_message.append(
                {
                    "role": "system",
                    "content": PREFER_LANGUAGE_SYSTEM_PROMPT.format(
                        language=self.preferred_language
                    ),
                }
            )
        return raw_message


    def __parse_json_response(
        self, content: str, pattern: str, default_res: dict = None
    ) -> Any:
        """Parses JSON response from LLM output.
        
        Args:
            content: The raw text response from the LLM.
            pattern: Regex pattern to extract the JSON string.
            default_res: Default result to return if parsing fails.
            
        Returns:
            Parsed JSON object or default_res if parsing fails.
        """
        matches = re.findall(pattern, content, re.DOTALL)
        if not matches:
            logger.error(f"No Json Found: {content}")
            return default_res
        try:
            json_res = json.loads(matches[0])
        except Exception as e:
            logger.error(f"Json Parse Error: {traceback.format_exc()}-{content}")
            return default_res
        return json_res


    def merge_shades(self, shade_info_list: List[ShadeMergeInfo]) -> ShadeMergeResponse:
        """Merges multiple shades based on their similarity.
        
        Args:
            shade_info_list: List of shade information to be evaluated for merging.
            
        Returns:
            ShadeMergeResponse object with merge results or error information.
        """
        try:
            for shade in shade_info_list:
                logger.info(f"shade: {shade}")

            user_prompt = self._build_user_prompt(shade_info_list)
            merge_decision_message = self._build_message(
                SHADE_MERGE_DEFAULT_SYSTEM_PROMPT, user_prompt
            )
            logger.info(f"Built merge_decision_message: {merge_decision_message}")

            response = self._call_llm_with_retry(merge_decision_message)
            content = response.choices[0].message.content
            logger.info(f"Shade Merge Decision Result: {content}")

            try:
                merge_shade_list = self.__parse_json_response(content, r"\[.*\]")
                logger.info(f"Parsed merge_shade_list: {merge_shade_list}")
            except Exception as e:
                raise Exception(
                    f"Failed to parse the shade merge list: {content}"
                ) from e

            # Validate if merge_shade_list is empty
            if not merge_shade_list:
                final_merge_shade_list = []
            else:
                # Calculate new cluster embeddings for each group of shades
                final_merge_shade_list = []
                for group in merge_shade_list:
                    shade_ids = group  # Directly use group as it's now a list
                    logger.info(f"Processing group with shadeIds: {shade_ids}")
                    if not shade_ids:
                        continue

                    # Fetch shades based on shadeIds
                    shades = [
                        shade for shade in shade_info_list if str(shade.id) in shade_ids
                    ]  # Ensure shade.id is string type

                    # Skip current group if shades is empty
                    if not shades:
                        logger.info(
                            f"No valid shades found for shadeIds: {shade_ids}. Skipping this group."
                        )
                        continue

                    # Calculate the new cluster embedding (center vector)
                    new_cluster_embedd = self._calculate_merged_shades_center_embed(
                        shades
                    )
                    logger.info(
                        f"Calculated new cluster embedding: {new_cluster_embedd}"
                    )

                    final_merge_shade_list.append(
                        {"shadeIds": shade_ids, "centerEmbedding": new_cluster_embedd}
                    )

            result = {"mergeShadeList": final_merge_shade_list}
            response = ShadeMergeResponse(result=result, success=True)

        except Exception as e:
            logger.error(traceback.format_exc())
            response = ShadeMergeResponse(result=str(e), success=False)

        return response
