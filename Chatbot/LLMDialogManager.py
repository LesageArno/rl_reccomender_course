from typing import Optional, List, Dict, Set, Any
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

ChatMessage = Dict[str, str]  # {"role": "user" | "assistant" | "system", "content": "..."}

DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant specialized in job and skill counselling and course recommendation.

Your main responsibilities are:
- Help the user clarify their goals, background, and constraints.
- Help the user express skill preferences (skills they want to acquire, avoid, or already have).
- Explain and discuss course recommendations produced by an external RL-based recommender agent.
- Keep the conversation focused, concrete, and helpful for learning and career development.

Interaction rules:
- Always answer in English, unless the user explicitly asks for another language.
- Use a professional but friendly tone.
- Be concise but clear: avoid long essays, focus on key points and actionable suggestions.
- For simple explanatory questions, answer in at most 4 short sentences, unless the user explicitly asks for more detail.
- You may engage in light chitchat when the user speaks casually, but always bring the conversation back to skills, career exploration, learning paths, or course recommendations
- Whenever you answer, end with one short, friendly sentence inviting the user to continue or ask for more help.
- When you explain recommendations, refer explicitly to:
  - the user's goals and preferences
  - the skills covered by the courses
  - how the course sequence helps fill gaps or reach target roles.

Context you may receive:
- A textual summary of the user's profile and preferences (skills, goals, constraints).
- A description of course sequences recommended by the RL agent, possibly with scores or skill coverage.
Use the provided context to ground your answers about the user profile and the recommended courses.
Do NOT invent specific details about the course catalog or the user's history that are not in the context.
You may use your general knowledge for generic explanations about skills, technologies, and job roles.
""".strip()

class LLMDialogManager:
    """
    Minimal dialog manager wrapper around a Phi-3-mini-4k-instruct model.

    This first version only:
    - loads the tokenizer and model
    - configures 4-bit quantization (BitsAndBytes)
    - prepares basic generation parameters

    The actual prompting and generate_reply(...) method will be added later.
    """

    def __init__(
        self,
        model_card: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 40,
        temperature: Optional[float] = None,
        num_return_sequences: int = 1,
    ) -> None:
        """
        Initialize tokenizer, model and base generation configuration.

        Args:
            model_card: Hugging Face model identifier.
            max_new_tokens: Default max_new_tokens used for generation.
            temperature: Default temperature. None means greedy decoding.
            num_return_sequences: Number of candidate outputs to generate.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_card = model_card

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        # Align with the notebook: use EOS as pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define terminators as in the notebook
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # BitsAndBytes 4-bit quantization configuration
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model with quantization, device_map='auto' as in the notebook
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_card,
            return_dict=True,
            quantization_config=self.bnb_config,
            device_map=self.device,
        )

        # Use the model's default generation_config as base
        self.generation_config = self.model.generation_config
        self.generation_config.max_new_tokens = max_new_tokens
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.generation_config.temperature = temperature
        self.generation_config.num_return_sequences = num_return_sequences


    def build_messages(
        self,
        user_input: str,
        history: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
        extra_context: Optional[str] = None,
    ) -> List[ChatMessage]:
        """
        Build a list of chat messages to feed into the tokenizer chat template.

        Args:
            user_input: Latest user utterance (natural language).
            history: Optional past messages in HF chat format.
            system_prompt: Optional system-level instruction. If None,
                           DEFAULT_SYSTEM_PROMPT will be used.
            extra_context: Optional textual context, e.g. summary of the user's
                           preferences and RL course recommendations.

        Returns:
            A list of messages in the format expected by apply_chat_template.
        """
        messages: List[ChatMessage] = []

        # 1) System prompt: role + global behavior of the assistant
        sys_content = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        messages.append(
            {
                "role": "system",
                "content": sys_content,
            }
        )

        # 2) Extra context: e.g. RL recommendations, profile summary
        if extra_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Context information about the user and recommended courses:\n"
                        f"{extra_context}\n"
                        "Use this information to answer, but do not repeat it verbatim."
                    ),
                }
            )

        # 3) Past dialog history (if any)
        if history:
            messages.extend(history)

        # 4) Current user message
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        return messages


    
    def generate_reply(
        self,
        messages: List[ChatMessage],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a reply from the model given HF-style chat messages.

        Args:
            messages: Chat messages in HF format (system/user/assistant).
            max_new_tokens: Optional override for max_new_tokens.
            temperature: Optional override for temperature.

        Returns:
            Assistant reply as plain text.
        """
        # Turn messages into model input using the chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        # Start from base config and apply overrides without mutating the original
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature

        output_ids = self.model.generate(
            inputs,
            generation_config=gen_config,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Take only the newly generated part (after the prompt length)
        generated = output_ids[0]
        completion_ids = generated[inputs.shape[-1] :]
        reply = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
        ).strip()

        return reply
    
    def explain_updated_preferences(
        self,
        original_text: str,
        include_pairs: List[tuple[str, str]],
        avoid_pairs: List[tuple[str, str]],
        acquired_pairs: List[tuple[str, str]],
    ) -> str:
        """
        High-level helper: given the original user text and the extracted
        preferences, generate a short explanation for the user.
        """
        include_block = (
            "Include skills:\n"
            + "\n".join(f"- {name} (uid={uid})" for name, uid in include_pairs)
            if include_pairs
            else "Include skills:\n- None"
        )
        avoid_block = (
            "Avoid skills:\n"
            + "\n".join(f"- {name} (uid={uid})" for name, uid in avoid_pairs)
            if avoid_pairs
            else "Avoid skills:\n- None"
        )
        acquired_block = (
            "Acquired skills:\n"
            + "\n".join(f"- {name} (uid={uid})" for name, uid in acquired_pairs)
            if acquired_pairs
            else "Acquired skills:\n- None"
        )

        extra_context = (
            f"Original user text:\n{original_text}\n\n"
            f"From this text, the system extracted the following preferences:\n\n"
            f"{include_block}\n\n{avoid_block}\n\n{acquired_block}"
        )

        user_input_for_llm = (
            "Explain to the user, in a concise and clear way, which skill "
            "preferences have just been added or updated in their profile, "
            "and what this means for future course recommendations."
        )

        return self.chat(
            user_input=user_input_for_llm,
            history=None,
            system_prompt=None,
            extra_context=extra_context,
            max_new_tokens=160,
            temperature=0.2,
        )
    
    def build_recommendation_context(
        self,
        course_ids: List[str],
        skills_learned: Dict[str, str],
        include_pairs: Set[tuple[str, str]],
        avoid_pairs: Set[tuple[str, str]],
        acquired_pairs: Optional[Set[tuple[str, str]]] = None,
    ) -> str:
        """
        Build a textual context summarizing:
        - user preferences (include/avoid/acquired skills, if any)
        - RL recommendation (course IDs + union of skills covered)
        This context will be passed to the LLM as extra_context.
        """
        include_names = [name for (name, _uid) in include_pairs]
        avoid_names = [name for (name, _uid) in avoid_pairs]
        acquired_names = [name for (name, _uid) in acquired_pairs]

        include_block = (
            "Include skills: " + ", ".join(include_names)
            if include_names else "Include skills: None"
        )
        avoid_block = (
            "Avoid skills: " + ", ".join(avoid_names)
            if avoid_names else "Avoid skills: None"
        )
        acquired_block = (
            "Acquired skills: " + ", ".join(acquired_names)
            if acquired_names else "Acquired skills: None or unknown"
        )

        # RL recommendation summary
        course_seq_str = ", ".join(str(c) for c in course_ids) if course_ids else "None"

        # Skills learned from courses (with mastery level)
        if skills_learned:
            learned_lines = [
                f"- {skill_name} (target level: {level})"
                for skill_name, level in sorted(skills_learned.items())
            ]
            skills_learned_block = (
                "Skills covered by the recommended courses (union, with target levels):\n"
                + "\n".join(learned_lines)
            )
        else:
            skills_learned_block = (
                "Skills covered by the recommended courses (union, with target levels):\n"
                "- None"
            )

        # Check if user has explicit preferences
        has_preferences = bool(include_names or avoid_names)

        if has_preferences:
            note = (
                "The user has explicit skill preferences (include/avoid). "
                "The RL agent tried to select courses that match include skills "
                "and avoid avoid skills."
            )
        else:
            note = (
                "The user did not specify explicit skill preferences. "
                "The RL agent selected courses to improve the user's general job "
                "applicability given the current job market."
            )

        context = (
            "User preferences:\n"
            f"- {include_block}\n"
            f"- {avoid_block}\n"
            #f"- {acquired_block}\n\n"
            "RL recommendation:\n"
            f"- Sequence of course IDs: {course_seq_str}\n"
            f"- {skills_learned_block}\n\n"
            f"Additional note:\n- {note}"
        )

        user_input_for_llm = (
                "Explain to the user why this sequence of courses is appropriate for them, "
                "based on their preferences and the skills covered by the courses. "
                "Be concise and concrete."
            )
        
        reply = self.chat(
            user_input=user_input_for_llm,
            history=None,
            system_prompt=None,
            extra_context=context,
            max_new_tokens=200,
            temperature=0.2,
        )

        return reply



    def chat(
        self,
        user_input: str,
        history: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
        extra_context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Convenience method:
        - build messages from system_prompt, history, extra_context and user_input
        - generate a reply
        """
        messages = self.build_messages(
            user_input=user_input,
            history=history,
            system_prompt=system_prompt,
            extra_context=extra_context,
        )
        return self.generate_reply(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


def extract_skills_from_cv_text(self, cv_text: str, max_new_tokens: int = 512) -> List[Dict[str, Any]]:
    """
    Extract technical skills from raw CV text and return them as a JSON-like list of dicts.
    Each entry contains:
      - snippet: exact substring copied from the CV
      - skill_name: short normalized name of the skill
      - level: 1 (beginner), 2 (intermediate), 3 (advanced)
    """
    system_prompt = """
You extract technical skills from raw CV text.
Return ONLY a valid JSON array with objects of this exact form:
{
  "snippet": "...text copied from the CV...",
  "skill_name": "Python",
  "level": 3
}
Rules:
- Copy the snippet exactly from the CV, without rewriting it.
- Do not invent skills that do not appear in the text.
- Use only integers 1, 2, or 3 for 'level'.
- Output only the JSON array. No explanations.
""".strip()

    user_input = (
        "Extract skills from the following CV text. Output only JSON.\n\n"
        "```text\n"
        f"{cv_text}\n"
        "```"
    )

    raw_reply = self.chat(
        user_input=user_input,
        history=None,
        system_prompt=system_prompt,
        extra_context=None,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    try:
        data = json.loads(raw_reply)
    except json.JSONDecodeError:
        data = []

    if not isinstance(data, list):
        data = []

    return data
