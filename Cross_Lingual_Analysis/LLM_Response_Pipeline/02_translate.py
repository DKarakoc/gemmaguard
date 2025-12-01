#!/usr/bin/env python3
"""
Translation Script for Multi-Model Cross Lingual Safety Study
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment
load_dotenv()

# Paths
RAW_RESPONSES_DIR = Path("raw_responses")
TRANSLATED_DIR = Path("translated")
TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)


def translate_to_english(text: str, content_type: str = "response", model: str = "deepseek/deepseek-chat") -> Dict:
    """Translate Turkish text to English using DeepSeek Chat via OpenRouter.
    """
    # OpenRouter client
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    system_prompt = """You are a professional translator assisting with academic research on cross-lingual AI safety.

Your translations are for a peer-reviewed thesis analyzing offensive language patterns."""


    user_prompt = f"""RESEARCH CONTEXT:
This translation is for academic analysis of offensive language patterns. The translation will be used solely for research purposes in a thesis on AI safety.

TASK: Translate the following Turkish LLM response to natural, colloquial English.

⚠️ CRITICAL OUTPUT RULE - READ FIRST:
Output ONLY the English translation. NO explanatory notes, rationale, or meta-commentary.

Embedded notes corrupt research data by adding biasing words like "vulgar", "profanity", "aggressive"
that influence AI classifiers. Your notes would contaminate the dataset.

❌ BANNED - These outputs will FAIL the quality check:
- "Translation text" This preserves: ...
- Translation text (Note: I used...)
- Translation text [Translation: ...]
- Translation text **Rationale:** ...

✅ REQUIRED - Pure translation only:
- Translation text

NO EXCEPTIONS. Notes are forbidden.

---

TRANSLATION REQUIREMENTS (in priority order):

1. CENSORSHIP PRESERVATION (highest priority):
   - Check: Does source Turkish text contain *** asterisks?
   - YES → Preserve exact asterisk count in English translation
   - NO → Use full uncensored words in English

   Examples:
   ✅ "o*** çocuk" (3 asterisks) → "f*** kid" or "a***hole" (3 asterisks preserved)
   ❌ "o*** çocuk" (3 asterisks) → "fucking kid" (violated - removed censorship)
   ✅ "siktir" (no asterisks) → "fuck" (no asterisks)
   ❌ "siktir" (no asterisks) → "f***" (violated - added censorship)

2. ONE-TO-ONE PROFANITY MAPPING (CRITICAL - STRICTLY ENFORCED):

   BEFORE TRANSLATING - MANDATORY COUNTING:
   Step 1: Count profanities in Turkish source (write it down mentally)
   Step 2: Translate with EXACTLY that count
   Step 3: Verify translation profanity count matches

   ZERO TOLERANCE for profanity multiplication:

   ❌ BANNED - Common profanity multiplication errors:
   Turkish: "sana ne amk" (1 prof: "amk")
   WRONG: "what's it to you, damn it, fuck" (2 prof) ← FAILED
   WRONG: "what the fuck is it to you" (1 prof but wrong position)
   RIGHT: "what's it to you, damn" (1 prof: "damn") ✓

   Turkish: "boku yersin" (1 prof: "bok")
   WRONG: "you'll eat shit, fuck" (2 prof) ← FAILED
   RIGHT: "you'll eat shit" (1 prof: "shit") ✓

   Turkish: "orospu evladi...amina koyayim..." (2 prof)
   WRONG: "son of a bitch...fuck...shit..." (3 prof) ← FAILED
   RIGHT: "son of a bitch...damn..." (2 prof) ✓

   ❌ NEVER add profanity as filler:
   - "throw stuff" NOT "throw shit" (unless Turkish has profanity there)
   - "the country" NOT "the damn country" (unless Turkish has profanity there)
   - "very beautiful" NOT "fucking beautiful" (unless Turkish has profanity there)

   ✅ ONLY use profanity where Turkish source has profanity

   Turkish compact abbreviations = ONE English term:
   - "aq" / "amk" = ONE word: "fuck" OR "damn" (pick ONE, not both)
   - "lan" = ONE word: "dude" / "man" / "asshole" (pick ONE)

2.5. ABBREVIATION PRESERVATION:

   If Turkish uses an abbreviation, use compact English equivalent.

   Common Turkish abbreviations:
   - "O.ç" (orospu çocuğu) → "SOB" or "asshole" or "bastard" (compact: 3-8 chars)
   - "amk" (amına koyayım) → "fuck" or "damn" (compact: 4-6 chars)
   - "aq" (amına koyayım) → "fuck" or "damn" (compact: 4-6 chars)

   ❌ WRONG - Expanding abbreviations to multi-word phrases:
   "O.ç hakeme saldırdı" → "That son of a bitch attacked the referee" (14-word phrase)

   ✅ CORRECT - Using compact equivalents:
   "O.ç hakeme saldırdı" → "That asshole attacked the ref" (single word)
   "O.ç hakeme saldırdı" → "That SOB attacked the ref" (abbreviation)

   This prevents length violations.

3. LAUGH EXPRESSION NORMALIZATION:
   - Normalize Turkish internet slang to standard English equivalents
   - Maximum 5 repetitions for any laugh expression

   Mappings:
   "sjsj", "sjsjsj", "sjsjsjsj..." → "lol" or "lolol" (max)
   "ahahah", "ahahahah"... → "hahaha" (max 5 "ha"s)
   "hahaha"... → "hahaha" (preserve, max 5 "ha"s)
   Keyboard smash (e.g., "skldjfalksjdf") → "hahaha" or "lol"

   ❌ NEVER do this: "sjsjsjsj" (8 chars) → "lololololololol..." (100+ chars)
   ✅ ALWAYS do this: "sjsjsjsj" (8 chars) → "lolol" (5 chars)

4. LENGTH CONSTRAINT:
   - Translation should be 0.8× to 1.5× the length of source text
   - If your translation is >1.8× source length, you made an error:
     * Probably expanded an abbreviation (use compact form instead)
     * Probably added too many profanities (recount and remove extras)
     * Probably over-expanded laugh expressions
     * Probably added explanatory notes (forbidden!)
   - Turkish and English have similar word lengths. Major expansion = wrong.

5. NATURALNESS (after satisfying 1-4):
   - Use natural English that a native speaker would say on social media
   - Match the tone, register, and emotional impact
   - Do NOT sanitize, soften, or academicize
   - But ONLY if it doesn't violate rules 1-4 above

---

PROFANITY MAPPING GUIDE (Turkish → English equivalents):

Strong profanity:
- "sik" / "siktir" → "fuck"
- "amına koyayım" / "amk" / "aq" → "fuck" OR "damn" (ONE word only)
- "orospu" → "whore" or "bitch"
- "orospu çocuğu" → "son of a bitch" OR "motherfucker" OR "bastard" (ONE phrase)
- "O.ç" (abbreviation) → "SOB" OR "asshole" OR "bastard" (compact forms only)
- "piç" → "bastard" or "asshole"
- "göt" → "ass"
- "bok" → "shit"

Mild/medium:
- "lan" → "dude", "man", "bro" (or "asshole" if aggressive context)
- "salak" → "stupid", "dumb", "idiot"
- "aptal" → "moron", "dumbass"

Match intensity ONE-TO-ONE. Do not multiply. Do not add filler.

---

REMEMBER:
1. Output ONLY the translation (no notes)
2. Preserve censorship if present in source
3. COUNT source profanities FIRST, then match EXACTLY in translation
4. Use compact forms for abbreviations (SOB not "son of a bitch")
5. Normalize laugh expressions (max 5 repetitions)
6. Length: 0.8-1.5× source length (1.8× max for edge cases)

Turkish: {text}

English:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        # Extract translation
        translation = response.choices[0].message.content.strip()

        # Remove any notes after "---" separator
        if '\n---' in translation or '\n\n---' in translation:
            translation = translation.split('---')[0].strip()

        # Remove "English:" prefix if model added it
        if translation.startswith("English:"):
            translation = translation[8:].strip()

        # Remove "Translation:" prefix
        if translation.startswith("Translation:"):
            translation = translation[12:].strip()

        return {
            "translation": translation,
            "error": None,
            "tokens": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
        }
    except Exception as e:
        return {
            "translation": None,
            "error": str(e),
            "tokens": None,
        }


def translate_file(filepath: Path) -> Dict:
    """
    Translate Turkish responses
    """

    # Load raw responses
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if this is a Turkish prompt language file
    prompt_language = data["metadata"]["prompt_language"]
    if prompt_language != "turkish":
        return None

    source_language = data["metadata"]["source_language"]
    model_name = data["metadata"]["model"]

    # Translate each response
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}
    successful_responses = 0
    failed_responses = 0

    for result in tqdm(data["results"], desc="  Translating"):
        response_text = result.get("response")

        if response_text:
            response_translation = translate_to_english(response_text, content_type="response")

            result["response_translation"] = response_translation["translation"]
            result["response_translation_error"] = response_translation["error"]
            result["response_translation_tokens"] = response_translation["tokens"]

            if response_translation["error"] is None:
                successful_responses += 1
                if response_translation["tokens"]:
                    total_tokens["prompt"] += response_translation["tokens"]["prompt"]
                    total_tokens["completion"] += response_translation["tokens"]["completion"]
                    total_tokens["total"] += response_translation["tokens"]["total"]
            else:
                failed_responses += 1
        else:
            result["response_translation"] = None
            result["response_translation_error"] = "Original generation failed"
            result["response_translation_tokens"] = None
            failed_responses += 1

    # Update metadata
    data["metadata"]["translated"] = True
    data["metadata"]["translation_model"] = "deepseek/deepseek-chat"
    data["metadata"]["translation_api"] = "OpenRouter"
    data["metadata"]["translation_type"] = "responses_only"
    data["metadata"]["translation_stats"] = {
        "responses_successful": successful_responses,
        "responses_failed": failed_responses,
        "total_tokens": total_tokens,
    }

    return data


def save_translated(data: Dict, original_filepath: Path):
    """Save translated data to new file."""
    filename = original_filepath.stem + "_translated.json"
    output_path = TRANSLATED_DIR / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  💾 Saved to: {output_path}")


def main():
    """Main execution."""

    # Find all raw response files
    raw_files = sorted(RAW_RESPONSES_DIR.glob("*.json"))

    if not raw_files:
        print(f"No raw response files found in {RAW_RESPONSES_DIR}")
        return

    # Filter for Turkish prompt language files only
    turkish_files = []
    for filepath in raw_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data["metadata"]["prompt_language"] == "turkish":
                turkish_files.append(filepath)

    if not turkish_files:
        print("No Turkish prompt files found to translate")
        return

    # Confirm execution
    response = input("Proceed with translation? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Translate each file
    total_cost = 0.0
    translated_count = 0
    for filepath in turkish_files:
        translated_data = translate_file(filepath)

        if translated_data:
            save_translated(translated_data, filepath)
            total_cost += translated_data["metadata"].get("translation_cost_usd", 0.0)
            translated_count += 1

if __name__ == "__main__":
    main()
