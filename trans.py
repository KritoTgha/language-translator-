# Import pipeline form HuggingsFace Transformer
from transformers import pipeline
# Print all the list of language for user  
print(""" List of all language 
       ****************************
Language       Code
English        en
French         fr
German         de
Spanish        es
Italian        it
Chinese        zh
Russian        ru
      --------------------------------
      Type exit or any prompt to quit

      """
     )

while True:
    # 1️⃣ Get source language from the user
    source_lang = input("Source language (name/code): ").strip().lower()
    if source_lang == "exit":  # Exit if user types 'exit'
        break

    # 2️⃣ Get target language from the user
    target_lang = input("Target language (name/code): ").strip().lower()
    if target_lang == "exit":
        break

    # 3️⃣ Get the text to translate
    text = input("Enter text: ").strip()
    if text.lower() == "exit":
        break

    # 4️⃣ Check if both languages are supported
    if source_lang not in language_map or target_lang not in language_map:
        print(" Unsupported language.\n")
        continue  # Skip to the next loop

    # 5️⃣ Convert names to language codes (e.g., 'english' → 'en')
    src_code = language_map[source_lang]
    tgt_code = language_map[target_lang]

    # 6️⃣ Prevent translating into the same language
    if src_code == tgt_code:
        print("⚠ Source and target languages are the same.\n")
        continue

    # 7️⃣ Direct translation if English is source or target
    if src_code == "en" or tgt_code == "en":
        translator = pipeline(
            f"translation_{src_code}_to_{tgt_code}",  # Pipeline task name
            model=f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"  # Specific translation model
        )
        result = translator(text)[0]['translation_text']  # Extract translated text

    else:
        # 8️⃣ If English is NOT involved → translate in two steps:
        # Step 1: Source → English
        to_en = pipeline(
            f"translation_{src_code}_to_en",
            model=f"Helsinki-NLP/opus-mt-{src_code}-en"
        )
        # Step 2: English → Target
        en_to_tgt = pipeline(
            f"translation_en_to_{tgt_code}",
            model=f"Helsinki-NLP/opus-mt-en-{tgt_code}"
        )
        # First translation: Source to English
        intermediate = to_en(text)[0]['translation_text']
        # Second translation: English to Target
        result = en_to_tgt(intermediate)[0]['translation_text']

    # 9️⃣ Show the result to the user
    print(f"   Source language : {source_lang} ({src_code})")
    print(f"   Target language : {target_lang} ({tgt_code})")
    print(f"   Original text   : {text}\n")
    print(f"    Translation: {result}\n")
