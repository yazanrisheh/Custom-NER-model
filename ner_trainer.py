import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from spacy.scorer import Scorer
import random
import streamlit as st

def create_spacy_format(text, selected_entities):
    """Converts user selections into the spaCy training data format.
       Deduplicates overlapping or duplicate entity annotations."""
    unique_entities_set = {(entity['start'], entity['end'], entity['label']) for entity in selected_entities}
    unique_entities = sorted(list(unique_entities_set), key=lambda x: x[0])
    return (text, {"entities": unique_entities})

def train_spacy_model(train_data, output_dir="custom_ner_model", epochs=50, drop=0.5, batch_size_config=2, progress_callback=None):
    """Trains or updates a spaCy NER model and updates progress via the callback."""
    model_name = "en_core_web_lg"

    try:
        nlp = spacy.load(model_name)
        st.write(f"Loaded existing spaCy model: {model_name}")
    except OSError:
        st.write(f"Could not load model '{model_name}'.  Using blank 'en' model.")
        nlp = spacy.blank("en")
        st.write("Created a blank 'en' model.")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in train_data:
        for ent in annotations.get('entities', []):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for epoch in range(epochs):
            random.shuffle(train_data)
            losses = {}

            if isinstance(batch_size_config, int):
                batches = minibatch(train_data, size=batch_size_config)
            elif isinstance(batch_size_config, tuple):
                batches = minibatch(train_data, size=compounding(*batch_size_config))
            else:
                batches = minibatch(train_data, size=iter(batch_size_config))

            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=drop, losses=losses, sgd=optimizer)
            st.write(f"Epoch {epoch + 1}/{epochs} completed. Losses: {losses}")
            if progress_callback:
                progress_callback((epoch + 1) / epochs)
    nlp.to_disk(output_dir)
    st.write(f"Model saved to: {output_dir}")
    return nlp

def test_model(nlp, test_texts):
    """Tests the trained model, displays results, and shows evaluation metrics."""
    st.write("## Test Results")
    scorer = Scorer()
    examples = []

    for text in test_texts:
        doc = nlp(text)
        
        gold_dict = {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
        example = Example.from_dict(doc, gold_dict)
        example.predicted = nlp(example.predicted)
        examples.append(example)

        entities_output = ', '.join([f"{ent.text} ({ent.label_})" for ent in doc.ents])
        st.write(f'**Original Text:** {text}')
        st.write(f'**Entities:** {entities_output if entities_output else "No entities found."}')

        # Replace entities in text with their label (for display)
        replaced_text = text
        for ent in doc.ents:
            replaced_text = replaced_text.replace(ent.text, f"[{ent.label_}]")
        st.write(f"**Text with Entities Replaced:** {replaced_text}")
        st.write("---")

    scores = scorer.score(examples)
    st.write("## Evaluation Metrics (on Test Data)")
    st.write(f"**Precision:** {scores['ents_p']:.3f}")
    st.write(f"**Recall:**    {scores['ents_r']:.3f}")
    st.write(f"**F1-Score:**  {scores['ents_f']:.3f}")
