import streamlit as st
from ner_trainer import train_spacy_model, test_model, create_spacy_format
import spacy
import json
import re

def main():
    st.title("Custom NER Training with Streamlit")

    st.write(
        """
        **Instructions:**

        1. **Choose Input Method:** Select "Text Input", "JSON Upload", or "Generate Sample Dataset".
        2. **Enter Training Data:**
            - **Text Input:** Enter overall text, entity text, and label. Use "Add More" and the trash bin icon. *Label every instance of every entity.*
            - **JSON Upload:** Upload a JSON file.
            - **Generate Sample Dataset:** Use a pre-populated dataset.
        3. **Manage Custom Labels:** Add or remove custom labels (expand "Manage Labels").
        4. **Train Model:** Set parameters and click "Train Model".
        5. **Test Model:** Enter test text or use examples and click "Test Model".
        """
    )

    # --- Overall Text Input ---
    text = st.text_area(
        "Enter overall text:",
        placeholder="Enter your paragraph or text document here.",
        help="Enter the complete sentence or paragraph here."
    )

    # --- Training Data Input ---
    st.sidebar.header("Training Data Input")
    input_method = st.sidebar.radio(
        "Choose input method:", ["Text Input", "JSON Upload", "Generate Sample Dataset"]
    )

    # --- Custom Labels Management ---
    with st.sidebar.expander("Manage Labels", expanded=False):
        if "allowed_labels" not in st.session_state:
            st.session_state.allowed_labels = ["QUANTITY", "PRODUCT", "ORG", "GPE", "COMPANY", "NAME"]

        new_label = st.text_input("Add Custom Label:", "").upper()
        if st.button("Add Label") and new_label:
            if new_label not in st.session_state.allowed_labels:
                st.session_state.allowed_labels.append(new_label)
            else:
                st.warning("Label already exists.")

        st.write("Current Labels:")
        label_cols = st.columns(3)
        col_index = 0
        for label in st.session_state.allowed_labels:
            if label_cols[col_index % 3].button(f"{label} 🗑️", key=f"del_label_{label}"):
                st.session_state.allowed_labels.remove(label)
                st.rerun()
            col_index += 1

    # --- Initialize train_data ---
    train_data = []

    if input_method == "Text Input":
        if "text_inputs" not in st.session_state:
            st.session_state.text_inputs = []

        # --- Entity Input Table ---
        st.subheader("Entity Input")
        cols = st.columns([3, 3, 1])
        cols[0].markdown("**Entity Text**")
        cols[1].markdown("**Label**")
        cols[2].markdown("**Delete**")

        for i in range(len(st.session_state.text_inputs)):
            with st.container():
                cols = st.columns([3, 3, 1])
                entity_text_key = f"entity_text_{i}"
                label_key = f"label_{i}"

                current_row = st.session_state.text_inputs[i]
                current_row['entity_text'] = cols[0].text_input(
                    "Entity Text", key=entity_text_key, value=current_row.get('entity_text', ''),
                    label_visibility="collapsed", help="The word or phrase."
                )
                current_row['label'] = cols[1].selectbox(
                    "Label", options=st.session_state.allowed_labels, key=label_key,
                    index=st.session_state.allowed_labels.index(current_row.get('label'))
                          if current_row.get('label') in st.session_state.allowed_labels else 0,
                    label_visibility="collapsed"
                )

                if cols[2].button("🗑️", key=f"del_{i}"):
                    st.session_state.text_inputs.pop(i)
                    st.experimental_rerun()
                    break

        if st.button("Add Data"):
            st.session_state.text_inputs.append({'entity_text': "", 'label': ''})

        if text:
            selected_entities = []
            for input_data in st.session_state.text_inputs:
                entity_text = input_data.get('entity_text')
                label = input_data.get('label')
                if entity_text and label:
                    # Find all occurrences of the entity text in the overall text
                    for match in re.finditer(re.escape(entity_text), text):
                        start_index = match.start()
                        end_index = match.end()
                        selected_entities.append({
                            'start': start_index,
                            'end': end_index,
                            'label': label
                        })

            if selected_entities:
                # Deduplicate and convert to spaCy format
                spacy_data = create_spacy_format(text, selected_entities)
                train_data.append(spacy_data)

    elif input_method == "JSON Upload":
        uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
        if uploaded_file is not None:
            try:
                train_data = json.load(uploaded_file)
                if not isinstance(train_data, list):
                    raise ValueError("Uploaded JSON must be a list of training examples.")
                for example in train_data:
                    if not isinstance(example, tuple) or len(example) != 2 or not isinstance(example[0], str) or not isinstance(example[1], dict) or "entities" not in example[1]:
                        raise ValueError("Invalid training data format in JSON.")
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
                train_data = []

    elif input_method == "Generate Sample Dataset":
        train_data = [
            ("What is the price of 10 bananas?", {"entities": [(21, 23, "QUANTITY"), (24, 31, "PRODUCT")]}),
            ("I need 5 apples and 2 oranges.", {"entities": [(7, 8, "QUANTITY"), (9, 15, "PRODUCT"), (20, 21, "QUANTITY"), (22, 29, "PRODUCT")]}),
            ("Can I have a dozen roses, please?", {"entities": [(12, 17, "QUANTITY"), (18, 23, "PRODUCT")]})
        ]
        st.write("Sample dataset loaded.")

    # --- Model Training ---
    with st.sidebar.expander("Model Training", expanded=True):
        output_dir = st.text_input(
            "Output Directory Name:", "custom_ner_model",
            help="The directory where the trained model will be saved."
        )
        epochs = st.number_input(
            "Number of Epochs:", min_value=1, value=50,
            help="The number of times the model will iterate over the entire training dataset."
        )

    # --- Hyperparameter Tuning ---
    with st.sidebar.expander("Hyperparameter Tuning", expanded=True):
        drop = st.number_input(
            "Dropout Rate:", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="The probability of randomly dropping out (ignoring) neurons during training. Helps prevent overfitting."
        )

        batch_size_type = st.radio(
            "Batch Size Type:", ["Constant", "Compounding", "Custom Iterable"],
            help="Choose the method for determining the batch size during training."
        )
        if batch_size_type == "Constant":
            batch_size = st.number_input(
                "Batch Size:", min_value=1, value=2, step=1,
                help="The fixed number of training examples processed in each batch."
            )
            batch_size_config = batch_size
        elif batch_size_type == "Compounding":
            start = st.number_input("Start:", min_value=1.0, value=4.0, help="The initial batch size.")
            stop = st.number_input("Stop:", min_value=1.0, value=32.0, help="The maximum batch size.")
            compound = st.number_input("Compound:", min_value=1.001, value=1.001, help="The compounding factor for increasing the batch size.")
            batch_size_config = (start, stop, compound)
        elif batch_size_type == "Custom Iterable":
            batch_sizes_str = st.text_input("Batch Sizes (comma-separated):", "1,2,4,8", help="Enter a comma-separated list of batch sizes to use.")
            try:
                batch_size_config = [int(x.strip()) for x in batch_sizes_str.split(",")]
            except ValueError:
                st.error("Invalid batch sizes. Please enter comma-separated integers.")
                batch_size_config = [2]

    if st.sidebar.button("Train Model") and train_data:
        if not text:
            st.sidebar.error("Please enter the overall text.")
        else:
            progress_bar = st.progress(0)
            with st.spinner("Training..."):
                trained_nlp = train_spacy_model(
                    train_data, output_dir, epochs, drop, batch_size_config,
                    progress_callback=lambda prog: progress_bar.progress(int(prog * 100))
                )
            st.success("Training completed!")

    # --- Model Testing ---
    with st.sidebar.expander("Model Testing", expanded=True):
        test_input_method = st.radio("Choose test input method:", ["Text Input", "Predefined Examples"])
        test_texts = []

        if test_input_method == "Text Input":
            test_text = st.text_area("Enter test text:", "How much for 5 apples?")
            if test_text:
                test_texts.append(test_text)
        elif test_input_method == "Predefined Examples":
            test_texts = [
                "How much for 3 oranges?",
                "I want 15 chairs for the conference.",
                "Can you give me the price for 6 desks?"
            ]
            st.write("Using predefined test examples.")

        if st.button("Test Model"):
            try:
                trained_nlp = spacy.load(output_dir)
                st.success(f"Loaded trained model from: {output_dir}")
            except OSError:
                try:
                    trained_nlp = spacy.load("en_core_web_lg")
                    st.warning(f"Could not load from '{output_dir}'. Loaded base model: 'en_core_web_lg' instead.")
                except OSError:
                    st.error(f"Could not load '{output_dir}' or 'en_core_web_lg'.")
                    trained_nlp = None

            if trained_nlp:
                test_model(trained_nlp, test_texts)

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        """
        Made by [Yazan Risheh](https://www.linkedin.com/in/yazan-risheh-8b87211a1/) | 
        [GitHub](https://github.com/yazanrisheh?tab=repositories)
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
