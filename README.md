# Custom NER Training with Streamlit

This project provides a Streamlit application for training and testing custom Named Entity Recognition (NER) models using spaCy. This will later on be extended in another repo how to use a fine tuned NER model in a RAG application using LangChain & LangGraph.

## Features

*   **Interactive Data Input:**
    *   **Text Input:** Manually enter text and label entities.
    *   **JSON Upload:** Upload training data in spaCy's JSON format.
    *   **Sample Dataset:** Use a pre-populated sample dataset to get started quickly.
*   **Custom Label Management:** Add and remove custom entity labels.
*   **Hyperparameter Tuning:** Adjust training parameters:
    *   **Epochs:** Number of training iterations.
    *   **Dropout Rate:** Regularization to prevent overfitting.
    *   **Batch Size:**  Choose between constant, compounding, or custom iterable batch sizes.
*   **Model Training:** Train a new spaCy NER model or fine-tune an existing one (defaults to `en_core_web_lg`).
*   **Model Testing:** Evaluate the trained model on test data and view performance metrics (Precision, Recall, F1-Score).  Entities in the test text are replaced with their labels in brackets (e.g., `[COMPANY]`).
*   **Model Saving:** Save the trained model to a specified directory.

## Getting Started

### Prerequisites

*   Python 3.9+
*   spaCy
*   Streamlit

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```
    (If you're not using git, you can download the files as a ZIP and extract them.)

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    You'll also need to download the spaCy model `en_core_web_lg`
    ```bash
     python -m spacy download en_core_web_lg
    ```
 

### Running the Application

1.  **Navigate to the project directory:**

    ```bash
    cd ner_project
    ```

2.  **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

This will open the application in your web browser.

## Usage

1.  **Data Input:** Choose your preferred input method ("Text Input", "JSON Upload", or "Generate Sample Dataset").
2.  **Enter Training Data:**
    *   **Text Input:**  Enter your overall text in the text area.  Then, use the table to specify the entity text and corresponding label for each entity within the overall text.  Use the "Add More" button to add more rows, and the trash bin icon to delete rows.  *It is crucial to label every instance of every entity you want the model to recognize, even if the same word or phrase appears multiple times.* For example:

        ```
        Overall Text: Yazan presented the report.  Afterward, we asked Yazan a few questions.  Yazan's answers were clear, and the team thanked Yazan for his work.

        Entity Input Table:
        | Entity Text | Label |
        |-------------|-------|
        | Yazan       | NAME  |
        | Yazan       | NAME  |
        | Yazan       | NAME  |
        | Yazan       | NAME  |
        ```
        This provides the model with multiple examples of "Yazan" used as a "NAME" in different contexts, which is essential for learning. Whats happening internally is its being converted into spaCy training format. For example, the first "Yazan" is positioned starting from index 0 (Letter Y) and ends at index 5 (Letter n) and so on.
        ```
        ({"entities": [(0, 5, "NAME"), (46, 51, "NAME"), (71, 76, "NAME"), (112, 117, "NAME")]})
        ```

    *   **JSON Upload:** Upload a JSON file containing your training data in spaCy's format.  The format should be a list of tuples, where each tuple contains the text and a dictionary with an "entities" key.  The "entities" value should be a list of `(start, end, label)` tuples.  Example:
        ```json
        [
          ("This is a sentence.", {"entities": [(0, 4, "LABEL")]}),
          ("Another sentence here.", {"entities": [(8, 16, "ANOTHER_LABEL")]})
        ]
        ```
    *   **Generate Sample Dataset:**  This option loads a small, pre-defined dataset to demonstrate the application's functionality.
3.  **Manage Custom Labels (Optional):**  Use the "Manage Labels" expander in the sidebar to add or remove custom entity labels.
4.  **Model Training:**
    *   **Output Directory Name:** Specify the directory where the trained model will be saved.
    *   **Number of Epochs:** Set the number of training iterations.
    *   **Hyperparameter Tuning:** Adjust the dropout rate and batch size settings as needed.
    *   Click the "Train Model" button.
5.  **Model Testing:**
    *   Choose "Text Input" to enter your own test sentences, or "Predefined Examples" to use built-in examples.
    *   Click the "Test Model" button to see the model's predictions and evaluation metrics.  The test text will be displayed with the identified entities replaced by their labels in brackets (e.g., `[COMPANY]`).

## Project Structure
## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License
[MIT](https://choosealicense.com/licenses/mit/)
## Author

[Yazan Risheh](https://www.linkedin.com/in/yazan-risheh-8b87211a1/) | [GitHub](https://github.com/yazanrisheh?tab=repositories)