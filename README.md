# LLM Thesis: Automated Assignment Question Generation and Evaluation

This repository contains a comprehensive pipeline for generating, improving, and evaluating assignment questions using Large Language Models (LLMs). The project demonstrates the complete workflow from raw PDF data extraction to fine-tuned model deployment and evaluation, with integrated data annotation using Argilla.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workflow Steps](#workflow-steps)
- [Usage Instructions](#usage-instructions)
- [Evaluation Framework](#evaluation-framework)
- [Argilla Integration](#argilla-integration)
- [Results and Outputs](#results-and-outputs)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements an end-to-end system for:
1. **Data Extraction**: Extracting text from PDF documents
2. **Question Generation**: Using LLMs to generate assignment questions
3. **Data Cleaning**: Improving and structuring the generated questions
4. **Topic Classification**: Assigning topics and learning outcomes
5. **Model Fine-tuning**: Training custom models on the generated data
6. **Performance Comparison**: Comparing base vs fine-tuned models
7. **Code Evaluation**: Automated code generation and testing
8. **Data Annotation**: Using Argilla for human-in-the-loop evaluation

## 📁 Project Structure

```
code/
├── 1_HEC_togath_llma3_2_3b_int_turbo.ipynb     # Initial data extraction and question generation
├── 2_cleaner_Local.ipynb                        # Data cleaning and preprocessing
├── 3_DataImprover.ipynb                         # Question improvement and PDF generation
├── 4_HEC_togh_attach_Topics.ipynb              # Topic and CLO assignment
├── 5_SGenMistral_Nemo_Base_2407_bnb_finetuning_Working.ipynb  # Model fine-tuning
├── 6_NormalVSFinetune_Compare.ipynb            # Performance comparison
├── 7_CodeEvalAgent.ipynb                       # Code evaluation agent
├── argilla_annotation/                          # Argilla data annotation setup
│   ├── annotation_interface.py                 # Custom annotation interface
│   ├── dataset_preparation.py                  # Dataset preparation for annotation
│   └── evaluation_metrics.py                   # Evaluation metrics calculation
└── humaneval/                                   # Evaluation framework (Next.js app)
    ├── src/
    ├── public/
    └── package.json
```

## 🔧 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (for fine-tuning)
- Google Colab Pro (recommended for GPU access)
- API Keys:
  - Together AI API Key
  - OpenAI API Key (for code evaluation)
- Argilla Server (for data annotation)

## 📦 Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd LLM-Thesis/code

# Install required packages
pip install together pymupdf reportlab
pip install bitsandbytes accelerate xformers peft trl triton
pip install unsloth sentencepiece protobuf datasets huggingface_hub
pip install openai gradio fpdf PyPDF2

# Install Argilla for data annotation
pip install argilla[server]
```

### 2. API Key Configuration

Set up your API keys in Google Colab:

```python
# For Together AI
from google.colab import userdata
TOGETHER_API_KEY = userdata.get('TOGETHER_API_KEY')

# For OpenAI
OPENAI_API_KEY = userdata.get('OPENAI')

# For Argilla
import argilla as rg
rg.init(api_key="your_argilla_api_key", api_url="http://localhost:6900")
```

### 3. Argilla Server Setup

```bash
# Start Argilla server
argilla server start

# Or using Docker
docker run -d --name argilla -p 6900:6900 argilla/argilla-server:latest
```

## 🔄 Workflow Steps

### Step 1: Data Extraction and Initial Question Generation
**File**: `1_HEC_togath_llma3_2_3b_int_turbo.ipynb`

**Purpose**: Extract text from PDF documents and generate initial assignment questions using Llama-3.2-3B-Instruct-Turbo.

**Key Features**:
- PDF text extraction using PyMuPDF
- Text chunking for manageable processing
- Question generation with structured prompts
- JSON output formatting

**Usage**:
```python
# Configure parameters
PDF_PATH = "DataRaw/AI_Russell_Norvig.pdf"
CHUNK_SIZE = 2000
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

# Extract and generate
text_chunks = extract_text_from_pdf(PDF_PATH, CHUNK_SIZE)
generate_assignment_questions(text_chunks)
```

### Step 2: Data Cleaning and Preprocessing
**File**: `2_cleaner_Local.ipynb`

**Purpose**: Clean and format the generated questions by removing unwanted formatting and improving readability.

**Key Features**:
- Regex-based text cleaning
- Removal of markdown formatting
- Structured output preservation

**Usage**:
```python
# Clean questions
pattern = r"\*\*.*?\*\*|\n\n"
# Process and save cleaned data
```

### Step 3: Question Improvement and PDF Generation
**File**: `3_DataImprover.ipynb`

**Purpose**: Enhance question quality using few-shot prompting and generate professional PDF assignments.

**Key Features**:
- Few-shot prompting for better question structure
- Professional PDF generation using ReportLab
- Structured assignment formatting

**Usage**:
```python
# Improve questions
improved_questions = generate_assignment_questions(input_text)

# Generate PDF
markdown_to_pdf(improved_text, "assignment.pdf")
```

### Step 4: Topic and CLO Assignment
**File**: `4_HEC_togh_attach_Topics.ipynb`

**Purpose**: Automatically assign topics and Course Learning Outcomes (CLOs) to generated questions.

**Key Features**:
- Automated topic classification
- CLO assignment based on question content
- Dataset preparation for fine-tuning

**Usage**:
```python
# Process questions and assign topics/CLOs
for question in questions:
    topic, clo = assign_topic_and_clo(question)
    # Save to JSON format
```

### Step 5: Model Fine-tuning
**File**: `5_SGenMistral_Nemo_Base_2407_bnb_finetuning_Working.ipynb`

**Purpose**: Fine-tune Mistral-Nemo-Base-2407 model on the generated question dataset using Unsloth framework.

**Key Features**:
- 4-bit quantization for memory efficiency
- LoRA adapters for parameter-efficient fine-tuning
- Alpaca format dataset preparation
- Gradient checkpointing for long sequences

**Usage**:
```python
# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Prepare dataset and train
```

### Step 6: Performance Comparison
**File**: `6_NormalVSFinetune_Compare.ipynb`

**Purpose**: Compare the performance of base models vs fine-tuned models on question generation tasks.

**Key Features**:
- Automated response generation
- CSV and PDF output generation
- Performance metrics comparison
- Batch processing capabilities

**Usage**:
```python
# Compare models
base_responses = generate_with_base_model(prompts)
finetuned_responses = generate_with_finetuned_model(prompts)

# Save results
save_comparison_results(base_responses, finetuned_responses)
```

### Step 7: Code Evaluation Agent
**File**: `7_CodeEvalAgent.ipynb`

**Purpose**: Automated code generation, execution, and evaluation using OpenAI's GPT-4.

**Key Features**:
- Automated code generation
- Safe code execution
- Error feedback and refinement
- Test case generation and validation

**Usage**:
```python
# Generate and test code
query = "Write a function to calculate factorial"
auto_chain_agent(query, max_retries=3)
```

## 🏷️ Argilla Integration

### Data Annotation Workflow

Argilla is integrated throughout the pipeline for human-in-the-loop evaluation and data quality assurance.

#### Setup Argilla Dataset

```python
import argilla as rg
from argilla import TextClassificationRecord

# Create dataset for question quality evaluation
def create_annotation_dataset(questions):
    records = []
    for i, question in enumerate(questions):
        record = TextClassificationRecord(
            text=question["Question"],
            metadata={
                "topic": question["Topic"],
                "difficulty": question["Difficulty"],
                "type": question["Type"],
                "clo": question["CLO"]
            },
            annotation=None,  # Will be filled by annotators
            id=i
        )
        records.append(record)
    
    # Push to Argilla
    dataset = rg.DatasetForTextClassification(records)
    rg.log(dataset, name="question_quality_evaluation")
```

#### Custom Annotation Interface

```python
# argilla_annotation/annotation_interface.py
import argilla as rg
from argilla import TextClassificationRecord

class QuestionQualityAnnotator:
    def __init__(self, dataset_name="question_quality_evaluation"):
        self.dataset_name = dataset_name
        
    def create_annotation_task(self, questions):
        """Create annotation task for question quality evaluation"""
        records = []
        for question in questions:
            record = TextClassificationRecord(
                text=question["Question"],
                metadata={
                    "topic": question["Topic"],
                    "difficulty": question["Difficulty"],
                    "type": question["Type"]
                },
                annotation=None
            )
            records.append(record)
        
        dataset = rg.DatasetForTextClassification(records)
        rg.log(dataset, name=self.dataset_name)
        
    def get_annotated_data(self):
        """Retrieve annotated data from Argilla"""
        dataset = rg.load(self.dataset_name)
        return dataset
        
    def calculate_quality_metrics(self):
        """Calculate quality metrics from annotations"""
        dataset = self.get_annotated_data()
        # Calculate metrics like agreement, quality scores, etc.
        return metrics
```

#### Evaluation Metrics

```python
# argilla_annotation/evaluation_metrics.py
import pandas as pd
import numpy as np

def calculate_annotation_metrics(annotated_dataset):
    """Calculate various metrics from Argilla annotations"""
    
    # Inter-annotator agreement
    agreement_scores = calculate_agreement(annotated_dataset)
    
    # Quality distribution
    quality_distribution = calculate_quality_distribution(annotated_dataset)
    
    # Topic-wise quality analysis
    topic_quality = analyze_topic_quality(annotated_dataset)
    
    return {
        "agreement_scores": agreement_scores,
        "quality_distribution": quality_distribution,
        "topic_quality": topic_quality
    }

def create_evaluation_report(metrics):
    """Generate comprehensive evaluation report"""
    report = {
        "summary": {
            "total_questions": len(metrics["quality_distribution"]),
            "average_quality_score": np.mean(metrics["quality_distribution"]),
            "agreement_rate": metrics["agreement_scores"]["overall"]
        },
        "detailed_analysis": metrics
    }
    return report
```

### Integration Points

1. **Post-Generation Quality Check**: After Step 1, use Argilla to evaluate initial question quality
2. **Pre-Fine-tuning Validation**: Before Step 5, validate dataset quality through human annotation
3. **Post-Fine-tuning Evaluation**: After Step 6, compare human evaluations of base vs fine-tuned outputs
4. **Continuous Improvement**: Use Argilla feedback to improve prompts and model performance

## 🎯 Evaluation Framework

### HumanEval Dashboard
**Location**: `humaneval/`

A Next.js web application for evaluating and visualizing the generated questions and model performance.

**Setup**:
```bash
cd humaneval
npm install
npm run dev
```

**Features**:
- Interactive question evaluation interface
- Performance metrics visualization
- User feedback collection
- Export capabilities
- Integration with Argilla annotations

### Argilla Dashboard
**Access**: `http://localhost:6900`

**Features**:
- Custom annotation interfaces
- Quality control workflows
- Inter-annotator agreement analysis
- Export annotated datasets
- Integration with Hugging Face datasets

## 📊 Results and Outputs

### Generated Files
- `assignment_questions.json` - Raw generated questions
- `cleaned_questions-Removed-Pres.json` - Cleaned questions
- `updated_questions.json` - Questions with topics and CLOs
- `formatted_question_generation_dataset.json` - Fine-tuning dataset
- `together_ai_responses.pdf/csv` - Comparison results
- `generated_code.py` - Generated code examples
- `argilla_annotations.json` - Human annotations and quality scores

### Performance Metrics
- Question quality scores (human-annotated)
- Model response accuracy
- Code execution success rates
- Fine-tuning efficiency metrics
- Inter-annotator agreement rates
- Topic-wise quality analysis

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable 4-bit quantization

2. **API Rate Limits**
   - Implement delays between requests
   - Use batch processing
   - Monitor API usage

3. **Model Loading Issues**
   - Check GPU memory availability
   - Verify model compatibility
   - Update dependencies

4. **Argilla Connection Issues**
   - Verify server is running on correct port
   - Check API key configuration
   - Ensure network connectivity

### Performance Optimization

1. **Memory Management**
   - Use 4-bit quantization
   - Enable gradient checkpointing
   - Implement proper cleanup

2. **Speed Optimization**
   - Use Unsloth framework
   - Implement batch processing
   - Optimize data loading

3. **Annotation Efficiency**
   - Use pre-annotation with model predictions
   - Implement active learning strategies
   - Batch annotation tasks

## 📝 Usage Instructions

### Quick Start

1. **Setup Environment**:
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Set API keys
   export TOGETHER_API_KEY="your_key_here"
   export OPENAI_API_KEY="your_key_here"
   
   # Start Argilla server
   argilla server start
   ```

2. **Run Complete Pipeline**:
   ```bash
   # Execute notebooks in order
   jupyter notebook 1_HEC_togath_llma3_2_3b_int_turbo.ipynb
   jupyter notebook 2_cleaner_Local.ipynb
   jupyter notebook 3_DataImprover.ipynb
   jupyter notebook 4_HEC_togh_attach_Topics.ipynb
   
   # Setup Argilla annotation
   python argilla_annotation/dataset_preparation.py
   
   # Continue with fine-tuning
   jupyter notebook 5_SGenMistral_Nemo_Base_2407_bnb_finetuning_Working.ipynb
   jupyter notebook 6_NormalVSFinetune_Compare.ipynb
   jupyter notebook 7_CodeEvalAgent.ipynb
   ```

3. **Evaluate Results**:
   ```bash
   # Access Argilla dashboard
   # Open http://localhost:6900
   
   # Access HumanEval dashboard
   cd humaneval
   npm run dev
   # Open http://localhost:3000
   ```

### Customization

- **Model Selection**: Modify `MODEL_NAME` variables in notebooks
- **Dataset Size**: Adjust `trainSize` parameters for different dataset splits
- **Question Types**: Modify prompts for different question categories
- **Evaluation Metrics**: Add custom metrics in evaluation notebooks
- **Annotation Workflows**: Customize Argilla interfaces for specific evaluation needs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Together AI for providing the LLM API
- Unsloth for the efficient fine-tuning framework
- OpenAI for the code evaluation capabilities
- Hugging Face for the model hosting and datasets
- Argilla for the data annotation and evaluation platform

---

**Note**: This project is part of an academic thesis on LLM-based assignment question generation. For research purposes, please cite appropriately. 