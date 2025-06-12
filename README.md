# Prompt Engineering Techniques

A comprehensive implementation of various prompt engineering techniques using LangChain and LangGraph with Azure OpenAI.

## Overview

This repository demonstrates six different prompt engineering techniques implemented as graph-based agents:

1. **Zero-Shot Prompting**: Simple instruction-based prompting without examples
2. **Few-Shot Learning**: Providing examples to guide model behavior
3. **Meta-Prompting**: Instructions about how to approach different types of problems
4. **Generated Knowledge**: Having the model generate relevant knowledge before answering
5. **Chain-of-Thought**: Step-by-step reasoning through complex problems
6. **Self-Consistency**: Multiple reasoning paths to improve reliability

Each technique is implemented as a working LangGraph agent with configurable parameters for tools usage and message history management.

## Project Structure

- `graph.py`: Core implementation of the StateGraph agent architecture
- `prompts.py`: Definitions of the six different prompting techniques
- `all_graphs.py`: Factory module that creates instances of each graph
- `tools.py`: Tool implementations (e.g., weather forecasting)

## Getting Started

### Prerequisites

- Python 3.13+
- Azure OpenAI API access

### Installation

```bash
# Clone the repository
git clone https://github.com/ksankaran/prompt-techniques.git
cd prompt-techniques

# Install dependencies
pip install -e .
```

### Environment Setup

Create a `.env` file with your Azure OpenAI credentials:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

```python
from all_graphs import (
    zero_shot_prompt_graph,
    few_shot_prompt_graph,
    meta_prompt_graph,
    generated_knowledge_prompt_graph,
    chain_of_thought_prompt_graph,
    self_consistency_prompt_graph
)

# Example: Use the few-shot prompt graph
response = few_shot_prompt_graph.invoke({
    "messages": ["What's the weather like in New York?"]
})
print(response)
```

## Technique Highlights

### Zero-Shot Prompting

The most basic technique that relies on the model's pre-trained knowledge without examples.

### Few-Shot Learning

Improves performance by demonstrating the expected pattern of interaction, particularly useful for tool usage as shown in the weather forecast example.

### Meta-Prompting

Provides structured problem-solving instructions that guide the model's thinking process across different problem types.

### Generated Knowledge

Explicitly asks the model to generate relevant knowledge before answering, improving reliability for domain-specific questions.

### Chain-of-Thought

Guides the model through step-by-step reasoning for mathematical and logical problems, as demonstrated in the mileage calculation examples.

### Self-Consistency

Extends chain-of-thought by generating multiple reasoning paths and finding consensus, improving reliability for complex problems.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.