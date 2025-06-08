from graph import get_graph
from prompts import (
    few_shot_prompt,
    zero_shot_prompt,
    meta_prompt,
    generated_knowledge_prompt,
    chain_of_thought_prompt,
    self_consistency_prompt
)

few_shot_prompt_graph = get_graph(few_shot_prompt)
zero_shot_prompt_graph = get_graph(zero_shot_prompt)
meta_prompt_graph = get_graph(meta_prompt)
generated_knowledge_prompt_graph = get_graph(generated_knowledge_prompt, with_tools=False)
chain_of_thought_prompt_graph = get_graph(chain_of_thought_prompt, with_tools=False)
self_consistency_prompt_graph = get_graph(self_consistency_prompt, with_tools=False, delete_messages=False)
