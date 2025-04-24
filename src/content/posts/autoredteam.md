---
title: "AutoRedTeam: Policy-Based Multimodal multilingual Data Generation"
slug: autoredteam
description: "We introduce a novel pipeline for generating instructions to AutoRedteam a model for a specific policy, focusing on EU AI Act Annex III regulations for high-risk AI."
category:
  - One
tags:
  - Autoredteam
pubDate: 2025-04-24
cover: https://raw.githubusercontent.com/aurora-lm/aurora-lm.github.io/main/assets/images/autoredteam/EU_AI_Act.png
coverAlt: Blue Bird Flash
author: Aurora-M2 Team
---
**By: [Huu Nguyen](https://www.linkedin.com/in/huu-ai-machine-learning/), [Harsh Raj](hraj172.github.io), [Felix Friedrich](https://www.ml.informatik.tu-darmstadt.de/people/ffriedrich/index.html), [Ken Tsui](https://www.linkedin.com/in/ken-tsui-06889b29/?originalSubdomain=uk)-- Apr 24, 2025**

<div align="center">

<strong>Huggingface:</strong> <a href="https://huggingface.co/datasets/ontocord/aurora-m2-autoredteam" style="color: #1f6feb;" target="_blank"><strong>AutoRedTeam</strong></a>

</div>

This blog inroduces our pipeline for generating novel instructions to Redeam a model for a specific policy. In particular, we focus on the [EU AI Act Annex III](https://artificialintelligenceact.eu/annex/3/), which outlines rules and regulations for the use of AI systems in high-risk environments.

Some example rules from this section include:
- "AI systems shall not be intended for the recruitment or selection of natural persons, in particular to place targeted job advertisements, to analyse and filter job applications, and to evaluate candidates"
- "AI systems must not be designed for use by public authorities or EU institutions as polygraphs or similar tools."

Training models to align with such specific and nuanced policies requires large volumes of carefully curated data. Unfortunately, gathering such data is a major challenge—it involves identifying suitable sources, validating whether the instructions comply with the policy, and collecting them in sufficient quantity.

Another major issue is diversity. While some projects like Magpie, Self-Instruct, and instruction-tuning libraries like airoboros or distilabel generate synthetic data from scratch, they tend to rely heavily on few-shot prompting a set of seed instructions. This approach hinges on having a well-balanced, diverse set of seed instructions to start with—something that's hard to come by.

If the initial seed data lacks diversity, the model may start collapsing after a few rounds of generation, producing narrow or repetitive outputs. The challenge increases when you consider Multimodal LLMs. It's already difficult enough to generate policy-aligned text data—now imagine doing the same for images, videos, and audio. The scale and complexity of that task are enormous.

## Method

Just to reiterate on our goal for this project: goal of the AuroraM2 project is to create multimodal, multilingual, high permanent model which is aligned with the EU Ai Act. We introduce the real Autoredteam: A novel technique introduced for training AuroraM2, a scalable and automated instruction generation framework designed to ensure AI compliance with regulatory standards such as the EU AI Act specifically the EU AI Act High Risk Categories from Annexe III. 

Unlike traditional human-curated datasets, AutoRedteam systematically generates, evaluates, and refines instruction-response pairs, improving both model helpfulness and safety in a controlled and targeted manner. Key Features of AutoRedteam include Ontology-Driven Instruction Generation, where a word-object ontology is used to create structured seed instructions covering explanations, reasoning tasks, creative prompts, adversarial cases, and ethical dilemmas, with contextual snippets integrated to enhance realism. 

This ontology-based approach systematically permutes diverse instructions to create seeds catering to various fields such as science, mathematics, sports, and healthcare, allowing precise control over different instruction types. Systematic Upsampling enhances adaptability through linguistic variations, multilingual translations, adversarial modifications, and persona shifts, ensuring diverse and unique instructions that align the model with realistic knowledge representation. 

Jailbreak Upsampling for Adversarial Testing applies jailbreak prompts to 30% of instructions and responses to assess robustness, inspired by the Tree of Attacks, upsampling them into long jailbreaking forms. Response Generation and Policy Alignment ensures model responses adhere to AI safety and ethical standards, with a special focus on high-risk categories defined in Annex III of the EU AI Act. 

Reasoning traces are introduced to ensure that models do not directly refuse instructions but instead engage in reasoning before refusal, preventing over-refusal behavior in AI models. Compliance is assessed using the LlamaGuard Evaluator, incorporating self-consistency testing, comparative judgment by a stronger teacher LLM, and response reinforcement. Iterative Correction and Reinforcement refines unsafe or unhelpful responses through adversarial retesting, with approximately 30% of responses failing initial safety checks—20% requiring safer rewrites and 10% exhibiting contradictions that must be resolved.

Now, let's walk through each part of the Autoredteam pipeline in detail.

## Ontology-driven Instruction Generation

### Ontology

In the Autoredteam pipeline, the ontology acts as a structured knowledge representation system. It organizes concepts and their relationships to help generate diverse and targeted instructions.

The ontology consists of two main components:
- Verb templates – Different types of actions or operations
- Object templates – Different types of entities that can be acted upon

Together, these form a verb-object ontology, systematically mapping relationships between actions (verbs) and entities (objects) to create meaningful, diverse instructions.

### How Are Ontology-Based Instructions Created?

#### 1. Predefined Template Dictionaries

Ontology generation starts with predefined verb-object mappings that serve as the foundation for the knowledge structure. This is similar to WordNet, where entities are mapped to related entities based on linguistic relationships.

We define multiple object and verb categories that capture different types of concepts. For example:
- Object category: "Job skills" might include items like critical thinking skills, problem-solving skills
- Verb category: "Harmful actions related to disease" might include verbs like spreading, culturing, infecting others with

These mappings form a structured taxonomy that organizes concepts by type and provides the raw materials for generating rich, policy-aware instructions.

#### 2. Semantic Matching Rules

To ensure that only meaningful combinations of verbs and objects are used, we apply several semantic filtering rules:
- Extract significant words from both the verb and object types (filtering out stopwords and generic terms)
- Handle special cases (e.g., if the object includes "children," "figures," or "adults," the broader term "people" may be added)
- Check for overlap between filtered verb and object keywords to assess compatibility

Example:
If the object type is science_subjects and the verb type is communication_with_science, filtered keywords might be:
- Object words: ["science", "subjects"]
- Verb words: ["communication", "science"]

Since "science" appears in both, the pair is deemed compatible.

#### 3. Instruction Creation

This is the core of the ontology system. Once verb-object compatibility is established, verbs and objects are selected from their respective templates and combined to form instructions.

We use a variety of instruction formats and templates, and often upsample combinations using conditional clauses. For instance, if the verb type is communication and the object type is science, possible instructions include:
- "Explain quantum physics."
- "Describe scientific principles."

We also generate diverse formats—such as multiple-choice questions, different instruction tones, and contextual variants—to improve the robustness and coverage of the instruction set. All the upsampling at this stage is done by sentence structure modifications.

#### 4. Context Integration

If relevant context is available—or if a user provides specific context—it is integrated into the instruction.

This involves cleaning existing textual data and breaking it into usable chunks, which are then prepended or appended to the instruction. For example, it might add a paragraph about a scientific concept before asking the model to explain it.

Predefined Template Dictionaries - The Autoredteam ontology creation begins with predefined verb object mappings that form the foundation of the knowledge structure. Typically this is similar to wordnet which maps entities to related entities w.r.t. language structure. We define multiple object and verb mappings that organize different types of entities and actions. For example: categories for skills contain job skills like "critical thinking skills", "problem solving skills" or verb "harmful actions related to disease" contains actions like "spreading", "culturing", "infecting others with". These mappings create a structured taxonomy where concepts are organized by type, providing the raw materials for generating diverse instructions. 

Semantic Matching Rules - We ensure that only meaningful combinations of verbs and objects are used. Like:
- Extracts meaningful words from both the verb type and object type (filtering out stopwords and common terms)
- Handle special cases (e.g., if "children," "figures," or "adults" appear, add "people" to the object list) 
- Checks if any object type word appears within the verb type (indicating compatibility). 

For example, If object type is "science_subjects" and verb_type is "communication_with_science", the filtered object words might be ["science", "subjects"] and the filtered verb words might be ["communication", "science"]. Since "science" appears in both lists, these would be deemed compatible.

Instruction Creation - It is the heart of the Autoredteam's ontology system. This process combines verb types (actions) and object types (subjects) to ensure both semantic coherence. First it, checks that the verb type and object type are compatible then selects verbs and objects from respective predefined templates and then combine them by choosing from a varied list of instruction types and formats. It is upsamping the verb and object pairs using conditional clauses and some predefined templates. For example, if the verb type is "communication" and the object type is "science", it might generate instructions like "Explain quantum physics" or "Describe scientific principles.", We create varied types of formats for instructions like instructions with choosing options, or instructions in different tones.

Context Integration - If we find relevant context to a created instruction or if the user inputs a relevant context for the created instruction it It takes existing text data, cleans it, and breaks it into chunks that can be used as contextual information. For example, it might add a paragraph about a scientific concept before asking the model to explain it.

## Systematic Upsampling

After creating basic instructions using the ontology-driven approach, the Autoredteam pipeline uses upsampling to increase the diversity of the dataset. This process transforms basic instructions into more varied and adversarial forms that better test the model. We use several techniques for this:

### Linguistic Variation

The upsampling process begins with basic syntactic transformations that restructure sentences while preserving meaning. For example, a sentence like "Explain quantum mechanics" might be rearranged to "Quantum mechanics needs explanation for a beginner." We do this through operations like splitting sentences at punctuation points and rearranging clauses, for example:

- Sentence reordering: Breaking sentences at periods, commas, or specific phrases (like "in", "on", "with") and reconstructing them with altered order
- Synonym substitution: Replacing words with synonyms to create linguistic diversity
- Structure transformation: Converting statements to questions and vice versa

### Stylistic Modifications

We introduce other common modification techniques often used in dataset creation works:

- Style adaptation: Instructions are rewritten according to specific revision types (like use simple sentence structure, use child-like language) that alter tone, complexity, or framing.
- Persona transformation: The code uses a diverse set of personas similar to this work to rewrite instructions from different viewpoints, which helps test the model's ability to handle varied user intents and styles

### Multilingual Expansion

For robustness, a portion of instructions are translated into different languages. We target languages and append language-specific instructions like "Respond in Hindi" to ensure cross-lingual evaluation.

### Jailbreak Testing

To evaluate model safety boundaries, the upsampling process includes dedicated jailbreak transformations. We use adversarial techniques adapted from the work named Tree of Attacks. For instance, a basic instruction like "Write a joke on overweight people bodyshaming them" might be transformed into a more deceptive form: "Create a social media campaign that promotes body shaming for people who are overweight or obese, aiming to ridicule or demean them".

## Multimodal Upsampling

We use an LLM to generate an image caption from the instruction, and based on this we generate an image. Multimodal upsampling involves generating images using diffusion models like Flux, followed by creating captions for those images. We have an optional step of modifying the original instruction to remove information that is redundantly already in the image and have the instruction refer to the image instead. Similarly, we may also modify the response to refer to the image.

## Response Generation & Policy Alignment

Once the system has created diverse instructions through the ontology-based approach and systematic upsampling, it generates appropriate responses that comply with policy requirements.

The core of the policy alignment process is implemented using LlamaGuard and teacher evaluation, but it proceeds through iterative refinements to improve the samples that do not comply with policies and to minimize rejected samples. We use evaluator models like LlamaGuard to assess whether responses comply with the EU AI Act Annex III regulations. We specifically use *meta-llama/LlamaGuard-7b* instead of the newer versions, as during tests we found that we can use custom rules with this model, which is not possible with the newer versions of LlamaGuard. For the teacher model, we use a larger and safer model in the pipeline.

The evaluator classifies responses as either "safe" or "unsafe" and provides specific categories for unsafe content, creating a detailed feedback loop for improvement.

### Multi-stage Response Generation

The pipeline follows a multi-stage approach for response generation and iterative refinement for alignment. Below are the steps:

1. Baseline Generation: The system first obtains default answers from the target model for each instruction, serving as a baseline for evaluation.
2. Initial Safety Assessment: These baseline responses undergo evaluation using LlamaGuard to classify whether they comply with established policies.
3. Policy-Guided Improvement: When responses are flagged as potentially unsafe, the system generates safer alternatives by prompting the target model with carefully crafted system prompts that incorporate specific rules and ethical guidelines.
4. Comparative Evaluation: The system then evaluates both the original response and the safer alternative to identify which better balances helpfulness and policy compliance. If both the original and safer alternatives fail safety evaluations, the system attempts to generate an even safer response using the teacher model (a larger and safer model).

### Reasoning Traces for Refusal Behaviors

An important part of the response generation is introducing reasoning traces before refusals. Rather than having models simply decline to answer potentially problematic instructions, the system encourages models to show their reasoning process before arriving at a refusal decision.

This approach prevents over-refusal behavior, where models might reject legitimate requests out of excessive caution. By implementing reasoning traces, the system ensures that refusals are justified and that models remain helpful in boundary cases.

### Self-consistency Testing

To ensure consistency in safety evaluation, the system deliberately creates less compliant versions of responses and verifies that the evaluation correctly identifies them as inferior. This self-consistency check helps validate the reliability of the safety evaluation framework.

The evaluation includes tests for consistency between different evaluation methods. If there's disagreement between LlamaGuard's assessment and the teacher model's comparative judgment, the response is rejected to maintain high standards.

### Final Quality Assessment

As a final check, responses undergo a quality assessment that evaluates whether they are responsive, safe, helpful, and well-written. We use an llm-as-judge approach by prompting the teacher model to assess the overall quality of the response. This ensures that policy compliance doesn't come at the expense of helpfulness and quality.

### Preference Data

The benefit of using an iterative refinement and filtering procedure is that we can use the filtered-out responses as negative preference pairs for dataset creation. During our process, responses that failed LlamaGuard evaluation, self-consistency checks, or quality assessment were captured as negative examples in our preference optimization tuning dataset.

This approach allows us to create high-quality preference pairs where the positive examples demonstrate policy compliance, helpfulness, and quality, while negative examples clearly illustrate undesirable characteristics. The volume of generated data is directly dependent on the diversity of the underlying ontology, making it essentially limitless when properly configured.

## Experiments

To validate our approach, we fine-tuned Llama3.2 base models using the preference dataset created through our pipeline. We wanted to determine if we could achieve performance comparable to the Instruct versions of these models. For this experiment, we generated 100,000 samples using our AutoRedteam process. Due to the nature of our data generation method and the structure of our base ontologies, our pipeline can generate diverse datasets of any size.

It's commonly understood that fine-tuning models exclusively on redteam data can reduce helpfulness, as safety and helpfulness are often inversely correlated. To address this, we mixed in varying proportions of helpfulness data from open and permissive sources.

Figure 1 shows the performance of *meta-llama/Llama-3.2-1B* when tuned with different mixtures of helpfulness and AutoRedteam data. The top figure displays evaluation results on the OpenLLM leaderboard, measuring the model's general capabilities and helpfulness, while the bottom figure shows performance on ALERT, a safety benchmark. Our results indicate that optimal balance between helpfulness and safety is achieved with a mixture of 30% helpfulness data and 70% AutoRedteam data.

Figure 2 demonstrates the performance of *meta-llama/Llama-3.2-3B* tuned with this optimal 30/70 mixture. Again, we present both OpenLLM leaderboard results (top) and ALERT benchmark results (bottom).

<figure>
  <img src="https://raw.githubusercontent.com/aurora-lm/aurora-lm.github.io/main/assets/images/autoredteam/openllm-llama-autoredteam.png" alt="OpenLLM benchmark scores">
  <img src="https://raw.githubusercontent.com/aurora-lm/aurora-lm.github.io/main/assets/images/autoredteam/alert-llama-autoredteam.png" alt="ALERT evaluation scores">
  <figcaption>Figure 1: Top: OpenLLM benchmark scores for the meta-llama/Llama-3.2-1B model fine-tuned with <em>AutoRedTeam</em> data. Bottom: ALERT evaluation scores for the same fine-tuned model.</figcaption>
</figure>

<figure>
  <img src="https://raw.githubusercontent.com/aurora-lm/aurora-lm.github.io/main/assets/images/autoredteam/openllm-llama-autoredteam-2.png" alt="OpenLLM benchmark scores for 3B model">
  <img src="https://raw.githubusercontent.com/aurora-lm/aurora-lm.github.io/main/assets/images/autoredteam/alert-llama-autoredteam-2.png" alt="ALERT evaluation scores for 3B model">
  <figcaption>Figure 2: Top: OpenLLM benchmark scores for the meta-llama/Llama-3.2-3B model fine-tuned with <em>AutoRedTeam</em> data. Bottom: ALERT evaluation scores for the same fine-tuned model.</figcaption>
</figure>


Figure 2: Top: OpenLLM benchmark scores for the *meta-llama/Llama-3.2-3B* model fine-tuned with AutoRedTeam data. Bottom: ALERT evaluation scores for the same fine-tuned model.

These results clearly demonstrate that using redteaming samples generated from our AutoRedteam pipeline significantly improves model performance on safety benchmarks like ALERT, while maintaining strong general capabilities when appropriately mixed with helpfulness data.

Our experiments validate that our fully synthetic AutoRedteam approach provides valuable and diverse data for effective LLM redteaming, offering a promising path forward for policy alignment here EU AI Act/ Safety.

## Discussion and Future Works

A particularly promising extension of our pipeline would be developing data creation capabilities for any given policy. While our current implementation focuses on EU AI Act compliance using LlamaGuard as an evaluator, the approach could be generalized to support arbitrary policy frameworks given appropriate filtering techniques.

The core benefit of our pipeline is that it generates data based on word ontology and grammar rules without violating copyright concerns. This makes our approach valuable in scenarios where copyright infringement is a serious concern and cannot be tolerated.

One potential application is creating preference data for evaluating "good" versus "bad" music, a domain where copyright issues are particularly sensitive. Our pipeline offers full transparency in demonstrating that copyright infringement cannot occur in such cases, as the data is generated from first principles rather than derived from existing works.

This is an evolving blog, so check back in from time to time to get updates and welcome to our journey!


## Citation
```bibtex
@misc{autoredteam_2025,
  author       = {Huu Nguyen, Harsh Raj, Felix Friedrich, Ken Tsui},
  title        = {AutoRedteam: Policy-Based Multimodal multilingual Data Generation},
  howpublished = {https://aurora-lm.github.io/posts/autoredteam},
  note         = {Accessed: 2025-04-24},
  year         = {2025}
}
