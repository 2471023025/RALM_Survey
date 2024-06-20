# RALM_Survey
This is a repository of RALM surveys containing a summary of state-of-the-art RAG and other technologies according to according to our survey paper: [RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing](https://arxiv.org/abs/2404.19543v1) . In this repository, we will present the most central research approach of our thesis as well as keep up-to-date with work on RALM in the most accessible way possible. For more detailed information, please read our papers. Please cite our papers if you think they can help you with your research!

## News
This project is under development. You can hit the **STAR** and **WATCH** to follow the updates.
* Our survey：[RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing](https://arxiv.org/abs/2404.19543v1) on RALM is now public.

Citation Information:
```
@article{hu2024rag,
  title={RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing},
  author={Hu, Yucheng and Lu, Yuxing},
  journal={arXiv preprint arXiv:2404.19543},
  year={2024}
}
```

## Table of Contents
- [RALM_Survey](#ralm_survey)
  - [News](#news)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Related Surveys](#related-surveys)
  - [Definition](#definition)
    - [Sequential Single Interaction](#sequential-single-interaction)
    - [Sequential Multiple Interactions](#sequential-multiple-interactions)
    - [Parallel Interaction](#parallel-interaction)
  - [Retriever](#llm-augmented-kgs)
    - [Sparse Retrieval](#spars-retrieval)
    - [Dense Retrieval](#dense-retrieval)
    - [Internet Retrieval](#internet-retrieval)
    - [Hybrid Retrieval](#hybrid-retrieval)
  - [Language Models](#language-models)
    - [AutoEncoder Language Model](#autoencoder-language-model)
    - [AutoRegressive Language Model](#autoregressive-language-model)
    - [Encoder-Decoder Language Model](#encoder-decoder-language-model)
  - [RALM Enhancement](#ralm-enhancement)
    - [Retriever Enhancement](#retriever-enhancement)
    - [LM Enhancement](#lm-enhancement)
    - [Overall Enhancement](#overall-enhancement)
  - [Data Source](#data-source)
    - [Structured Data](#structured-data)
    - [Unstructured Data](#unstructured-data)
  - [Applications](#applications)
    - [RALM on NLG](#ralm-on-nlg)
    - [RALM on NLU](#ralm-on-nlu)
    - [RALM on Both NLU and NLG](#{ralm-on-both-nlu-and-nlu)
  - [Evaluation](#evaluation)
  
## Overview
This SURVEY of ours summarizes multiple aspects of RALM, including: definition, retriever, LM, enhancement, data source, application, evaluation, and more.

We hope this repository can help researchers and practitioners to get a better understanding of RALM.

<img src="/fig/fig1.png" width = "800" />

## Related Surveys
- Retrieval-Augmented Generation for AI-Generated Content: A Survey(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2402.19473.pdf)
- A Survey on Retrieval-Augmented Text Generation(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2202.01110.pdf)
- Retrieving Multimodal Information for Augmented Generation: A Survey(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2303.10868.pdf)
- Retrieval-Augmented Generation for Large Language Models: A Survey(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2312.10997.pdf)


## Definition
<img src="/fig/fig2.png" width = "800" />
<img src="/fig/fig3.png" width = "800" />

### Sequential Single Interaction
- Corrective Retrieval Augmented Generation(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2401.15884.pdf)
- SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.11511.pdf)
- Atlas: Few-shot Learning with Retrieval Augmented Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2208.03299.pdf)
- Efficient Retrieval Augmented Generationfrom Unstructured Knowledge for Task-Oriented Dialog(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2102.04643.pdf)
- FeB4RAG: Evaluating Federated Search in the Context of Retrieval Augmented Generation(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2402.11891.pdf)
- FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation(acm, 2023)[[paper]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591687)
- Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering(mit, 2024)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00530/114590)
- End-to-End Training of Neural Retrievers for Open-Domain Question Answering(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2101.00408.pdf)
- REALM: Retrieval-Augmented Language Model Pre-Training(mlr, 2020)[[paper]](http://proceedings.mlr.press/v119/guu20a/guu20a.pdf)
- In-Context Retrieval-Augmented Language Models(mit, 2023)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00605/118118)
- Learning to Filter Context for Retrieval-Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2311.08377.pdf)
- MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2210.02928.pdf)
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(neurips, 2020)[[paper]](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
- Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2007.01282.pdf)
- Improving Language Models by Retrieving from Trillions of Tokens(mlr, 2022)[[paper]](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf)
- When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2212.10511.pdf)
- Check Your Facts and Try Again: Improving Large Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2302.12813.pdf)
- RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01352.pdf)
- SAIL: Search-Augmented Instruction Learning(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.15225.pdf)
- MAKING RETRIEVAL-AUGMENTED LANGUAGE MODELS ROBUST TO IRRELEVANT CONTEXT(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01558.pdf)
- RECOMP: IMPROVING RETRIEVAL-AUGMENTED LMS WITH COMPRESSION AND SELECTIVE AUGMENTATION(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.04408.pdf)
- Latent Retrieval for Weakly Supervised Open Domain Question Answering(Arxiv, 2019)[[paper]](https://arxiv.org/pdf/1906.00300.pdf)
- End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering(neurips, 2021)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/da3fde159d754a2555eaa198d2d105b2-Paper.pdf)
- DISTILLING KNOWLEDGE FROM READER TO RETRIEVER FOR QUESTION ANSWERING(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2012.04584.pdf)
- REPLUG: Retrieval-Augmented Black-Box Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2301.12652.pdf)
- REVEAL: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory(thecvf, 2023)[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.pdf)
- Neural Argument Generation Augmented with Externally Retrieved Evidence(Arxiv, 2018)[[paper]](https://arxiv.org/pdf/1805.10254.pdf)
  
### Sequential Multiple Interactions
- Active Retrieval Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.06983.pdf)
- Rethinking with Retrieval: Faithful Large Language Model Inference(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2301.00303.pdf)
- DEMONSTRATE–SEARCH–PREDICT:Composing retrieval and language models for knowledge-intensive NLP(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2212.14024.pdf)
- Improving Language Models via Plug-and-Play Retrieval Feedback(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.14002.pdf)
- Retrieval Augmentation Reduces Hallucination in Conversation(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2104.07567.pdf)
- WikiChat: Stopping the Hallucination of Large Language Model Chatbots by Few-Shot Grounding on Wikipedia (Findings of EMNLP 2023) [[paper]](https://arxiv.org/abs/2305.14292) [[code]](https://github.com/stanford-oval/WikiChat) [[demo]](https://wikichat.genie.stanford.edu/)

### Parallel Interaction
- GENERALIZATION THROUGH MEMORIZATION:NEAREST NEIGHBOR LANGUAGE MODELS(Arxiv, 2020)[[paper]](https://arxiv.org/pdf/1911.00172.pdf)
- Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval(mlr, 2022)[[paper]](https://proceedings.mlr.press/v162/alon22a/alon22a.pdf)
- Efficient Nearest Neighbor Language Models(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2109.04212.pdf)
- You can’t pick your neighbors, or can you? When and how to rely on retrieval in the kNN-LM(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2210.15859.pdf)

## Retriever
<img src="/fig/fig4.png" width = "800" />

### Sparse Retrieval
- Okapi at TREC-3(google, 1995)[[paper]](https://books.google.com/books?hl=en&lr=&id=j-NeLkWNpMoC&oi=fnd&pg=PA109&dq=Okapi+at+TREC-3&ots=YkB0HpzoLG&sig=I6F49LQ5R5bbnEYNZr-LR1Hs4f0#v=onepage&q=Okapi%20at%20TREC-3&f=false)
- Learning to retrieve passages without supervision(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2112.07708.pdf)
- Generation-Augmented Retrieval for Open-Domain Question Answering(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2009.08553.pdf)
- GENERALIZATION THROUGH MEMORIZATION:NEAREST NEIGHBOR LANGUAGE MODELS(Arxiv, 2020)[[paper]](https://arxiv.org/pdf/1911.00172.pdf)
- Adaptive Semiparametric Language Models(mit, 2021)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00371/100688)
- MemPrompt: Memory-assisted Prompt Editing with User Feedback(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2201.06009.pdf)

### Dense Retrieval
- Unsupervised Dense Information Retrieval with Contrastive Learning(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2112.09118.pdf)
- Large Dual Encoders Are Generalizable Retrievers(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2112.07899.pdf)
- ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2112.01488.pdf)
- How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2302.07452.pdf)
- Dense Passage Retrieval for Open-Domain Question Answering(Arxiv, 2020)[[paper]](https://arxiv.org/pdf/2004.04906.pdf)
- REPLUG: Retrieval-Augmented Black-Box Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2301.12652.pdf)
- End-to-End Training of Neural Retrievers for Open-Domain Question Answering(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2101.00408.pdf)
- REALM: Retrieval-Augmented Language Model Pre-Training(mlr, 2020)[[paper]](http://proceedings.mlr.press/v119/guu20a/guu20a.pdf)
- Latent Retrieval for Weakly Supervised Open Domain Question Answering(Arxiv, 2019)[[paper]](https://arxiv.org/pdf/1906.00300.pdf)
- End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering(neurips, 2021)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/da3fde159d754a2555eaa198d2d105b2-Paper.pdf)
- MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2210.02928.pdf)
- RE-IMAGEN: RETRIEVAL-AUGMENTED TEXT-TO-IMAGE GENERATOR(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2209.14491.pdf)
- MEMORY-DRIVEN TEXT-TO-IMAGE GENERATION(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2208.07022.pdf)
- Retrieval-Augmented Diffusion Models(neurips, 2022)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/62868cc2fc1eb5cdf321d05b4b88510c-Paper-Conference.pdf)

### Internet Retrieval
- Active Retrieval Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.06983.pdf)
- MAKING RETRIEVAL-AUGMENTED LANGUAGE MODELS ROBUST TO IRRELEVANT CONTEXT(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01558.pdf)
- Internet-augmented dialogue generation(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2107.07566.pdf)
- Webgpt: Browser-assisted question-answering with human feedback(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2112.09332.pdf)

### Hybrid Retrieval
- SAIL: Search-Augmented Instruction Learning(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.15225.pdf)
- Internet-augmented language models through few-shot prompting for open-domain question answering(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2203.05115.pdf)
- REVEAL: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory(thecvf, 2023)[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_REVEAL_Retrieval-Augmented_Visual-Language_Pre-Training_With_Multi-Source_Multimodal_Knowledge_Memory_CVPR_2023_paper.pdf)
- Neural Argument Generation Augmented with Externally Retrieved Evidence(Arxiv, 2018)[[paper]](https://arxiv.org/pdf/1805.10254.pdf)
- Boosting search engines with interactive agents(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2109.00527.pdf)
- Off the beaten path: Let’s replace term-based retrieval with k-NN search(Arxiv, 2016)[[paper]](https://arxiv.org/pdf/1610.10001.pdf)

## Language Models
<img src="/fig/fig5.png" width = "800" />

### AutoEncoder Language Model
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(Arxiv, 2019)[[paper]](https://arxiv.org/pdf/1810.04805.pdf)
- RoBERTa: A Robustly Optimized BERT Pretraining Approach(Arxiv, 2019)[[paper]](https://arxiv.org/pdf/1907.11692.pdf)
- DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter(Arxiv, 2020)[[paper]](https://arxiv.org/pdf/1910.01108.pdf)
- ConvBERT: Improving BERT with Span-based Dynamic Convolution(neurips, 2020)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2020/file/96da2f590cd7246bbde0051047b0d6f7-Paper.pdf)

### AutoRegressive Language Model
- Llama 2: Open Foundation and Fine-Tuned Chat Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2307.09288.pdf)
- GPT-4 Technical Report(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2303.08774.pdf)
- GPT-NeoX-20B: An Open-Source Autoregressive Language Model(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2204.06745.pdf)
- OPT: Open Pre-trained Transformer Language Models(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2205.01068.pdf)
- LLaMA: Open and Efficient Foundation Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2302.13971.pdf)
- Few-shot Learning with Multilingual Generative Language Models(aclanthology, 2022)[[paper]](https://aclanthology.org/2022.emnlp-main.616.pdf)
- QWEN TECHNICAL REPORT(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2309.16609.pdf)
- Language Models are Unsupervised Multitask Learners(amazonaws, 2019)[[paper]](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)
- ADAPTIVE INPUT REPRESENTATIONS FOR NEURAL LANGUAGE MODELING(Arxiv, 2019)[[paper]](https://arxiv.org/pdf/1809.10853.pdf)
- Mistral 7B(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.06825.pdf)
- Language models are few-shot learners(neurips, 2020)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- BLOOM: A 176B-Parameter Open-Access Multilingual Language Model(hal open science, 2023)[[paper]](https://inria.hal.science/hal-03850124/document)

### Encoder-Decoder Language Model
- Attention Is All You Need(neurips, 2017)[[paper]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer(jmlr, 2020)[[paper]](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf)
- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension(Arxiv, 2019)[[paper]](https://arxiv.org/pdf/1910.13461.pdf)

## RALM Enhancement
<img src="/fig/fig6.png" width = "800" />

### Retriever Enhancement
- Corrective Retrieval Augmented Generation(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2401.15884.pdf)
- SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.11511.pdf)
- RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01352.pdf)
- MAKING RETRIEVAL-AUGMENTED LANGUAGE MODELS ROBUST TO IRRELEVANT CONTEXT(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01558.pdf)
- RECOMP: IMPROVING RETRIEVAL-AUGMENTED LMS WITH COMPRESSION AND SELECTIVE AUGMENTATION(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.04408.pdf)
- Learning to Filter Context for Retrieval-Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2311.08377.pdf)
- Active Retrieval Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.06983.pdf)
- In-Context Retrieval-Augmented Language Models(mit, 2023)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00605/118118)
- Improving Language Models via Plug-and-Play Retrieval Feedback(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.14002.pdf)
- When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2212.10511.pdf)
- Rethinking with Retrieval: Faithful Large Language Model Inference(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2301.00303.pdf)
- DEMONSTRATE–SEARCH–PREDICT:Composing retrieval and language models for knowledge-intensive NLP(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2212.14024.pdf)

### LM Enhancement
- FiD-Light: Efficient and Effective Retrieval-Augmented Text Generation(acm, 2023)[[paper]](https://dl.acm.org/doi/pdf/10.1145/3539618.3591687)
- Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2007.01282.pdf)
- End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering(neurips, 2021)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/da3fde159d754a2555eaa198d2d105b2-Paper.pdf)
- DISTILLING KNOWLEDGE FROM READER TO RETRIEVER FOR QUESTION ANSWERING(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2012.04584.pdf)
- Scaling Instruction-Finetuned Language Models(jmlr, 2024)[[paper]](https://www.jmlr.org/papers/volume25/23-0870/23-0870.pdf)
- RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01352.pdf)
- SAIL: Search-Augmented Instruction Learning(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.15225.pdf)
- MemPrompt: Memory-assisted Prompt Editing with User Feedback(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2201.06009.pdf)
- Internet-augmented language models through few-shot prompting for open-domain question answering(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2203.05115.pdf)
- GENERALIZATION THROUGH MEMORIZATION:NEAREST NEIGHBOR LANGUAGE MODELS(Arxiv, 2020)[[paper]](https://arxiv.org/pdf/1911.00172.pdf)
- Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval(mlr, 2022)[[paper]](https://proceedings.mlr.press/v162/alon22a/alon22a.pdf)
- Efficient Nearest Neighbor Language Models(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2109.04212.pdf)
- You can’t pick your neighbors, or can you? When and how to rely on retrieval in the kNN-LM(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2210.15859.pdf)
- Training Language Models with Memory Augmentation(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2205.12674.pdf)
- IMPROVING NEURAL LANGUAGE MODELS WITH A CONTINUOUS CACHE(Arxiv, 2016)[[paper]](https://arxiv.org/pdf/1612.04426.pdf)
- Adaptive Semiparametric Language Models(mit, 2021)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00371/100688)

### Overall Enhancement
- REALM: Retrieval-Augmented Language Model Pre-Training(mlr, 2020)[[paper]](http://proceedings.mlr.press/v119/guu20a/guu20a.pdf)
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(neurips, 2020)[[paper]](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
- End-to-End Training of Neural Retrievers for Open-Domain Question Answering(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2101.00408.pdf)
- Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering(mit, 2024)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00530/114590)
- End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering(neurips, 2021)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2021/file/da3fde159d754a2555eaa198d2d105b2-Paper.pdf)
- Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory(neurips, 2023)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/887262aeb3eafb01ef0fd0e3a87a8831-Paper-Conference.pdf)
- Check Your Facts and Try Again: Improving Large Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2302.12813.pdf)

## Data Source
<img src="/fig/fig7.png" width = "800" />

### Structured Data
- Natural Questions: A Benchmark for Question Answering Research(mit, 2019)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518)
- HOTPOTQA: A Dataset for Diverse, Explainable Multi-hop Question Answering(Arxiv, 2018)[[paper]](https://arxiv.org/pdf/1809.09600.pdf)
- Wikidata5M-SI(madata, 2023)[[dataset]](https://madata.bib.uni-mannheim.de/424/)
- OGB-LSC: WIKIKG90MV2 TECHNICAL REPORT(stanford, 2023)[[paper]](https://ogb.stanford.edu/paper/neurips2022/wikikg90mv2_wikiwiki.pdf)
- OpenDialKG: Explainable Conversational Reasoning with Attention-based Walks over Knowledge Graphs(aclanthology, 2019)[[paper]](https://arxiv.org/pdf/1809.09600.pdf)

### Unstructured Data
- SQuAD: 100,000+ Questions for Machine Comprehension of Text(Arxiv, 2016)[[paper]](https://arxiv.org/pdf/1606.05250.pdf)
- FEVER: a large-scale dataset for Fact Extraction and VERification(Arxiv, 2018)[[paper]](https://arxiv.org/pdf/1803.05355.pdf)
- MULTIMODALQA: COMPLEX QUESTION ANSWERING OVER TEXT, TABLES AND IMAGES(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2104.06039.pdf)
- LAION-5B: An open large-scale dataset for training next generation image-text models(neurips, 2021)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/a1859debfb3b59d094f3504d5ebb6c25-Paper-Datasets_and_Benchmarks.pdf)
- AudioCaps: Generating Captions for Audios in The Wild(aclanthology, 2019)[[paper]](https://aclanthology.org/N19-1011.pdf)
- AUDIO SET: AN ONTOLOGY AND HUMAN-LABELED DATASET FOR AUDIO EVENTS(googleapis, 2022)[[paper]](https://pub-tools-public-publication-data.storage.googleapis.com/pdf/45857.pdf)
- Clotho: an Audio Captioning Dataset(academia, 2020)[[paper]](https://ieeexplore.ieee.org/abstract/document/9052990)
- VideoQA: question answering on news video(academia, 2003)[[paper]](https://dl.acm.org/doi/abs/10.1145/957013.957146)

## Applications
<img src="/fig/fig8.png" width = "800" />

### RALM on NLG
- Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory(neurips, 2023)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/887262aeb3eafb01ef0fd0e3a87a8831-Paper-Conference.pdf)
- Training Language Models with Memory Augmentation(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2205.12674.pdf)
- Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.03184.pdf)
- Learning to Filter Context for Retrieval-Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2311.08377.pdf)
- RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01352.pdf)
- REPLUG: Retrieval-Augmented Black-Box Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2301.12652.pdf)
- Knowledge Graph-Augmented Language Models for Knowledge-Grounded Dialogue Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.18846.pdf)

### RALM on NLU
- Robust Retrieval Augmented Generation for Zero-shot Slot Filling(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2108.13934.pdf)
- MEMORY-DRIVEN TEXT-TO-IMAGE GENERATION(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2208.07022.pdf)
- Retrieval-Augmented Diffusion Models(neurips, 2022)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/file/62868cc2fc1eb5cdf321d05b4b88510c-Paper-Conference.pdf)
- RE-IMAGEN: RETRIEVAL-AUGMENTED TEXT-TO-IMAGE GENERATOR(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2209.14491.pdf)
- KNN-Diffusion: Image Generation via Large-Scale Retrieval(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2204.02849.pdf)
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(neurips, 2020)[[paper]](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
-  Atlas: Few-shot Learning with Retrieval Augmented Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2208.03299.pdf)
-  Learning to Filter Context for Retrieval-Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2311.08377.pdf)
-  Retrieval-Enhanced Generative Model for Large-Scale Knowledge Graph Completion(ACM, 2023)[[paper]](https://dl.acm.org/doi/pdf/10.1145/3539618.3592052)
-  Active Retrieval Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.06983.pdf)
-  Learning to retrieve in-context examples for large language models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2307.07164v2.pdf)
-  Retrieval-augmented multilingual keyphrase generation with retriever-generator iterative training(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2205.10471.pdf)
-  Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.15294.pdf)

### RALM on Both NLU and NLG
- RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2310.01352.pdf)
- Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory(neurips, 2023)[[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/file/887262aeb3eafb01ef0fd0e3a87a8831-Paper-Conference.pdf)
- Learning to retrieve in-context examples for large language models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2307.07164v2.pdf)
- Retrieval-augmented multilingual keyphrase generation with retriever-generator iterative training(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2205.10471.pdf)
- Augmented Large Language Models with Parametric Knowledge Guiding(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.04757.pdf)
- Think and Retrieval: A Hypothesis Knowledge Graph Enhanced Medical Large Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2312.15883v1.pdf)
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks(neurips, 2020)[[paper]](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)
- In-Context Retrieval-Augmented Language Models(mit, 2023)[[paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00605/118118)
- Learning to Filter Context for Retrieval-Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2311.08377.pdf)
- REPLUG: Retrieval-Augmented Black-Box Language Models(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2301.12652.pdf)
- Active Retrieval Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2305.06983.pdf)
- Rethinking with Retrieval: Faithful Large Language Model Inference(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2301.00303.pdf)
- DEMONSTRATE–SEARCH–PREDICT:Composing retrieval and language models for knowledge-intensive NLP(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2212.14024.pdf)
- Retrieval Augmented Code Generation and Summarization(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2108.11601.pdf)
- When language model meets private library(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2210.17236.pdf)
- RACE: Retrieval-Augmented Commit Message Generation(Arxiv, 2022)[[paper]](https://arxiv.org/pdf/2203.02700.pdf)
- RETRIEVAL-AUGMENTED GENERATION FOR CODE SUMMARIZATION VIA HYBRID GNN(Arxiv, 2021)[[paper]](https://arxiv.org/pdf/2006.05405.pdf)

## Evaluation
<img src="/fig/fig9.png" width = "800" />

- RAGAS: Automated Evaluation of Retrieval Augmented Generation(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2309.15217.pdf)
- Benchmarking Large Language Models in Retrieval-Augmented Generation(AAAI, 2024)[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29728)
- CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2401.17043v2.pdf)
- ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2311.09476.pdf)
- Recall: A benchmark for llms robustness against external counterfactual knowledge(Arxiv, 2023)[[paper]](https://arxiv.org/pdf/2311.08147.pdf)
- Benchmarking Retrieval-Augmented Generation for Medicine(Arxiv, 2024)[[paper]](https://arxiv.org/pdf/2402.13178.pdf)

