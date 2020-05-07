# Evaluating Explainable AI
This is the codebase for "Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?" (*ACL 2020*) [[pdf]](https://arxiv.org/abs/2005.01831)  

## Repository Structure

```
|__ text/ --> Directory with all text models and experimental scripts
        |__ data/ --> includes task data and simulation test data for conducting tests
        |__ anchor/ --> Anchor code from original authors
        |__ lime/ --> LIME code from original authors
        |__ saved_models/ --> training reports for task and prototype models
        |__ src/
            |models/ --> code for neural task model and prototype model, including  decision boundary and prototype explanation methods 
            |classes/ --> supporting classes and utility functions for explanations
            |*/ --> directories for supporting classes including network layers and data loaders
        |__ figure_examples.py --> script for generating example explanations used in paper figures
        |__ gather-experiment-data.py --> script for gathering simulation test data
        |__ nearest-neighbors.py --> script for finding nearerest neighbors to prototypes
        |__ run_tagger.py --> script for evaluating classifier accuracy
        |__ requirements.txt --> package requirements for text experiments       
|__ tabular/ -->
        |__ data/ --> includes task data and simulation test data for conducting tests
        |__ anchor/ --> Anchor code from original authors
        |__ saved_models/ --> training reports for task and prototype models
        |__ src/
            |models/ --> code for neural task model and prototype model, including  decision boundary and prototype explanation methods 
            |classes/ --> supporting classes and utility functions for explanations
            |*/ --> directories for supporting classes including network layers and data loaders
        |__ gather-experiment-data.py --> script for gathering simulation test data
        |__ nearest-neighbors.py --> script for finding nearerest neighbors to prototypes
        |__ run_tagger.py --> script for evaluating classifier accuracy
        |__ requirements.txt --> package requirements for tab experiments

results_analysis.Rmd --> R markdown file that computes all empirical/statistical results in paper

```

## Requirements

- Python 3.6
- see 'requirements.txt' in each subdirectory for data domain specific requirements

## Reproducing Experiments 

1. Set-up: Install the requirements. Run `python -m spacy download en_core_web_lg`. Download the 840B.300d glove embeddings from https://nlp.stanford.edu/projects/glove/. 

2. Task models: Training both prototype and blackbox models can be done with the `main.py` scripts in the text and tabular directories. For text data, provide the glove path as `--emb-fn <file_path>`. See training reports in each `saved_model` directory for training arguments for hyperparameters to the prototype models. By default, the script trains blackbox models.

3. Simulation test data: Simulation test data is collected with `gather-experiment-data.py` in either directory, using trained neural and prototype models.

4. Statistical results: `results_analysis.Rmd` computes all empirical/statistical analysis in the paper

## Citation

@inproceedings{hase_evaluating_2020,
    title={Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?},
    author={Peter Hase and Mohit Bansal},
    booktitle={ACL},
    url={https://arxiv.org/abs/2005.01831},
    year={2020}
}


