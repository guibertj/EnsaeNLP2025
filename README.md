# EnsaeNLP2025
Hello, here is my final submission for the "Labs + Mini-Project" track.

## Project
I have chosen to try and improve "Sentiment Analysis on Movie Reviews".
* Problematic : the vanilla classification algorithm face issues while encountering review with mixed feelings. For instance review 757 is very difficult to label either positive or negative : "I loved most of the script but did not like the ending. [...] Seeing the completed movie I have to say I am amazed. [...] I have two main problems with the movie."
* Idea : I want my algorithm to learn to discriminate reviews with mixed feelings and remove them from the accuracy computation. Indeed, I don't think the labeling "good" or "bad" makes sense for those comments and so the error of the classifier. The discriminative power of my algorithm is captured by a parameter in range [0, 1].
* Result : I will show the evolution of the Sentiment Analysis classification performance according to the discriminative power on mixed feelings review.

## Github Structure

```
EnsaeNLP2025/
├── Labs    
    ├── nlp-lab-language-models/  
    ├── nlp-lab-text-classification/  
    └── nlp-lab-text-embedding/ 
├── Mini_Project         
    ├── Code/
    │   ├── a_load.py   
    │   ├── b_model.py
    │   ├── c_mix_review.py 
    │   ├── d_visualisation.py
    │   └── main.ipynb 
    ├── Data/
    │   └── Dataset_name.txt
    └── Mini_project.pdf                
└── ReadME.md
```
