# EnsaeNLP2025
Hello, here is my final submission for the "Labs + Mini-Project" track.

## Project
I have chosen to try and improve "Sentiment Analysis on Movie Reviews".
* Problematic : the vanilla classification algorithm face issues while encountering review with mixed feelings. For instance review 757 is very difficult to label either positive or negative : "I loved most of the script but did not like the ending. [...] Seeing the completed movie I have to say I am amazed. [...] I have two main problems with the movie."
* Idea : I want my algorithm to learn to discriminate reviews with mixed feelings and remove them from the accuracy computation. The discriminative power of my algorithm is captured by a parameter in range [0, 1].
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
    │   ├── dxf_converter.py   
    │   ├── dxf_processing.py  
    │   └── dxf_utilities.py   
    ├── Data/
    │   └── dxf_converter.py
    ├── Images/
    └── Project.pdf                
└── ReadME.md
```
