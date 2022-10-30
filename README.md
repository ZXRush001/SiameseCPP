# SiameseCPP

*Background:* Cell-Penetrating Peptides (CPPs) have received a lot of attention as a means of transporting pharmacologically active molecules into living cells without damaging cell membrane, thus holding great promise as future therapeutics. Recently, a number of machine learning algorithms have been proposed to predict CPPs. However, most of the existing prediction methods ignore to consider the agreement (disagreement) between the similar (dissimilar) CPPs and heavily depend on expert knowledge-based handcrafted features. 

*Results:* In this study, we introduce SiameseCPP, a novel deep learning framework for automated CPPs prediction that directly extracts characteristics from primary sequences. SiameseCPP constructs representations of CPPs based on a pre-trained model and a Siamese neural network made up of a Transformer and gated recurrent units. In particular, contrastive learning, for the first time, is adopted to build CPPS prediction model. Comprehensive experiments demonstrate that our proposed SiameseCPP is superior to existing baselines on CPPs prediction. Moreover, SiameseCPP also achieve good performances in other functional peptides datasets, which exhibits its satisfactory generalization ability.

## Pretrain code

You should download prot_bert_bfd （https://huggingface.co/Rostlab/prot_bert_bfd） into pretrain folder before you use, then run code in pretrain.py and save the pretrain embedding.

## Run code

```
python main.py
```
