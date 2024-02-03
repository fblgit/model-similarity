# Model Similarities

This is a small script that compares two models.

You will need:
```
pandas
numpy
torch
transformers
scikit-learn
matplotlib
seaborn
plotly
```

And you can run something like:
`python main.py --model1_path fblgit/UNA-TheBeagle-7b-v1 --model2_path udkai/Garrulus --output_file thebeagle_vs_garrulus.csv --use_cuda`

Once you have the `.csv` you can run `gen_html.py` and select the CSV to process, it will generate an `.html` file.

The script is very experimental and is intended to provide some metrics regarding the similarity between the two models. The higher similarity, the more near to the base you get.

I've uploaded a few examples but is up to you to test it further and perform your analysis.

## Experiments
Added cosine similarity, but its experimental.. feel free to contribute with more mechanisms, etc.

## Contribute
Some plots, graphs, enhancements, etc will be very welcomed. Feel free to push a PR with useful code.

## Citations
If you find the tool useful, use it for a research, etc then don't forget to add a reference to the repo.
