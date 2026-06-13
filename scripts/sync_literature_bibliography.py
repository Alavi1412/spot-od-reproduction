from pathlib import Path
import re

bib_path = Path('paper/references.bib')
bib = bib_path.read_text(encoding='utf-8') if bib_path.exists() else ''
entries = {
'vallado2006revisiting': r'''@inproceedings{vallado2006revisiting,
  title        = {Revisiting Spacetrack Report \#3},
  author       = {Vallado, David A. and Crawford, Paul and Hujsak, Richard and Kelso, T. S.},
  booktitle    = {AIAA/AAS Astrodynamics Specialist Conference and Exhibit},
  year         = {2006},
  doi          = {10.2514/6.2006-6753},
  url          = {https://doi.org/10.2514/6.2006-6753}
}
''',
'scarselli2009graph': r'''@article{scarselli2009graph,
  title        = {The Graph Neural Network Model},
  author       = {Scarselli, Franco and Gori, Marco and Tsoi, Ah Chung and Hagenbuchner, Markus and Monfardini, Gabriele},
  journal      = {IEEE Transactions on Neural Networks},
  volume       = {20},
  number       = {1},
  pages        = {61--80},
  year         = {2009},
  doi          = {10.1109/TNN.2008.2005605},
  url          = {https://doi.org/10.1109/TNN.2008.2005605}
}
''',
'gilmer2017neural': r'''@inproceedings{gilmer2017neural,
  title        = {Neural Message Passing for Quantum Chemistry},
  author       = {Gilmer, Justin and Schoenholz, Samuel S. and Riley, Patrick F. and Vinyals, Oriol and Dahl, George E.},
  booktitle    = {Proceedings of the 34th International Conference on Machine Learning},
  pages        = {1263--1272},
  year         = {2017},
  series       = {Proceedings of Machine Learning Research},
  volume       = {70},
  url          = {https://proceedings.mlr.press/v70/gilmer17a.html}
}
''',
'battaglia2016interaction': r'''@inproceedings{battaglia2016interaction,
  title        = {Interaction Networks for Learning about Objects, Relations and Physics},
  author       = {Battaglia, Peter W. and Pascanu, Razvan and Lai, Matthew and Rezende, Danilo Jimenez and Kavukcuoglu, Koray},
  booktitle    = {Advances in Neural Information Processing Systems},
  volume       = {29},
  year         = {2016},
  url          = {https://papers.nips.cc/paper_files/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html}
}
''',
'battaglia2018relational': r'''@article{battaglia2018relational,
  title        = {Relational Inductive Biases, Deep Learning, and Graph Networks},
  author       = {Battaglia, Peter W. and Hamrick, Jessica B. and Bapst, Victor and Sanchez-Gonzalez, Alvaro and Zambaldi, Vinicius and Malinowski, Mateusz and Tacchetti, Andrea and Raposo, David and Santoro, Adam and Faulkner, Ryan and Gulcehre, Caglar and Song, Francis and Ballard, Andrew and Gilmer, Justin and Dahl, George and Vaswani, Ashish and Allen, Kelsey and Nash, Charles and Langston, Victoria and Dyer, Chris and Heess, Nicolas and Wierstra, Daan and Kohli, Pushmeet and Botvinick, Matthew and Vinyals, Oriol and Li, Yujia and Pascanu, Razvan},
  journal      = {arXiv preprint arXiv:1806.01261},
  year         = {2018},
  url          = {https://arxiv.org/abs/1806.01261}
}
''',
'sanchezgonzalez2020learning': r'''@inproceedings{sanchezgonzalez2020learning,
  title        = {Learning to Simulate Complex Physics with Graph Networks},
  author       = {Sanchez-Gonzalez, Alvaro and Godwin, Jonathan and Pfaff, Tobias and Ying, Rex and Leskovec, Jure and Battaglia, Peter W.},
  booktitle    = {Proceedings of the 37th International Conference on Machine Learning},
  pages        = {8459--8468},
  year         = {2020},
  series       = {Proceedings of Machine Learning Research},
  volume       = {119},
  url          = {https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html}
}
''',
'chen2018neural': r'''@inproceedings{chen2018neural,
  title        = {Neural Ordinary Differential Equations},
  author       = {Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David K.},
  booktitle    = {Advances in Neural Information Processing Systems},
  volume       = {31},
  year         = {2018},
  url          = {https://papers.nips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html}
}
''',
'kipf2018neural': r'''@inproceedings{kipf2018neural,
  title        = {Neural Relational Inference for Interacting Systems},
  author       = {Kipf, Thomas N. and Fetaya, Ethan and Wang, Kuan-Chieh and Welling, Max and Zemel, Richard},
  booktitle    = {Proceedings of the 35th International Conference on Machine Learning},
  pages        = {2688--2697},
  year         = {2018},
  series       = {Proceedings of Machine Learning Research},
  volume       = {80},
  url          = {https://proceedings.mlr.press/v80/kipf18a.html}
}
''',
}
changed = False
for key, entry in entries.items():
    if not re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,', bib, flags=re.IGNORECASE):
        if bib and not bib.endswith('\n'):
            bib += '\n'
        bib += '\n' + entry
        changed = True
if changed:
    bib_path.write_text(bib, encoding='utf-8')
print('literature_bib_sync=ok')
print('keys=' + ','.join(entries))
