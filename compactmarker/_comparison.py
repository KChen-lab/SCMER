from typing import Union

class Comparison:
    def __init__(self):
        """
        Methods for compare gene sets
        """
        pass

    @staticmethod
    def read_gmt(file: str, keep_description: bool = False):
        """
        Read gene set(s) in gmt format
        https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats#GMT:_Gene_Matrix_Transposed_file_format_.28.2A.gmt.29

        :param file: gmt file name/path
        :param keep_description: whether to also return the description of gene sets
        :return: genesets as in {'pathway1': [gene1, gene2, ...], 'pathway2': [gene3, gene4, ...], ...}
            (and if applicable, descriptions as in {'pathway1': 'description1', 'pathway2': 'description2', ...})
        """
        genesets = {}
        if keep_description:
            descriptions = {}
        with open(file, "r") as f:
            for i in f.readlines():
                name, description, *geneset = i.split('\t')
                genesets[name] = geneset
                if keep_description:
                    descriptions[name] = description
        if keep_description:
            return genesets, descriptions
        else:
            return genesets

    @staticmethod
    def compare(y_true: Union[list, set], y_pred: Union[list, set], for_print=True):
        """
        Compare two gene sets

        :param x: gene set 1
        :param y: gene set 2
        :return: [number of overlapping genes, number of genes in gene set, number of genes in prediction, list of overlapping genes]
        """
        y_true = set(y_true)
        y_pred = set(y_pred)
        intersect = y_true.intersection(y_pred)
        res = [len(intersect), len(y_true), len(y_pred), intersect]

        if for_print:
            return f"{res[0]}/{res[1]} recalled in the gene set by {res[2]} predicted genes, " \
                   f"overlapping genes are {', '.join(res[3])}."
        else:
            return res

    @staticmethod
    def make_recall_curve(y_true, y_pred):
        y = [0]
        x = [0]
        for i, g in enumerate(y_pred):
            if g in y_true:
                y.append(y[-1])
                y.append(y[-1] + 1)
                x.append(i - 1)
                x.append(i)
        return x, y
