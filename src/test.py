from opinion import Opinion
from verbatim import verbatim
from opinion_dataset import OpinionDataset


def test():
    dataset = OpinionDataset(path='../data/01_FROM_ILPS/00_OPINIONS_SAMPLE/01_CORE')
    for opinion in iter(dataset):
        for _ in verbatim(citing_opinion=opinion, search_radius=100, cited_opinions=OpinionDataset(path='../data/01_FROM_ILPS/00_OPINIONS_SAMPLE/02_CITED')):
            pass

    # Really it's only a way to use the debugger to track how the function behaves
    # but not really a unit test
    assert True


if __name__ == '__main__':
    test()
