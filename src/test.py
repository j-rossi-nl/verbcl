import pyarrow.dataset as ds

from utils import Opinion


def test_citation_to_json():
    dataset = ds.dataset('../data/01_FROM_ILPS/00_OPINIONS_SAMPLE')
    df = dataset.to_table().to_pandas()
    sample = df.iloc[0]

    opinion = Opinion(sample['opinion_id'], sample['html_with_citations'])
    for _ in opinion.to_jsonl(max_words_before_after=100):
        pass

    # Really it's only a way to use the debugger to track how the function behaves
    # but not really a unit test
    assert True


if __name__ == '__main__':
    test_citation_to_json()
