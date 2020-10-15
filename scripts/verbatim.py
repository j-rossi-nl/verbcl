import webbrowser
import pyarrow.dataset as ds
import pandas as pd
import os

from courtlistener import Opinion


def main():
    cited_path = '/home/juju/PycharmProjects/courtlistener/data/01_FROM_ILPS/00_OPINIONS_SAMPLE/02_CITED'

    core_path = '/home/juju/PycharmProjects/courtlistener/data/01_FROM_ILPS/00_OPINIONS_SAMPLE/01_CORE'
    core: pd.DataFrame = ds.dataset(core_path).to_table().to_pandas().sample(frac=1.).set_index('opinion_id')
    cited: pd.DataFrame = ds.dataset(cited_path).to_table().to_pandas().set_index('opinion_id')

    assert core.index.is_unique
    assert cited.index.is_unique

    for op_id, op in core.iterrows():
        op_id: int
        op = Opinion(opinion_id=op_id, opinion_html=op['html_with_citations'])
        for v in op.verbatim():
            cit = cited.loc[v['cited_opinion_id']]
            if cit.shape[0] == 1:
                op_cit = cit.iloc[0]
                new_html = \
                    """
                    <!DOCTYPE html>
                    <html>
                    <head>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                    * {
                    box-sizing: border-box;
                    }
                    
                    /* Create two equal columns that floats next to each other */
                    .column {
                    float: left;
                    width: 50%;
                    padding: 10px;
                    }
                    
                    /* Clear floats after the columns */
                    .row:after {
                    content: "";
                    display: table;
                    clear: both;
                    }
                    </style>
                    </head>
                    <body>
                    
                    <h2>Is this Verbatim Quotation?</h2>
                    
                    <div class="row">
                    <div class="column" style="background-color:#aaa;">
                    """ \
                    + v['snippet'] + \
                    """
                    <hr>
                    """ \
                    + v['verbatim'] + \
                    """
                    </div>
                    <div class="column" style="background-color:#bbb;">
                    """ \
                    + op_cit + \
                    """
                    </div>
                    </div>
                    
                    </body>
                    </html>
                    """

                with open('temp.html', 'w') as html:
                    html.write(new_html)
                webbrowser.open_new_tab(os.path.abspath('temp.html'))
                input('Next ?')
    os.remove('temp.html')


if __name__ == '__main__':
    main()
