import numpy as np
from nose.tools import assert_equal, assert_almost_equal, ok_
from IPython.display import display, HTML, clear_output


def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")


