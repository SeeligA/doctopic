import logging

import ipywidgets as widgets
from colorama import Fore

out = widgets.Output(placeholder="Logging Output",
                     layout=widgets.Layout(width='100%', height='200px', border='solid', overflow='auto'))


class log_viewer(logging.Handler):
    """ Class to redistribute python logging data

    source: https://github.com/jupyter-widgets/ipywidgets/issues/1936
    """

    # have a class member to store the existing logger
    logger_instance = logging.getLogger("__name__")

    def __init__(self, *args, **kwargs):
        # Initialize the Handler
        logging.Handler.__init__(self, *args)

         # optional take format
         # setFormatter function is derived from logging.Handler
        for key, value in kwargs.items():
            if "{}".format(key) == "format":
                self.setFormatter(value)

        # make the logger send data to this class
        self.logger_instance.addHandler(self)

    def emit(self, record):
        """ Overload of logging.Handler method """

        record = self.format(record)
        with out:
            print(record)


logger = logging.getLogger(__name__)
logger.addHandler(log_viewer(format=logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')))
logger.setLevel(20)
