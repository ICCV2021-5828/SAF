
from traceback import format_exc

from saf.train import *


if __name__ == '__main__':
    # train session begins
    try:
        main()
    except:
        # print(format_exc())
        # cfg.logger.exception(format_exc())
        cfg.logger.critical(format_exc())
