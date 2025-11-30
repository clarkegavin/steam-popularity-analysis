#eda/__init__.py
from .factory import EDAFactory
from .class_balance_eda import ClassBalanceEDA
from .wordcloud_eda import WordCloudEDA
from .duplicate_check_eda import DuplicateCheckEDA
from .describe_info_eda import DescribeInfoEDA
from .info_eda import InfoEDA

EDAFactory.register_eda("class_balance", ClassBalanceEDA)
EDAFactory.register_eda("wordcloud_global", lambda: WordCloudEDA(per_class=False))
EDAFactory.register_eda("wordcloud_by_class", lambda: WordCloudEDA(per_class=True))
EDAFactory.register_eda("duplicate_check", DuplicateCheckEDA)
EDAFactory.register_eda("describe_info", DescribeInfoEDA)
EDAFactory.register_eda("info", InfoEDA)



__all__ = [
    "EDAFactory",
    "ClassBalanceEDA",
    "WordCloudEDA",
    "DuplicateCheckEDA",
    "DescribeInfoEDA",
    "InfoEDA",
]
