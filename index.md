## Scraping Reddit Data

This small project takes a look at reddit data utilizing the Reddit API and the PRAW package. Reddit is a highly popular social media website centered around smaller clumps of communities called "subreddits" centered on a particular topic - whether it's general topics like cooking or politics or even niche communities. Inevitably, this may lead to fragmented communities that become echo chambers, like r/conservative which is frequented by conservative-minded people. Alternatively, subreddits can become heavily contested by users with a variety of opinions. Reddit traditionally has a western audience, but has begun to become increasingly global in its userbase. 

Through this project, we can visualize and analyze Reddit posts and comments in a similar vein to review websites and social media. The challenge lies in sifting through the myriad of comments that do not necessarily reflect user sentiment.

Special mentions to the following guides:


### Starter
```markdown
# Packages and settings

import praw
import os
import pandas as pd
import numpy as np

pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ferrelltan/MktA-Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
