# just some analysis on image log
import pandas as pd

imlogfp = r"\\132.239.170.55\Users\User\Desktop\image_log.txt"
df = pd.read_csv(imlogfp, sep='\t', parse_dates=[1])
