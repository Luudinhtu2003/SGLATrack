
import os


SEQ_PATH = r"/home/getac2/tuld3/tu_workspace/track_uav/SGLATrack/data/UAV123/data_seq/UAV123"

list_sequences = os.listdir(SEQ_PATH)
print("UAV123 sequences: ", list_sequences)

import re

unique_classes = sorted({
    re.match(r'[a-zA-Z]+', name).group()
    for name in list_sequences
})

print(list(unique_classes))
import re

with open("mapping.txt", "w") as f:
    for name in list_sequences:
        cls = re.match(r'[a-zA-Z]+', name).group()
        f.write(f"{name}\t{cls}\n")