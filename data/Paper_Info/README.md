### Extra Information

#### `all_data_paper.txt`:
* We added year, conference/workshop information at the end of each line.
* ID is still at the 4th column of each line.

#### `paper_info.txt`:
* 3 columns: year, ID, paper title.
* Note that this file is in the same order as `all_data_paper.txt`, so it has duplicates.

To map these 2 files, simply map on both the year and the ID. 

Through this mapping process, we manually found 3 instances in our original dataset that are not official review-rebuttal passage pairs. We removed them in the files here. Now, there are 4760 passage pairs.

Let me know if you find any error in the current mapping, and if you want more information about the paper! We do have other information such as paper link on openreview / final decision / rating score, just need some time to compile those information :)
