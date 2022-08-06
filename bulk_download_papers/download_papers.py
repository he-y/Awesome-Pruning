import arxiv
import wget

import csv
import time

paper_link_path = "./paper_links.txt"   # Path to paper_links.txt file
save_dir_path = r"/mnt/f/paper"     # Paper download destination path, None: current working dir

arxiv_papers = list()
openreview_papers = list()
openaccess_papers = list()
orphans = list()

with open('./paper_links.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if "#" in line or "" == line.strip():
            continue

        # Example: https://arxiv.org/abs/2105.12971
        elif "arxiv" in line:   # get arxiv ids for query
            id_ = line.split("/")[-1].strip()
            arxiv_papers.append(id_)

        # Example: https://openreview.net/forum?id=rumv7QmLUue
        elif "openreview" in line:  # get openreview pdf links
            id_ = line.split("=")[-1].strip()
            # Example: https://openreview.net/pdf?id=rumv7QmLUue
            openreview_papers.append("https://openreview.net/pdf?id=" + str(id_))

        # Example: http://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.html
        elif "openaccess" in line:
            # https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.pdf
            url = line.replace("html", "papers", 1)
            url = url.replace("html", "pdf")
            openaccess_papers.append(url)

        else:
            orphans.append(line)

print("Orphans Papers")
for item in orphans:
    print(item, end="")
print(len(orphans), "paper are not in download list")
print("Please download mannually")

# =========================== DOWNLOAD arxiv Papers with proper names ========================
time_start = time.time()
# Query the arxiv API
papers = arxiv.Search(id_list=arxiv_papers).results()
print("query_finished", time.time() - time_start)

with open('./download_arxiv_output.csv', "w") as out:
    csv_writer = csv.writer(out)
    for paper in papers:
        time_paper_start = time.time()

        title = paper.title.split(":")[0]   # purge everything after : in file name
        paper.download_pdf(dirpath=save_dir_path, filename=title + ".pdf")

        time_paper_spent = time.time() - time_paper_start
        csv_row = [paper.entry_id, title, int(time_paper_spent)]
        csv_writer.writerow(csv_row)
        print(csv_row)
print("Arxiv FINISHED, Total time spent:", int(time.time() - time_start), "seconds")

# =========================== DOWNLOAD open review/acess Papers ========================
time_start = time.time()
for paper in openreview_papers:
    print(paper)
    wget.download(paper, out = save_dir_path)

for paper in openaccess_papers:
    print(paper, end = "")
    wget.download(paper, out = save_dir_path)
    print()

print(r"OpenAccess/Review FINISHED, Total time spent:", int(time.time() - time_start), "seconds")
