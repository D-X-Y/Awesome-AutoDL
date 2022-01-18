#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2022.01 #
########################################################################################
# python -m awesome_autodl.bins.list_email
########################################################################################
import argparse
from awesome_autodl import autodl_topic2papers


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analysis the AutoDL papers.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="The path to save the final directory.",
    )
    args = parser.parse_args()

    author_email_title = []
    topic2papers = autodl_topic2papers()
    for topic, papers in topic2papers.items():
        # print(f'Collect {len(papers)} papers for "{topic}"')
        for paper in papers:
            for author, email in paper.author_email.items():
                print(f"{author:25s}, {email:15s} : {paper.title}")
                author_email_title.append((author, email, paper.title))
    print(f"There are {len(author_email_title)} items in total.")
